# train_pgd.py
import argparse
import logging
import os
import time
import csv
from pathlib import Path

import numpy as np
import random as pyrandom
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import apex.amp as amp
except Exception:
    amp = None
from torch.utils.data import DataLoader, SubsetRandomSampler

# models
from preact_resnet import PreActResNet18
from resnet import ResNet18

# utils (must include normalize, std, clamp, get_loaders, evaluate_pgd, evaluate_standard)
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
                   evaluate_pgd, evaluate_standard, normalize)

logger = logging.getLogger(__name__)

# helper: convert python scalar to tensor on the same device/dtype as ref_tensor
def _scalar_on_device(scalar, ref_tensor):
    # if scalar is already a tensor, just convert dtype/device without copying raw bytes unnecessarily
    if torch.is_tensor(scalar):
        return scalar.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    # otherwise create a tensor on the requested device/dtype
    return torch.tensor(float(scalar), dtype=ref_tensor.dtype, device=ref_tensor.device)


# -------------------------
# Atomic checkpoint helpers (save model+opt+rng+subset indices)
# -------------------------
def save_full_checkpoint(path, model_state, opt_state, epoch=0, batch_idx=0,
                         best_test=-1.0, train_subset_indices=None, logger=None, extra=None):
    state = {
        'epoch': int(epoch),
        'batch_idx': int(batch_idx),
        'model_state': model_state,
        'opt_state': opt_state,
        'best_test_robust_acc': best_test,
        'train_subset_indices': None if train_subset_indices is None else list(train_subset_indices),
        'rng_numpy': np.random.get_state(),
        'rng_python': pyrandom.getstate(),
        'rng_torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['rng_cuda_all'] = torch.cuda.get_rng_state_all()
    if extra is not None:
        state['extra'] = extra
    Path(os.path.dirname(path) or '.').mkdir(parents=True, exist_ok=True)
    tmp = path + '.tmp'
    torch.save(state, tmp)
    Path(tmp).replace(path)
    if logger is not None:
        logger.info(f"Saved checkpoint: {path}")

def safe_load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device)


# -------------------------
# deterministic subsample (match AWP)
# -------------------------
def build_train_loader_with_fraction(train_loader_full, train_fraction, seed, batch_size):
    """
    If train_fraction < 1.0, deterministically choose first k indices out of a permutation seeded by `seed`.
    Returns (loader, train_subset_indices_or_None)
    """
    if train_fraction >= 1.0:
        return train_loader_full, None

    dataset = train_loader_full.dataset
    n = len(dataset)
    k = max(1, int(n * train_fraction))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    subset_indices = perm[:k]
    sampler = SubsetRandomSampler(subset_indices)
    num_workers = getattr(train_loader_full, 'num_workers', 2)
    pin_memory = getattr(train_loader_full, 'pin_memory', True)
    drop_last = getattr(train_loader_full, 'drop_last', False)
    subset_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                               num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    return subset_loader, subset_indices


# -------------------------
# arguments
# -------------------------
def get_args():
    parser = argparse.ArgumentParser()
    # dataset / training
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-test', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep', 'piecewise', 'cosine'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--resume-from', default='', type=str,
                    help='Path to checkpoint to resume from')

    # adversarial params
    parser.add_argument('--epsilon', default=8, type=int, help='epsilon in pixel space (0-255)')
    parser.add_argument('--attack-iters-train', default=7, type=int, help='PGD iterations during training')
    parser.add_argument('--attack-iters-test', default=20, type=int, help='PGD iterations during testing/eval')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2.0, type=float, help='PGD step size in pixel space')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'])

    # experiment / misc
    parser.add_argument('--out-dir', default='train_pgd_output', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'])
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'])
    parser.add_argument('--master-weights', action='store_true')

    # compatibility with AWP CLI
    parser.add_argument('--model', default='PreActResNet18', type=str)
    parser.add_argument('--train-fraction', type=float, default=1.0)

    return parser.parse_args()

EVAL_EVERY = 5
QUICK_PGD_ITERS = 3

# -------------------------
# main training
# -------------------------
def main():
    args = get_args()

    # out dir + logging
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    # also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logger.info(args)

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    pyrandom.seed(args.seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders
    train_loader_full, test_loader = get_loaders(args.data_dir, args.batch_size)
    train_loader, train_subset_indices = build_train_loader_with_fraction(train_loader_full, args.train_fraction, args.seed, args.batch_size)
    if train_subset_indices is not None:
        logger.info(f"Using {len(train_subset_indices)} / {len(train_loader_full.dataset)} training samples (seed={args.seed})")
        np.save(os.path.join(args.out_dir, 'train_subset_indices.npy'), train_subset_indices)

    # convert eps/alpha to normalized units (std from utils)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.pgd_alpha / 255.) / std

    epsilon = epsilon.to(device=device, dtype=torch.float32)
    alpha = alpha.to(device=device, dtype=torch.float32)

    # build model
    model_name = args.model.lower()
    if model_name in ['preactresnet18', 'preact_resnet18', 'preactresnet']:
        model = PreActResNet18()
    elif model_name in ['resnet18', 'resnet']:
        model = ResNet18()
    else:
        logger.warning("Unknown model %s; defaulting to PreActResNet18", args.model)
        model = PreActResNet18()

    # wrap as AWP does and move to device
    model = nn.DataParallel(model).to(device)

    model.train()

    # optimizer + AMP
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    if amp is not None and args.opt_level != 'O0':
        model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    # LR scheduler: keep similar logic to earlier scripts (cyclic or multistep fallback)
    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                     step_size_up=max(1, lr_steps // 2), step_size_down=max(1, lr_steps // 2))
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[max(1, lr_steps // 2), max(1, lr_steps * 3 // 4)], gamma=0.1)

    # -------------------------
    # Resume-from checkpoint (epoch-level)
    # -------------------------
    start_epoch = 0
    if args.resume_from:
        ckpt_path = args.resume_from
        if os.path.exists(ckpt_path):
            logger.info(f"Attempting to resume from checkpoint: {ckpt_path}")
            try:
                ckpt = safe_load_checkpoint(ckpt_path, device=device)
                # load model state (permissive)
                if 'model_state' in ckpt:
                    try:
                        model.load_state_dict(ckpt['model_state'])
                        logger.info("Model weights loaded from checkpoint.")
                    except Exception as e:
                        try:
                            model.load_state_dict(ckpt['model_state'], strict=False)
                            logger.warning(f"Model loaded with strict=False: {e}")
                        except Exception as e2:
                            logger.warning(f"Failed to load model_state from checkpoint: {e2}")

                # load optimizer if present
                if 'opt_state' in ckpt:
                    try:
                        opt.load_state_dict(ckpt['opt_state'])
                        logger.info("Optimizer state restored from checkpoint.")
                    except Exception as e:
                        logger.warning(f"Could not restore optimizer state: {e}")

                # restore RNGs (best-effort)
                try:
                    if 'rng_numpy' in ckpt:
                        np.random.set_state(ckpt['rng_numpy'])
                    if 'rng_python' in ckpt:
                        pyrandom.setstate(ckpt['rng_python'])
                    if 'rng_torch' in ckpt:
                        rt = ckpt['rng_torch']
                        # handle tensor / numpy array or list
                        if isinstance(rt, torch.Tensor):
                            torch.set_rng_state(rt)
                        else:
                            try:
                                torch.set_rng_state(torch.tensor(rt, dtype=torch.uint8))
                            except Exception:
                                logger.warning("Could not restore CPU RNG from checkpoint (type mismatch).")
                    if torch.cuda.is_available() and 'rng_cuda_all' in ckpt:
                        try:
                            cuda_states = []
                            for s in ckpt['rng_cuda_all']:
                                if isinstance(s, torch.Tensor):
                                    cuda_states.append(s)
                                else:
                                    cuda_states.append(torch.tensor(s, dtype=torch.uint8))
                            torch.cuda.set_rng_state_all(cuda_states)
                        except Exception:
                            logger.warning("Could not restore CUDA RNG states from checkpoint.")
                except Exception as e:
                    logger.warning(f"RNG restore failed: {e}")

                # restore bookkeeping
                start_epoch = int(ckpt.get('epoch', 0)) + 1
                if 'train_subset_indices' in ckpt and ckpt['train_subset_indices'] is not None:
                    # replace train_subset_indices if checkpoint has it (keeps training subset consistent)
                    train_subset_indices = ckpt['train_subset_indices']
                    np.save(os.path.join(args.out_dir, 'train_subset_indices_restored.npy'), train_subset_indices)
                    logger.info(f"Restored train subset of size {len(train_subset_indices)} from checkpoint")
                logger.info(f"Resuming training from epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {ckpt_path}: {e}")
        else:
            logger.warning(f"Requested resume-from path does not exist: {ckpt_path}")

    # metrics CSV header (match AWP)
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "wall_time_train",    # seconds spent in training loop this epoch
                "wall_time_epoch",    # seconds spent in evaluation / rest of epoch
                "lr",
                "train_loss",
                "train_acc",
                "train_robust_loss",
                "train_robust_acc",
                "test_loss",
                "test_acc",
                "test_robust_loss",
                "test_robust_acc",
            ])


    # Training loop
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    start_train_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        start_epoch_time = time.time()
        train_loss = 0.0
        train_acc = 0
        train_robust_loss = 0.0
        train_robust_acc = 0
        train_n = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            # move inputs to the correct device
            X = X.to(device)
            y = y.to(device)

            # initialize adversarial delta on same device as X
            delta = torch.zeros_like(X).to(X.device)
            if args.delta_init == 'random':
                # epsilon is per-channel tensor; uniform in [-eps, eps] per channel
                for c in range(len(epsilon)):
                    delta[:, c, :, :].uniform_(-epsilon[c][0][0].item(), epsilon[c][0][0].item())
                delta.data = clamp(delta, (_scalar_on_device(lower_limit, X) - X), (_scalar_on_device(upper_limit, X) - X))
            delta.requires_grad = True

            # perform PGD (training iterations)
            iters_train = args.attack_iters_train
            for _ in range(iters_train):
                # forward through model with normalization (matches AWP)
                adv_in = normalize(torch.clamp(X + delta, min=lower_limit.to(X.device), max=upper_limit.to(X.device)))
                output = model(adv_in)
                loss = criterion(output, y)
                if amp is not None and args.opt_level != 'O0':
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data = clamp(delta, (_scalar_on_device(lower_limit, X) - X), (_scalar_on_device(upper_limit, X) - X))
                delta.grad.zero_()

            # finalize delta and train on adversarial example
            delta = delta.detach()
            adv_in = normalize(torch.clamp(X + delta, min=lower_limit.to(X.device), max=upper_limit.to(X.device)))
            output = model(adv_in)
            loss = criterion(output, y)
            opt.zero_grad()
            if amp is not None and args.opt_level != 'O0':
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            opt.step()

            # track metrics (train_robust tracked same as robust step)
            train_robust_loss += loss.item() * y.size(0)
            train_robust_acc += (output.max(1)[1] == y).sum().item()

            # also evaluate clean forward for training stats (normalized)
            clean_out = model(normalize(X))
            train_loss += (criterion(clean_out, y).item()) * y.size(0)
            train_acc += (clean_out.max(1)[1] == y).sum().item()

            train_n += y.size(0)

            scheduler.step()

        # end epoch
        epoch_time = time.time() - start_epoch_time
        lr = opt.param_groups[0]['lr']
        train_loss_avg = (train_loss / train_n) if train_n else None
        train_acc_avg = (train_acc / train_n) if train_n else None
        train_robust_loss_avg = (train_robust_loss / train_n) if train_n else None
        train_robust_acc_avg = (train_robust_acc / train_n) if train_n else None

        logger.info('%d \t %.1f \t \t %.6f \t %.4f \t %.4f', epoch, epoch_time, lr,
                    train_loss_avg if train_loss_avg is not None else 0.0,
                    train_acc_avg if train_acc_avg is not None else 0.0)

        # save per-epoch full checkpoint (atomic)
        ckpt_path = os.path.join(args.out_dir, f'checkpoint_epoch_{epoch}.pth')
        save_full_checkpoint(ckpt_path, model.state_dict(), opt.state_dict(),
                             epoch=epoch, batch_idx=0,
                             best_test=-1.0, train_subset_indices=train_subset_indices, logger=logger)

        # Evaluation (test with stronger PGD)
        model.eval()
        test_model = model

        # evaluate robustly (evaluate_pgd signature now returns avg_loss, avg_acc, n)
        with torch.no_grad():
            if epoch % EVAL_EVERY == 0 or epoch == args.epochs - 1:
                # full evaluation
                iters_test = args.attack_iters_test
            else:
                # cheap evaluation
                iters_test = QUICK_PGD_ITERS
            
            pgd_loss, pgd_acc, pgd_n = evaluate_pgd(
                test_loader, test_model, iters_test, args.restarts
            )
            test_loss, test_acc, test_n = evaluate_standard(
                test_loader, test_model
            )

        # record which PGD was used this epoch (don't change existing metrics file)
        protocol_path = os.path.join(args.out_dir, 'eval_protocol.log')
        line = f"epoch={epoch}, mode={'FULL' if is_full_eval else 'QUICK'}, pgd_test_iters={iters_test}, pgd_restarts={args.restarts}, timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # print to console for immediate feedback
        print("EVAL PROTOCOL:", line.strip())
        
        # append to protocol log (creates the file if it doesn't exist)
        with open(protocol_path, 'a') as pf:
            pf.write(line)

        # compute times
        train_end_time = time.time()
        test_time = time.time()
        wall_time_train = train_end_time - start_epoch_time
        wall_time_epoch = test_time - train_end_time

        # safe average helpers
        def safe_avg(val, n):
            return float(val) if (n and n > 0) else None

        train_loss_avg = safe_avg(train_loss / train_n if train_n else None, train_n)
        train_acc_avg = safe_avg(train_acc / train_n if train_n else None, train_n)
        train_robust_loss_avg = safe_avg(train_robust_loss / train_n if train_n else None, train_n)
        train_robust_acc_avg = safe_avg(train_robust_acc / train_n if train_n else None, train_n)

        # test averages already computed by eval funcs; guard if their n==0
        test_loss_avg = test_loss if test_n else None
        test_acc_avg = test_acc if test_n else None
        test_robust_loss_avg = pgd_loss if pgd_n else None
        test_robust_acc_avg = pgd_acc if pgd_n else None

        # log nicely
        logger.info(
            '%d \t %.1f \t %.1f \t %.6f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch,
            wall_time_train,
            wall_time_epoch,
            lr,
            train_loss_avg if train_loss_avg is not None else float('nan'),
            train_acc_avg if train_acc_avg is not None else float('nan'),
            train_robust_loss_avg if train_robust_loss_avg is not None else float('nan'),
            train_robust_acc_avg if train_robust_acc_avg is not None else float('nan'),
            test_loss_avg if test_loss_avg is not None else float('nan'),
            test_acc_avg if test_acc_avg is not None else float('nan'),
            test_robust_loss_avg if test_robust_loss_avg is not None else float('nan'),
            test_robust_acc_avg if test_robust_acc_avg is not None else float('nan'),
        )

        # write CSV row
        def cell(x):
            return "" if x is None else f"{x:.6f}"

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{wall_time_train:.6f}",
                f"{wall_time_epoch:.6f}",
                f"{lr:.6f}",
                cell(train_loss_avg),
                cell(train_acc_avg),
                cell(train_robust_loss_avg),
                cell(train_robust_acc_avg),
                cell(test_loss_avg),
                cell(test_acc_avg),
                cell(test_robust_loss_avg),
                cell(test_robust_acc_avg),
            ])


        model.train()

if __name__ == "__main__":
    main()
