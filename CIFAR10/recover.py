import argparse
import csv
import os
import glob
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

# Adapt imports to your project layout
from utils import get_loaders, evaluate_pgd, evaluate_standard
from preact_resnet import PreActResNet18
from resnet import ResNet18

# ---------------------------
# Helpers
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="train_pgd_output", help="Output dir used by training (checkpoints + metrics.csv)")
    p.add_argument("--data-dir", type=str, default="../../cifar-data", help="Path passed to get_loaders() during training")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size to use for reconstructing metrics")
    p.add_argument("--attack-iters-train", type=int, default=7, help="PGD iters to use when recomputing train robust metrics")
    p.add_argument("--attack-iters-test", type=int, default=20, help="PGD iters to use when recomputing test robust metrics")
    p.add_argument("--restarts", type=int, default=1, help="PGD restarts for robust eval")
    p.add_argument("--epochs", type=int, nargs="*", default=None, help="Specific epoch numbers to recover (e.g. 10 21 31). If omitted, scans checkpoints.")
    p.add_argument("--model", type=str, default="PreActResNet18", choices=["PreActResNet18", "ResNet18"], help="Model architecture used for training")
    p.add_argument("--overwrite", action="store_true", help="If set, overwrite existing epoch rows in metrics.csv. Otherwise only insert missing epochs.")
    p.add_argument("--recover-train-metrics", action="store_true", help="If set, attempt to recover train_* metrics using train_subset_indices (if present in checkpoint).")
    p.add_argument("--use-train-mode", action="store_true", help="If set, evaluate train-subset with model.train() instead of model.eval() (not recommended; non-deterministic).")
    return p.parse_args()

def find_checkpoints(out_dir):
    pattern = os.path.join(out_dir, "checkpoint_epoch_*.pth")
    paths = glob.glob(pattern)
    out = {}
    for p in paths:
        name = os.path.basename(p)
        try:
            # parse epoch number
            mid = name.split("checkpoint_epoch_")[1]
            epoch = int(mid.split(".pth")[0])
            out[epoch] = p
        except Exception:
            continue
    return out  # dict: epoch -> path

def read_existing_metrics(metrics_csv):
    if not os.path.exists(metrics_csv):
        return {}, None  # empty plus no header
    with open(metrics_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = {}
        for r in reader:
            try:
                e = int(r.get("epoch", r.get("Epoch", -1)))
            except Exception:
                continue
            rows[e] = r
        return rows, header

def backup_file(path):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak = f"{path}.bak.{ts}"
    Path(bak).write_bytes(Path(path).read_bytes())
    return bak

def create_model(name, device):
    if name.lower().startswith("preact"):
        model = PreActResNet18()
    else:
        model = ResNet18()
    model = nn.DataParallel(model).to(device)
    return model

def safe_cell(x):
    return "" if x is None else f"{x:.6f}"

# ---------------------------
# Main recovery routine
# ---------------------------
def main():
    args = parse_args()
    out_dir = args.out_dir
    metrics_csv = os.path.join(out_dir, "metrics.csv")
    recovered_temp_csv = os.path.join(out_dir, "recovered_rows.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get checkpoint map
    ckpt_map = find_checkpoints(out_dir)
    if not ckpt_map:
        print(f"[ERROR] No checkpoints found in {out_dir}. Exiting.")
        return

    # determine epochs to process
    if args.epochs and len(args.epochs) > 0:
        epochs_to_try = sorted(set(args.epochs))
    else:
        epochs_to_try = sorted(ckpt_map.keys())

    # read existing metrics
    existing_rows, header = read_existing_metrics(metrics_csv)
    if header is None:
        # create header matching training script
        header = [
            "epoch",
            "wall_time_train",
            "wall_time_epoch",
            "lr",
            "train_loss",
            "train_acc",
            "train_robust_loss",
            "train_robust_acc",
            "test_loss",
            "test_acc",
            "test_robust_loss",
            "test_robust_acc",
        ]

    # ensure 'recovered' flag column
    if "recovered" not in header:
        header = header + ["recovered"]

    # backup original metrics.csv
    if os.path.exists(metrics_csv):
        bak = backup_file(metrics_csv)
        print(f"[INFO] Backed up original metrics.csv to {bak}")

    # build loaders (we need the full train dataset to build subset loader)
    train_loader_full, test_loader = get_loaders(args.data_dir, args.batch_size)
    train_dataset = train_loader_full.dataset

    recovered_rows = {}  # epoch -> dict(row)

    # iterate epochs
    for epoch in epochs_to_try:
        if (not args.overwrite) and (epoch in existing_rows):
            print(f"[SKIP] Epoch {epoch} already present in metrics.csv (use --overwrite to replace).")
            continue

        ckpt_path = ckpt_map.get(epoch, None)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"[WARN] No checkpoint file found for epoch {epoch}; skipping.")
            continue

        print(f"[INFO] Recovering epoch {epoch} from {ckpt_path} ...")

        # load model
        model = create_model(args.model, device)
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint {ckpt_path}: {e}")
            continue

        if "model_state" not in ckpt:
            print(f"[WARN] checkpoint {ckpt_path} has no model_state; skipping.")
            continue

        try:
            model.load_state_dict(ckpt["model_state"])
        except Exception as e:
            print(f"[WARN] model.load_state_dict() raised: {e} — attempting strict=False")
            try:
                model.load_state_dict(ckpt["model_state"], strict=False)
            except Exception as e2:
                print(f"[ERROR] Fallback load also failed: {e2}; skipping epoch {epoch}")
                continue

        model.eval()

        # Recover lr if possible by creating opt and loading opt_state
        recovered_lr = ""
        try:
            # Create a compatible optimizer skeleton (hyperparams not critical for load)
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            if "opt_state" in ckpt:
                try:
                    opt.load_state_dict(ckpt["opt_state"])
                    recovered_lr = opt.param_groups[0].get("lr", "")
                except Exception as e:
                    print(f"[WARN] Could not load optimizer state to recover lr: {e}")
                    recovered_lr = ""
            else:
                recovered_lr = ""
        except Exception as e:
            print(f"[WARN] lr recovery failed: {e}")
            recovered_lr = ""

        # Recover test metrics (robust + clean)
        test_loss = test_acc = test_robust_loss = test_robust_acc = None
        try:
            with torch.enable_grad():
                pgd_loss, pgd_acc, pgd_n = evaluate_pgd(test_loader, model, args.attack_iters_test, args.restarts)
                test_robust_loss = pgd_loss
                test_robust_acc = pgd_acc
        except Exception as e:
            print(f"[WARN] evaluate_pgd(test) failed for epoch {epoch}: {e}")
            test_robust_loss = test_robust_acc = None

        try:
            with torch.no_grad():
                t_loss, t_acc, t_n = evaluate_standard(test_loader, model)
                test_loss = t_loss
                test_acc = t_acc
        except Exception as e:
            print(f"[WARN] evaluate_standard(test) failed for epoch {epoch}: {e}")
            test_loss = test_acc = None

        # Optionally recover train metrics using train_subset_indices
        train_loss = train_acc = train_robust_loss = train_robust_acc = None
        if args.recover_train_metrics:
            subset_indices = ckpt.get("train_subset_indices", None)
            if subset_indices is None:
                print(f"[INFO] No train_subset_indices present in checkpoint for epoch {epoch}; skipping train-metrics recovery.")
            else:
                # build loader for subset (Sampler)
                sampler = SubsetRandomSampler(list(subset_indices))
                # match training options roughly
                train_subset_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

                # If user requested train-mode evaluation, toggle model.train()
                eval_mode = True
                if args.use_train_mode:
                    print("[WARN] Evaluating train-subset in model.train() mode (results may differ and be non-deterministic).")
                    model.train()
                    eval_mode = False
                else:
                    model.eval()
                    eval_mode = True

                # robust on train subset
                try:
                    with torch.enable_grad():
                        tr_pgd_loss, tr_pgd_acc, tr_pgd_n = evaluate_pgd(train_subset_loader, model, args.attack_iters_train, args.restarts)
                        train_robust_loss = tr_pgd_loss
                        train_robust_acc = tr_pgd_acc
                except Exception as e:
                    print(f"[WARN] evaluate_pgd(train-subset) failed for epoch {epoch}: {e}")
                    train_robust_loss = train_robust_acc = None

                # clean on train subset
                try:
                    with torch.no_grad():
                        tr_loss, tr_acc, tr_n = evaluate_standard(train_subset_loader, model)
                        train_loss = tr_loss
                        train_acc = tr_acc
                except Exception as e:
                    print(f"[WARN] evaluate_standard(train-subset) failed for epoch {epoch}: {e}")
                    train_loss = train_acc = None

                # restore eval train state
                if not eval_mode:
                    model.eval()

        # compose row dict using header order
        row = {}
        # fill core fields used in training CSV
        row["epoch"] = str(epoch)
        row["wall_time_train"] = ""
        row["wall_time_epoch"] = ""
        row["lr"] = (str(recovered_lr) if recovered_lr != "" else "")
        row["train_loss"] = safe_cell(train_loss)
        row["train_acc"] = safe_cell(train_acc)
        row["train_robust_loss"] = safe_cell(train_robust_loss)
        row["train_robust_acc"] = safe_cell(train_robust_acc)
        row["test_loss"] = safe_cell(test_loss)
        row["test_acc"] = safe_cell(test_acc)
        row["test_robust_loss"] = safe_cell(test_robust_loss)
        row["test_robust_acc"] = safe_cell(test_robust_acc)
        row["recovered"] = "True"

        # ensure header keys exist in row (some older CSVs may have different capitalization)
        # normalize to header names
        normalized_row = {}
        for h in header:
            normalized_row[h] = row.get(h, "")

        recovered_rows[epoch] = normalized_row
        print(f"[OK] Recovered epoch {epoch}: test_acc={test_acc}, test_robust_acc={test_robust_acc}, train_robust_acc={train_robust_acc}")

    # Merge recovered rows into existing_rows (without overwriting unless --overwrite)
    merged = dict(existing_rows)  # copy
    for e, r in recovered_rows.items():
        if (e in merged) and (not args.overwrite):
            print(f"[SKIP-MERGE] Existing row for epoch {e} preserved (use --overwrite to replace).")
            continue
        merged[e] = r

    if not merged:
        print("[INFO] No rows to write to metrics.csv. Exiting.")
        return

    # write merged CSV to temp file then replace original
    tmp_out = os.path.join(out_dir, f"metrics_merged_{int(time.time())}.csv")
    with open(tmp_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for epoch in sorted(merged.keys()):
            row = merged[epoch]
            # ensure every header present
            write_row = {h: row.get(h, "") for h in header}
            writer.writerow(write_row)

    # overwrite metrics.csv (we already backed up)
    Path(tmp_out).replace(metrics_csv)
    print(f"[DONE] Wrote merged metrics to {metrics_csv} (backup created earlier).")

if __name__ == "__main__":
    main()
