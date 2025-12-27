# utils.py (device-agnostic version)

try:
    import apex.amp as amp
except ImportError:
    amp = None
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

# --- device-safe CIFAR mean/std helpers ---
# keep CPU tensors at import time; move to device/dtype when used
_cifar10_mean = torch.tensor(cifar10_mean, dtype=torch.float32).view(3, 1, 1)
_cifar10_std  = torch.tensor(cifar10_std,  dtype=torch.float32).view(3, 1, 1)

# expose mu/std names for backward-compatibility (still CPU)
mu = _cifar10_mean
std = _cifar10_std

def get_mu(device=None, dtype=None):
    """Return CIFAR mean on requested device/dtype (or cpu if None)."""
    if device is None and dtype is None:
        return _cifar10_mean
    return _cifar10_mean.to(device=device, dtype=dtype)

def get_std(device=None, dtype=None):
    """Return CIFAR std on requested device/dtype (or cpu if None)."""
    if device is None and dtype is None:
        return _cifar10_std
    return _cifar10_std.to(device=device, dtype=dtype)

def normalize(X):
    """
    Normalize tensor X with CIFAR mean/std on the same device and dtype as X.
    Usage: X_norm = normalize(X)
    """
    mu_t = get_mu(device=X.device, dtype=X.dtype)
    std_t = get_std(device=X.device, dtype=X.dtype)
    return (X - mu_t) / std_t
# --- end block ---

# upper and lower limits are device-agnostic CPU tensors; callers should move to X.device when needed
# keep as CPU tensors for compatibility; clamp helper will move/compare on same device as arguments
upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)

def clamp(X, lower_limit, upper_limit):
    """
    Clamp X elementwise between lower_limit and upper_limit.
    lower_limit / upper_limit may be tensors or scalars; they should already be on the same device as X.
    """
    # ensure limits are tensors on the same device/dtype as X
    if not torch.is_tensor(lower_limit):
        lower_limit = torch.tensor(lower_limit, dtype=X.dtype, device=X.device)
    else:
        lower_limit = lower_limit.to(device=X.device, dtype=X.dtype)
    if not torch.is_tensor(upper_limit):
        upper_limit = torch.tensor(upper_limit, dtype=X.dtype, device=X.device)
    else:
        upper_limit = upper_limit.to(device=X.device, dtype=X.dtype)
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,   # safer for CPU runs
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=2,
    )
    return train_loader, test_loader


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    """
    Device-aware PGD attack used for training/evaluation.
    - model: expects normalized inputs, so we call model(normalize(X + delta))
    - epsilon and alpha are expected to be per-channel tensors (shape (3,1,1)) or scalars.
    """
    device = X.device
    max_loss = torch.zeros(y.shape[0], device=device, dtype=torch.float32)
    max_delta = torch.zeros_like(X, device=device)

    for _ in range(restarts):
        # initialize delta on same device as X
        delta = torch.zeros_like(X, device=device)
        # random init per-channel if epsilon is tensor-like
        if torch.is_tensor(epsilon):
            for c in range(epsilon.shape[0]):
                eps_val = float(epsilon[c].view(-1)[0].item()) if epsilon[c].numel() == 1 else float(epsilon[c].view(-1)[0].item())
                delta[:, c, :, :].uniform_(-eps_val, eps_val)
        else:
            delta.uniform_(-float(epsilon), float(epsilon))

        delta = clamp(delta, (lower_limit.to(device)), (upper_limit.to(device)))
        delta.requires_grad_(True)

        for _ in range(attack_iters):
            # forward on normalized inputs
            output = model(normalize(torch.clamp(X + delta, min=lower_limit.to(device), max=upper_limit.to(device))))
            # early stop indices: those already misclassified? (keep same semantics as original)
            index = torch.where(output.max(1)[1] == y)[0]
            if index.numel() == 0:
                break
            loss = F.cross_entropy(output, y)
            # don't use amp scaling here; evaluations and inner attacks are fine in FP32
            loss.backward()
            grad = delta.grad.detach()
            # apply step on selected indices
            d = delta[index, :, :, :].clone()
            g = grad[index, :, :, :].clone()
            if torch.is_tensor(alpha):
                step = alpha
            else:
                step = float(alpha)
            d = clamp(d + step * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, (lower_limit.to(device) - X[index, :, :, :]), (upper_limit.to(device) - X[index, :, :, :]))
            with torch.no_grad():
                delta[index, :, :, :] = d
            delta.grad = None

        # compute loss for this restart and keep best per-sample
        with torch.no_grad():
            all_loss = F.cross_entropy(model(normalize(torch.clamp(X + delta, min=lower_limit.to(device), max=upper_limit.to(device)))), y, reduction='none')
            mask = all_loss >= max_loss
            if mask.any():
                max_delta[mask] = delta.detach()[mask]
                max_loss = torch.max(max_loss, all_loss.detach())

    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    """
    Evaluate robustness with PGD on test_loader.
    Returns (avg_loss, avg_acc) for robust (PGD) evaluation.
    """
    # compute epsilon/alpha using CPU std (std is CPU tensor), then move to model device when used
    eps = (8. / 255.)
    a = (2. / 255.)
    pgd_loss = 0.0
    pgd_acc = 0
    n = 0
    model.eval()
    # infer device from model parameters (works with DataParallel)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device('cpu')

    # make per-channel epsilon/std tensors on model device
    eps_tensor = (torch.tensor(eps, dtype=torch.float32) / std).to(device=model_device)
    alpha_tensor = (torch.tensor(a, dtype=torch.float32) / std).to(device=model_device)

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(model_device)
            y = y.to(model_device)
            pgd_delta = attack_pgd(model, X, y, eps_tensor, alpha_tensor, attack_iters, restarts)
            output = model(normalize(torch.clamp(X + pgd_delta, min=lower_limit.to(model_device), max=upper_limit.to(model_device))))
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

    if n == 0:
        return 0.0, 0.0
    return pgd_loss / n, pgd_acc / n


def evaluate_standard(test_loader, model):
    """
    Standard (clean) evaluation.
    Returns (avg_loss, avg_acc).
    """
    test_loss = 0.0
    test_acc = 0
    n = 0
    model.eval()
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device('cpu')

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(model_device)
            y = y.to(model_device)
            output = model(normalize(X))
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

    if n == 0:
        return 0.0, 0.0
    return test_loss / n, test_acc / n
