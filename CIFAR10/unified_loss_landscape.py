# unified_loss_landscape_compare.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from pprint import pprint

# ---------------- USER CONFIG ----------------
CHECKPOINTS = {
    "AWP":  "/content/drive/MyDrive/TDL/AWP/AWP_10%/model_best.pth",
    "FAT":  "/content/drive/MyDrive/TDL/Fast AT/FastAT_10%/checkpoint_epoch_49.pth",
    "PART": "/content/drive/MyDrive/TDL/PART/PART_10%/latest.pth",
}
FIXED_BATCH_PATH = "/mnt/data/fixed_batch.pt"
OUT_SAVE = "/content/drive/MyDrive/TDL/plots/landscape_comparison.png"

NUM_POINTS = 21         # grid resolution per axis (reduce to 11 for speed)
ALPHA_RANGE = 0.05      # ± fraction of weight norm for directions
PGD_STEPS = 10
PGD_STEP_SIZE = 2/255.0
ATTACK_EPS = 8/255.0
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------

print("Device:", DEVICE)

# ---------------- Import ResNet18 from repo ----------------
resnet_candidates = [
    ("resnet", "ResNet18"),
    ("resnet", "ResNet"),
    ("preactresnet", "PreActResNet18"),
]
ModelClass = None
for mod_name, cls_name in resnet_candidates:
    try:
        module = __import__(mod_name)
        ModelClass = getattr(module, cls_name)
        print(f"Imported {cls_name} from {mod_name}.py")
        break
    except Exception:
        ModelClass = None

if ModelClass is None:
    try:
        import resnet as _resmod
        ModelClass = getattr(_resmod, "ResNet18", None) or getattr(_resmod, "ResNet", None)
        if ModelClass:
            print("Imported ResNet class from resnet.py")
    except Exception:
        ModelClass = None

if ModelClass is None:
    raise ImportError("Could not find ResNet class in your repo. Ensure resnet.py exists and defines ResNet18 or ResNet.")

# ---------- helper: robust loader (same as you used) ----------
def extract_state_dict(ckpt):
    candidates = [
        "state_dict", "model_state", "model_state_dict",
        "model", "model_state_dict_ema", "network",
    ]
    for c in candidates:
        if isinstance(ckpt, dict) and c in ckpt:
            candidate = ckpt[c]
            if isinstance(candidate, dict) and "state_dict" in candidate:
                return candidate["state_dict"]
            return candidate
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if isinstance(v, dict):
                sample_vals = list(v.values())[:4]
                if sample_vals and all(hasattr(x, "ndim") or isinstance(x, torch.Tensor) for x in sample_vals):
                    return v
    return None

def strip_module_prefix(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        if new_key.startswith("model_state."):
            new_key = new_key[len("model_state."):]
        new_sd[new_key] = v
    return new_sd

def load_ckpt_to_model(model, ckpt_path, device=DEVICE):
    # safe-ish load for local trusted ckpt
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state = extract_state_dict(ckpt)
    if state is None:
        raise RuntimeError(f"Could not find state_dict in checkpoint {ckpt_path}")
    # try direct, then strip prefixes, then strict=False
    try:
        model.load_state_dict(state)
    except Exception:
        s = strip_module_prefix(state)
        try:
            model.load_state_dict(s)
        except Exception:
            model.load_state_dict(s, strict=False)
    model.to(device)
    model.eval()
    return model

# ---------------- Fixed minibatch (create if missing) ----------------
if not os.path.exists(FIXED_BATCH_PATH):
    print(f"No fixed batch found at {FIXED_BATCH_PATH}. Attempting to create from torchvision CIFAR-10.")
    import torchvision
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor()])
    ds = torchvision.datasets.CIFAR10(root="/mnt/data", train=False, download=True, transform=transform)
    subset_idx = list(range(min(BATCH_SIZE, len(ds))))
    subset = Subset(ds, subset_idx)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, num_workers=0)
    inputs, targets = next(iter(loader))
    torch.save({"inputs": inputs, "targets": targets}, FIXED_BATCH_PATH)
    print("Saved fixed batch to", FIXED_BATCH_PATH)

fb = torch.load(FIXED_BATCH_PATH, map_location=DEVICE)
inputs = fb["inputs"].to(DEVICE)
targets = fb["targets"].to(DEVICE)
print("Using fixed batch:", inputs.shape)

# ---------------- Robust loss (PGD inner maximization) ----------------
loss_fn = nn.CrossEntropyLoss(reduction="mean")
def pgd_robust_loss(model, x, y, eps=ATTACK_EPS, steps=PGD_STEPS, step_size=PGD_STEP_SIZE, device=DEVICE):
    model.eval()
    x0 = x.detach()
    delta = torch.zeros_like(x0).uniform_(-eps, eps).to(device)
    delta.requires_grad_(True)
    for _ in range(steps):
        logits = model(x0 + delta)
        loss = loss_fn(logits, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = (delta + step_size * torch.sign(grad)).clamp(-eps, eps)
        delta.grad.zero_()
    with torch.no_grad():
        logits = model(x0 + delta)
        final_loss = loss_fn(logits, y).item()
    return final_loss

# ---------------- Parameter helpers (we will create unit directions once) ----------------
# instantiate a dummy model to know total_params (class must match)
tmp_model = ModelClass(num_classes=10).to(DEVICE)
param_list = [p for p in tmp_model.parameters() if p.requires_grad]
param_sizes = [p.numel() for p in param_list]
total_params = sum(param_sizes)
print("Total trainable params:", total_params)

def get_param_vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad]).cpu().numpy()

def set_param_vector(model, vec):
    vec = vec.astype(np.float32)
    ptr = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        new_val = torch.from_numpy(vec[ptr:ptr+n].reshape(p.shape)).to(DEVICE)
        p.data.copy_(new_val)
        ptr += n

# sample two global unit directions (same across models)
rng = np.random.RandomState(0)
u1 = rng.randn(total_params).astype(np.float32)
u2 = rng.randn(total_params).astype(np.float32)
u1 = u1 / np.linalg.norm(u1)
u2 = u2 / np.linalg.norm(u2)

alphas = np.linspace(-ALPHA_RANGE, ALPHA_RANGE, NUM_POINTS)
betas = np.linspace(-ALPHA_RANGE, ALPHA_RANGE, NUM_POINTS)
X, Y = np.meshgrid(alphas, betas)

# ---------------- Compute landscapes for all checkpoints ----------------
results = {}
stats = {}
for name, ckpt_path in CHECKPOINTS.items():
    print("Processing", name, ckpt_path)
    # instantiate fresh model and load checkpoint
    model = ModelClass(num_classes=10).to(DEVICE)
    load_ckpt_to_model(model, ckpt_path, device=DEVICE)

    base_vec = get_param_vector(model)
    base_norm = np.linalg.norm(base_vec)
    print(f"{name} base param norm:", base_norm)

    # scale unit directions by this model's base norm
    d1 = u1 * base_norm
    d2 = u2 * base_norm

    Z = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.float32)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            pert = base_vec + a * d1 + b * d2
            set_param_vector(model, pert)
            # compute robust loss on same fixed batch
            val = pgd_robust_loss(model, inputs, targets)
            Z[j, i] = val
        print(f"{name}: completed column {i+1}/{len(alphas)}")
    # restore model weights (not strictly necessary since we reload per-loop)
    set_param_vector(model, base_vec)

    center_idx = NUM_POINTS // 2
    Z_center = Z[center_idx, center_idx]
    Z_norm = Z - Z_center
    results[name] = Z_norm
    # compute simple scalar summaries for caption/table
    center_idx = NUM_POINTS // 2
    Zc = float(Z[center_idx, center_idx])
    max_inc = float(Z.max() - Zc)
    mean_inc = float(np.mean(Z) - Zc)
    stats[name] = {"center": Zc, "max_inc": max_inc, "mean_inc": mean_inc}
    print(f"{name} stats: center={Zc:.4f}, max_inc={max_inc:.6f}, mean_inc={mean_inc:.6f}")

# ---------------- Plot side-by-side with shared color scale ----------------
# compute global vmin/vmax
vmin = min(float(Z.min()) for Z in results.values())
vmax = max(float(Z.max()) for Z in results.values())
print("Global vmin,vmax:", vmin, vmax)

num_plots = len(results)
fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
if num_plots == 1:
    axes = [axes]
for ax, (name, Z) in zip(axes, results.items()):
    cf = ax.contourf(X, Y, Z, levels=50, vmin=vmin, vmax=vmax)
    ax.set_title(name, fontsize=14)
    ax.set_xlabel("alpha (direction 1 scale)")
    ax.set_ylabel("beta (direction 2 scale)")
    # mark center
    ax.scatter([0.0], [0.0], color="k", s=10)
fig.colorbar(cf, ax=axes, fraction=0.02, pad=0.04)
plt.suptitle("Robust loss landscapes (same color scale)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
os.makedirs(os.path.dirname(OUT_SAVE), exist_ok=True)
plt.savefig(OUT_SAVE, dpi=200, bbox_inches="tight")
plt.show()
print("Saved comparison to:", OUT_SAVE)

# print the scalar stats for quick copy-paste to paper
print("\nScalar summaries (center loss, max_increase, mean_increase):")
for name, s in stats.items():
    print(f"{name}: center={s['center']:.4f}, max_inc={s['max_inc']:.6f}, mean_inc={s['mean_inc']:.6f}")
