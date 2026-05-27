"""
train.py — Unified training script (CIFAR-10 / CIFAR-100 / ImageNet / ImageNet-100).

Training protocol
-----------------
CLEAN DATA ONLY by default. No corruption augmentation, no AugMix.
- CIFAR       : ToTensor + Normalize only (strictly no augmentation)
- ImageNet    : RandomResizedCrop + RandomHorizontalFlip + Normalize
- ImageNet-100: Same as ImageNet (100-class subset, same image scale)

AugMix (--augmix flag)
----------------------
Optional. Only activates when --augmix is explicitly passed.
Never interferes with clean training. Used only for the
"ShapeBiasNet + AugMix" ablation experiment.
Checkpoint saved separately as <model>_<dataset>_augmix.pt

Multi-GPU
---------
Automatically uses all available GPUs via DataParallel.
Checkpoints always save the underlying model weights (model.module)
so they are portable regardless of how many GPUs were used.

Experiment groups (--run_group)
--------------------------------
Predefined groups so you can run a full round with one command:

    cifar_all         : all models on CIFAR
    imagenet_core     : shape_res50 + baseline_res50            (Round 1)
    imagenet_deep     : shape_res101 + baseline_res101          (Round 2)
    imagenet_convnext : shape_convnext_tiny + baseline_convnext (Round 3)
    imagenet_effnet   : shape_effnet_b4 + baseline_effnet       (Round 4)
    imagenet_all      : all imagenet rounds combined
    imagenet100_core  : shape_res50 + baseline_res50 on ImageNet-100
    imagenet100_all   : all rounds on ImageNet-100

Usage
-----
    # Single model — CIFAR
    python train.py --model shape_res18 --dataset cifar10

    # Single model — full ImageNet
    python train.py --model shape_res50 --dataset imagenet --data_dir /pvc/imagenet

    # Single model — ImageNet-100
    python train.py --model shape_res50 --dataset imagenet100 --data_dir /pvc/imagenet100

    # Run a predefined group — ImageNet-100
    python train.py --run_group imagenet100_core --dataset imagenet100 --data_dir /pvc/imagenet100

    # Run a predefined group — full ImageNet
    python train.py --run_group imagenet_core --dataset imagenet --data_dir /pvc/imagenet

    # AugMix ablation
    python train.py --model shape_res50 --dataset imagenet100 --augmix --data_dir /pvc/imagenet100

Checkpointing
-------------
Saves full state (model + optimizer + scheduler + epoch) on every
accuracy improvement. Auto-resumes from existing checkpoint on restart.
"""

import argparse, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models import build_model, MODEL_NAMES

# ──────────────────────────────────────────────────────────────
#  EXPERIMENT GROUPS
# ──────────────────────────────────────────────────────────────
GROUPS = {
    # CIFAR
    "cifar_all":           ["baseline_res18", "baseline_res50",
                            "shape_custom", "shape_res18", "shape_res50"],
    # Full ImageNet rounds
    "imagenet_core":       ["baseline_res50",           "shape_res50"],
    "imagenet_deep":       ["baseline_res101",           "shape_res101"],
    "imagenet_convnext":   ["baseline_convnext_tiny",    "shape_convnext_tiny"],
    "imagenet_effnet":     ["baseline_efficientnet_b4",  "shape_effnet_b4"],
    "imagenet_all":        ["baseline_res50",            "shape_res50",
                            "baseline_res101",           "shape_res101",
                            "baseline_convnext_tiny",    "shape_convnext_tiny",
                            "baseline_efficientnet_b4",  "shape_effnet_b4"],
    # ImageNet-100 rounds (same model pairs, different dataset)
    "imagenet100_core":    ["baseline_res50",            "shape_res50"],
    "imagenet100_deep":    ["baseline_res101",           "shape_res101"],
    "imagenet100_convnext":["baseline_convnext_tiny",    "shape_convnext_tiny"],
    "imagenet100_effnet":  ["baseline_efficientnet_b4",  "shape_effnet_b4"],
    "imagenet100_all":     ["baseline_res50",            "shape_res50",
                            "baseline_res101",           "shape_res101",
                            "baseline_convnext_tiny",    "shape_convnext_tiny",
                            "baseline_efficientnet_b4",  "shape_effnet_b4"],
    # PMConv-L (Lorentzian, n_steps=1 — this branch, fast version)
    "pmconv_l_res50":      ["baseline_res50",            "pmconv_l_res50"],
    "pmconv_l_res101":     ["baseline_res101",           "pmconv_l_res101"],
    "pmconv_l_convnext":   ["baseline_convnext_tiny",    "pmconv_l_convnext_tiny"],
    "pmconv_l_effnet":     ["baseline_efficientnet_b4",  "pmconv_l_effnet_b4"],
    "pmconv_l_all":        ["baseline_res50",            "pmconv_l_res50",
                            "baseline_res101",           "pmconv_l_res101",
                            "baseline_convnext_tiny",    "pmconv_l_convnext_tiny",
                            "baseline_efficientnet_b4",  "pmconv_l_effnet_b4"],
    # RobustConv rounds — isotropic ablation
    "robustconv_res50":    ["baseline_res50",            "robustconv_res50"],
    "robustconv_res101":   ["baseline_res101",           "robustconv_res101"],
    "robustconv_all":      ["baseline_res50",            "robustconv_res50",
                            "baseline_res101",           "robustconv_res101"],
    # Four-way ablation: baseline / isotropic / anisotropic-L / dual-stream
    "ablation_res50":      ["baseline_res50",   "robustconv_res50",
                            "pmconv_l_res50",   "shape_res50"],
    "ablation_res101":     ["baseline_res101",  "robustconv_res101",
                            "pmconv_l_res101",  "shape_res101"],
    # PMConv-A (adaptive k + shared conductance — pmconv-adaptive branch)
    "pmconv_a_res50":      ["baseline_res50",            "pmconv_a_res50"],
    "pmconv_a_res101":     ["baseline_res101",           "pmconv_a_res101"],
    "pmconv_a_convnext":   ["baseline_convnext_tiny",    "pmconv_a_convnext_tiny"],
    "pmconv_a_effnet":     ["baseline_efficientnet_b4",  "pmconv_a_effnet_b4"],
    "pmconv_a_all":        ["baseline_res50",            "pmconv_a_res50",
                            "baseline_res101",           "pmconv_a_res101",
                            "baseline_convnext_tiny",    "pmconv_a_convnext_tiny",
                            "baseline_efficientnet_b4",  "pmconv_a_effnet_b4"],
    # L vs A ablation: fixed-k vs learnable-k anisotropic
    "ablation_a_res50":    ["baseline_res50",  "robustconv_res50",
                            "pmconv_l_res50",  "pmconv_a_res50"],
    "ablation_a_res101":   ["baseline_res101", "robustconv_res101",
                            "pmconv_l_res101", "pmconv_a_res101"],
    # ── FDN (freq-norm branch) ────────────────────────────────
    "fdn_res50":           ["baseline_res50",            "fdn_res50"],
    "fdn_res101":          ["baseline_res101",           "fdn_res101"],
    "fdn_convnext":        ["baseline_convnext_tiny",    "fdn_convnext_tiny"],
    "fdn_effnet":          ["baseline_efficientnet_b4",  "fdn_effnet_b4"],
    "fdn_all":             ["baseline_res50",            "fdn_res50",
                            "baseline_res101",           "fdn_res101",
                            "baseline_convnext_tiny",    "fdn_convnext_tiny",
                            "baseline_efficientnet_b4",  "fdn_effnet_b4"],
    # ── INSC (freq-norm branch) ───────────────────────────────
    "insc_res50":          ["baseline_res50",            "insc_res50"],
    "insc_res101":         ["baseline_res101",           "insc_res101"],
    "insc_convnext":       ["baseline_convnext_tiny",    "insc_convnext_tiny"],
    "insc_effnet":         ["baseline_efficientnet_b4",  "insc_effnet_b4"],
    "insc_all":            ["baseline_res50",            "insc_res50",
                            "baseline_res101",           "insc_res101",
                            "baseline_convnext_tiny",    "insc_convnext_tiny",
                            "baseline_efficientnet_b4",  "insc_effnet_b4"],
    # ── FDN+INSC combined ★ main contribution ────────────────
    "fdn_insc_res50":      ["baseline_res50",            "fdn_insc_res50"],
    "fdn_insc_res101":     ["baseline_res101",           "fdn_insc_res101"],
    "fdn_insc_convnext":   ["baseline_convnext_tiny",    "fdn_insc_convnext_tiny"],
    "fdn_insc_effnet":     ["baseline_efficientnet_b4",  "fdn_insc_effnet_b4"],
    "fdn_insc_all":        ["baseline_res50",            "fdn_insc_res50",
                            "baseline_res101",           "fdn_insc_res101",
                            "baseline_convnext_tiny",    "fdn_insc_convnext_tiny",
                            "baseline_efficientnet_b4",  "fdn_insc_effnet_b4"],
    # ── Four-way ablation: baseline / fdn / insc / fdn+insc ──
    "ablation_fn_res50":   ["baseline_res50",  "fdn_res50",
                            "insc_res50",      "fdn_insc_res50"],
    "ablation_fn_res101":  ["baseline_res101", "fdn_res101",
                            "insc_res101",     "fdn_insc_res101"],
    # ── Gated INSC (learnable gate per block) ─────────────────
    "fdn_ginsc_res50":     ["baseline_res50",    "fdn_ginsc_res50"],
    "fdn_ginsc_res101":    ["baseline_res101",   "fdn_ginsc_res101"],
    "fdn_ginsc_convnext":  ["baseline_convnext_tiny", "fdn_ginsc_convnext_tiny"],
    "fdn_ginsc_effnet":    ["baseline_efficientnet_b4", "fdn_ginsc_effnet_b4"],
    "fdn_ginsc_all":       ["baseline_res50",            "fdn_ginsc_res50",
                            "baseline_res101",           "fdn_ginsc_res101",
                            "baseline_convnext_tiny",    "fdn_ginsc_convnext_tiny",
                            "baseline_efficientnet_b4",  "fdn_ginsc_effnet_b4"],
    # ── Ablation: fdn_insc vs fdn_ginsc ───────────────────────
    "ablation_gate_res50":  ["baseline_res50",  "fdn_insc_res50",  "fdn_ginsc_res50"],
    "ablation_gate_res101": ["baseline_res101", "fdn_insc_res101", "fdn_ginsc_res101"],
}

# ──────────────────────────────────────────────────────────────
#  ARGS
# ──────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--model",       default="shape_res18", choices=MODEL_NAMES)
p.add_argument("--dataset",     default="cifar10",
               choices=["cifar10", "cifar100", "imagenet", "imagenet100"])
p.add_argument("--data_dir",    default="./data",
               help="Data root. ImageNet expects <data_dir>/train and <data_dir>/val")
p.add_argument("--ckpt_dir",    default="./checkpoints")
p.add_argument("--epochs",      type=int,   default=None,
               help="Default: 40 CIFAR / 90 ImageNet / 90 ImageNet-100")
p.add_argument("--batch",       type=int,   default=None,
               help="Default: 128 CIFAR / 256 ImageNet")
p.add_argument("--workers",     type=int,   default=None,
               help="Default: 2 CIFAR / 8 ImageNet")
p.add_argument("--lr",          type=float, default=0.1)
p.add_argument("--accum_steps", type=int,   default=1,
               help="Gradient accumulation steps")
p.add_argument("--seed",        type=int,   default=42)

run_group = p.add_mutually_exclusive_group()
run_group.add_argument("--run_all",   action="store_true",
                       help="Train all MODEL_NAMES for --dataset")
run_group.add_argument("--run_group", choices=list(GROUPS.keys()),
                       help="Train a predefined experiment group")

p.add_argument("--augmix", action="store_true",
               help="Use AugMix augmentation (ablation only). "
                    "Saves checkpoint as <model>_<dataset>_augmix.pt. "
                    "Does NOT affect clean training runs.")

args = p.parse_args()

# ImageNet-100 behaves identically to ImageNet at the transform/loader level
# (same 224×224 images, same normalization) — only num_classes differs
IS_IMAGENET    = args.dataset in ("imagenet", "imagenet100")
IS_IMAGENET100 = args.dataset == "imagenet100"
EPOCHS      = args.epochs  or (90  if IS_IMAGENET else 40)
BATCH       = args.batch   or (256 if IS_IMAGENET else 128)
WORKERS     = args.workers or (8   if IS_IMAGENET else 2)
NUM_CLASSES = (100  if IS_IMAGENET100 else
               1000 if IS_IMAGENET    else
               10   if args.dataset == "cifar10" else 100)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS    = torch.cuda.device_count()

# model_dataset_key used for build_model — imagenet100 uses imagenet stem
MODEL_DATASET = "imagenet" if IS_IMAGENET else args.dataset

os.makedirs(args.ckpt_dir, exist_ok=True)

if args.run_all:
    runs = MODEL_NAMES
elif args.run_group:
    runs = GROUPS[args.run_group]
else:
    runs = [args.model]


# ──────────────────────────────────────────────────────────────
#  REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(args.seed)


# ──────────────────────────────────────────────────────────────
#  DATA
# ──────────────────────────────────────────────────────────────
# ImageNet-100 uses the same normalization stats as full ImageNet
MEAN = {"cifar10":    (0.4914, 0.4822, 0.4465),
        "cifar100":   (0.5071, 0.4867, 0.4408),
        "imagenet":   (0.485,  0.456,  0.406),
        "imagenet100":(0.485,  0.456,  0.406)}
STD  = {"cifar10":   (0.247,  0.243,  0.261),
        "cifar100":  (0.2675, 0.2565, 0.2761),
        "imagenet":  (0.229,  0.224,  0.225),
        "imagenet100":(0.229,  0.224,  0.225)}
mean, std = MEAN[args.dataset], STD[args.dataset]


def get_transforms():
    if args.augmix:
        if not IS_IMAGENET:
            raise ValueError("--augmix is only supported for ImageNet/ImageNet-100.")
        train_tf = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.AugMix(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        val_tf = T.Compose([
            T.Resize(256), T.CenterCrop(224),
            T.ToTensor(), T.Normalize(mean, std),
        ])
        return train_tf, val_tf

    if IS_IMAGENET:
        # Same transforms for both imagenet and imagenet100
        train_tf = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        val_tf = T.Compose([
            T.Resize(256), T.CenterCrop(224),
            T.ToTensor(), T.Normalize(mean, std),
        ])
    else:
        train_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        val_tf   = train_tf

    return train_tf, val_tf


train_tf, val_tf = get_transforms()

if IS_IMAGENET:
    import pickle

    def load_imagefolder_cached(root, transform, cache_path):
        """ImageFolder with cached file index.
        First run: scans filesystem (slow on CephFS), saves cache to PVC.
        Subsequent runs: loads cache in ~3 seconds.
        Works for both full ImageNet and ImageNet-100.
        """
        ds = torchvision.datasets.ImageFolder(root, transform=transform)
        if os.path.exists(cache_path):
            print(f"  [Cache] Loading index from {cache_path}...")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            ds.samples = cached["samples"]
            ds.targets = cached["targets"]
            ds.imgs    = ds.samples
            print(f"  [Cache] Loaded {len(ds.samples)} samples")
        else:
            print(f"  [Cache] Building index (first time, may take a few min)...")
            with open(cache_path, "wb") as f:
                pickle.dump({"samples": ds.samples, "targets": ds.targets}, f)
            print(f"  [Cache] Saved {len(ds.samples)} samples to {cache_path}")
        return ds

    # Cache filename includes dataset name so imagenet and imagenet100
    # caches never overwrite each other
    cache_prefix = "imagenet100" if IS_IMAGENET100 else "imagenet"
    trainset = load_imagefolder_cached(
        os.path.join(args.data_dir, "train"), train_tf,
        os.path.join(args.data_dir, f"{cache_prefix}_train_cache.pkl"))
    testset  = load_imagefolder_cached(
        os.path.join(args.data_dir, "val"), val_tf,
        os.path.join(args.data_dir, f"{cache_prefix}_val_cache.pkl"))
else:
    DS = (torchvision.datasets.CIFAR10 if args.dataset == "cifar10"
          else torchvision.datasets.CIFAR100)
    trainset = DS(args.data_dir, train=True,  download=True, transform=train_tf)
    testset  = DS(args.data_dir, train=False, download=True, transform=val_tf)

trainloader = DataLoader(trainset, BATCH,   shuffle=True,
                         num_workers=WORKERS, pin_memory=True,
                         persistent_workers=(WORKERS > 0))
testloader  = DataLoader(testset,  BATCH*2, shuffle=False,
                         num_workers=WORKERS, pin_memory=True,
                         persistent_workers=(WORKERS > 0))

TRAIN_MODE = "augmix" if args.augmix else "clean"
if IS_IMAGENET:
    TRAIN_DESC = "AugMix (ablation)" if args.augmix else "clean (RandomResizedCrop+HFlip+Normalize)"
else:
    TRAIN_DESC = "clean only (ToTensor+Normalize)"


# ──────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────
def unwrap(model: nn.Module) -> nn.Module:
    """Return underlying model regardless of DataParallel wrapping."""
    return model.module if isinstance(model, nn.DataParallel) else model


# ──────────────────────────────────────────────────────────────
#  CLEAN EVAL
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module) -> float:
    model.eval()
    correct = total = 0
    for x, y in testloader:
        x, y    = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


# ──────────────────────────────────────────────────────────────
#  TRAIN ONE MODEL
# ──────────────────────────────────────────────────────────────
def train_model(name: str) -> float:
    set_seed(args.seed)

    ckpt_suffix = f"_{TRAIN_MODE}" if args.augmix else ""
    ckpt_path   = os.path.join(args.ckpt_dir,
                               f"{name}_{args.dataset}{ckpt_suffix}.pt")

    print(f"\n{'='*60}")
    print(f"  Model      : {name}")
    print(f"  Dataset    : {args.dataset}  ({NUM_CLASSES} classes)")
    print(f"  Epochs     : {EPOCHS}  |  Batch: {BATCH}  |  Workers: {WORKERS}")
    print(f"  Device     : {DEVICE}  |  GPUs: {NUM_GPUS}")
    print(f"  Train mode : {TRAIN_DESC}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'='*60}")

    # ── build model ──────────────────────────────────────────
    # Use "imagenet" as dataset key for build_model — imagenet100 uses
    # the same 7×7 stem as full ImageNet (224×224 images)
    model = build_model(name, NUM_CLASSES, dataset=MODEL_DATASET).to(DEVICE)

    # ── DataParallel disabled — causes NCCL hang on Nautilus multi-GPU nodes
    # Use single GPU only (GPU 0)
    if False and NUM_GPUS > 1:  # disabled
        print(f"  [DataParallel] Using {NUM_GPUS} GPUs")
        model = nn.DataParallel(model)

    opt  = torch.optim.SGD(unwrap(model).parameters(), lr=args.lr,
                           momentum=0.9, weight_decay=1e-4, nesterov=True)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_ep = 1
    best_acc = 0.0

    # ── auto-resume ──────────────────────────────────────────
    if os.path.exists(ckpt_path):
        print(f"  [Resume] Found checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = unwrap(model).load_state_dict(state["model"], strict=False)
        if missing or unexpected:
            print(f"  [Resume] Architecture mismatch — starting fresh.")
            print(f"    Missing : {missing}")
            print(f"    Unexpected: {unexpected}")
        else:
            opt.load_state_dict(state["opt"])
            sch.load_state_dict(state["sch"])
            start_ep = state["epoch"] + 1
            best_acc = state.get("best_acc", 0.0)
            print(f"  [Resume] Continuing from epoch {start_ep}, best={best_acc:.2f}%")

    if start_ep > EPOCHS:
        print("  Already completed. Skipping.")
        return best_acc

    # ── training loop ────────────────────────────────────────
    opt.zero_grad(set_to_none=True)
    for ep in range(start_ep, EPOCHS + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0

        for step, (x, y) in enumerate(trainloader):
            x, y  = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            loss  = crit(model(x), y) / args.accum_steps
            loss.backward()
            loss_sum += loss.item() * args.accum_steps
            if hasattr(unwrap(model), "update_prototypes"):
                unwrap(model).update_prototypes(y)

            if (step + 1) % args.accum_steps == 0:
                nn.utils.clip_grad_norm_(unwrap(model).parameters(), 5.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

        sch.step()
        acc     = evaluate(model)
        elapsed = time.time() - t0

        print(f"  [{ep:03d}/{EPOCHS}] "
              f"loss={loss_sum/len(trainloader):.4f} | "
              f"clean_acc={acc:.2f}% | "
              f"lr={sch.get_last_lr()[0]:.5f} | "
              f"{elapsed:.0f}s")

        # ── save on improvement ──────────────────────────────
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch":      ep,
                "model":      unwrap(model).state_dict(),
                "opt":        opt.state_dict(),
                "sch":        sch.state_dict(),
                "best_acc":   best_acc,
                "model_name": name,
                "dataset":    args.dataset,
                "train_mode": TRAIN_MODE,
            }, ckpt_path)
            print(f"  [Saved] best={best_acc:.2f}% → {ckpt_path}")

    print(f"\n  Best clean accuracy: {best_acc:.2f}%")
    return best_acc


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
print(f"\n  Running {len(runs)} model(s): {runs}")
print(f"  Dataset: {args.dataset}  |  Mode: {TRAIN_DESC}")

# for name in runs:
#     train_model(name)

for name in runs:
    train_model(name)
    # Release GPU memory between models
    torch.cuda.empty_cache()
    import gc
    gc.collect()