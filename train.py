"""
train.py — Training script for ShapeBiasNet (CIFAR / ImageNet).

Protocol
--------
CIFAR    : ToTensor + Normalize only (no augmentation)
ImageNet : RandomResizedCrop(224) + RandomHorizontalFlip + Normalize

Usage
-----
    # Single model
    python train.py --model shape_res18 --dataset cifar10
    python train.py --model shape_res50 --dataset imagenet100 --data_dir /pvc/imagenet100

    # Predefined group
    python train.py --run_group cifar_all --dataset cifar10
    python train.py --run_group imagenet_core --dataset imagenet100 --data_dir /pvc/imagenet100
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
    "cifar_all":        ["baseline_res18", "baseline_res50",
                         "shape_res18",    "shape_res50"],
    # ImageNet / ImageNet-100
    "imagenet_core":    ["baseline_res50",  "shape_res50"],
    "imagenet_deep":    ["baseline_res101", "shape_res101"],
    "imagenet_all":     ["baseline_res50",  "shape_res50",
                         "baseline_res101", "shape_res101"],
    "imagenet100_core": ["baseline_res50",  "shape_res50"],
    "imagenet100_all":  ["baseline_res50",  "shape_res50",
                         "baseline_res101", "shape_res101"],
    # Gating ablation (res18 / cifar10 only)
    "ablation_cifar10": ["baseline_res18",           "shape_res18",
                         "shape_res18_early_gate",   "shape_res18_late_gate",
                         "shape_res18_gate_only",    "shape_res18_early_gate_nofuse"],
}

# ──────────────────────────────────────────────────────────────
#  ARGS
# ──────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--model",       default="shape_res18", choices=MODEL_NAMES)
p.add_argument("--dataset",     default="cifar10",
               choices=["cifar10", "cifar100", "imagenet", "imagenet100"])
p.add_argument("--data_dir",    default="./data")
p.add_argument("--ckpt_dir",    default="./checkpoints")
p.add_argument("--epochs",      type=int,   default=None,
               help="Default: 40 CIFAR / 90 ImageNet")
p.add_argument("--batch",       type=int,   default=None,
               help="Default: 128 CIFAR / 256 ImageNet")
p.add_argument("--workers",     type=int,   default=None,
               help="Default: 2 CIFAR / 8 ImageNet")
p.add_argument("--lr",          type=float, default=0.1)
p.add_argument("--accum_steps", type=int,   default=1)
p.add_argument("--seed",        type=int,   default=42)

run_group = p.add_mutually_exclusive_group()
run_group.add_argument("--run_all",   action="store_true",
                       help="Train all MODEL_NAMES for --dataset")
run_group.add_argument("--run_group", choices=list(GROUPS.keys()))

args = p.parse_args()

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
MEAN = {"cifar10":     (0.4914, 0.4822, 0.4465),
        "cifar100":    (0.5071, 0.4867, 0.4408),
        "imagenet":    (0.485,  0.456,  0.406),
        "imagenet100": (0.485,  0.456,  0.406)}
STD  = {"cifar10":     (0.247,  0.243,  0.261),
        "cifar100":    (0.2675, 0.2565, 0.2761),
        "imagenet":    (0.229,  0.224,  0.225),
        "imagenet100": (0.229,  0.224,  0.225)}
mean, std = MEAN[args.dataset], STD[args.dataset]

if IS_IMAGENET:
    train_tf = T.Compose([
        T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    val_tf = T.Compose([T.Resize(256), T.CenterCrop(224),
                        T.ToTensor(), T.Normalize(mean, std)])
else:
    train_tf = val_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

TRAIN_DESC = ("clean (RandomResizedCrop+HFlip+Normalize)"
              if IS_IMAGENET else "clean only (ToTensor+Normalize)")

if IS_IMAGENET:
    import pickle

    def load_imagefolder_cached(root, transform, cache_path):
        ds = torchvision.datasets.ImageFolder(root, transform=transform)
        if os.path.exists(cache_path):
            print(f"  [Cache] Loading index from {cache_path}...")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            ds.samples = cached["samples"]
            ds.targets = cached["targets"]
            ds.imgs    = ds.samples
        else:
            print(f"  [Cache] Building index (first time)...")
            with open(cache_path, "wb") as f:
                pickle.dump({"samples": ds.samples, "targets": ds.targets}, f)
        return ds

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


# ──────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────
def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


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
    ckpt_path = os.path.join(args.ckpt_dir, f"{name}_{args.dataset}.pt")

    print(f"\n{'='*60}")
    print(f"  Model      : {name}")
    print(f"  Dataset    : {args.dataset}  ({NUM_CLASSES} classes)")
    print(f"  Epochs     : {EPOCHS}  |  Batch: {BATCH}  |  Workers: {WORKERS}")
    print(f"  Device     : {DEVICE}  |  GPUs: {NUM_GPUS}")
    print(f"  Train mode : {TRAIN_DESC}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'='*60}")

    model = build_model(name, NUM_CLASSES, dataset=MODEL_DATASET).to(DEVICE)

    opt  = torch.optim.SGD(unwrap(model).parameters(), lr=args.lr,
                           momentum=0.9, weight_decay=1e-4, nesterov=True)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_ep = 1
    best_acc = 0.0

    if os.path.exists(ckpt_path):
        print(f"  [Resume] Found checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = unwrap(model).load_state_dict(state["model"], strict=False)
        if missing or unexpected:
            print(f"  [Resume] Architecture mismatch — starting fresh.")
        else:
            opt.load_state_dict(state["opt"])
            sch.load_state_dict(state["sch"])
            start_ep = state["epoch"] + 1
            best_acc = state.get("best_acc", 0.0)
            print(f"  [Resume] Continuing from epoch {start_ep}, best={best_acc:.2f}%")

    if start_ep > EPOCHS:
        print("  Already completed. Skipping.")
        return best_acc

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
            }, ckpt_path)
            print(f"  [Saved] best={best_acc:.2f}% → {ckpt_path}")

    print(f"\n  Best clean accuracy: {best_acc:.2f}%")
    return best_acc


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
print(f"\n  Running {len(runs)} model(s): {runs}")
print(f"  Dataset: {args.dataset}  |  Mode: {TRAIN_DESC}")

for name in runs:
    train_model(name)
    torch.cuda.empty_cache()
    import gc; gc.collect()
