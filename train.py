# import torch, argparse, random, numpy as np, torchvision
# import torchvision.transforms as T
# from models import build_model

# # ---------------- setup ----------------
# SEED=42
# torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
# np.random.seed(SEED); random.seed(SEED)
# torch.backends.cudnn.deterministic=True
# DEVICE="cuda" if torch.cuda.is_available() else "cpu"

# # ---------------- args -----------------
# p = argparse.ArgumentParser()
# p.add_argument("--model", default="shape_custom")
# p.add_argument("--dataset", default="cifar10")
# p.add_argument("--epochs", type=int, default=40)
# p.add_argument("--run_all", action="store_true")
# args = p.parse_args()

# MODELS = ["baseline_res18","baseline_res50","shape_custom","shape_res18","shape_res50"]
# if args.run_all: runs = MODELS
# else: runs = [args.model]

# # ---------------- data -----------------
# mean10,std10=(0.4914,0.4822,0.4465),(0.247,0.243,0.261)
# mean100,std100=(0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)
# mean,std = (mean10,std10) if args.dataset=="cifar10" else (mean100,std100)

# tf = T.Compose([T.ToTensor(),T.Normalize(mean,std)])

# DS = torchvision.datasets.CIFAR10 if args.dataset=="cifar10" else torchvision.datasets.CIFAR100
# trainset = DS("./data",True,download=True,transform=tf)
# testset  = DS("./data",False,download=True,transform=tf)

# trainloader = torch.utils.data.DataLoader(trainset,128,shuffle=True,num_workers=2,pin_memory=True)
# testloader  = torch.utils.data.DataLoader(testset,256,shuffle=False,num_workers=2,pin_memory=True)

# # ---------------- train fn --------------
# @torch.no_grad()
# def test(model):
#     model.eval(); c=t=0
#     for x,y in testloader:
#         x,y=x.to(DEVICE),y.to(DEVICE)
#         c+=(model(x).argmax(1)==y).sum().item(); t+=y.size(0)
#     return 100*c/t

# # ---------------- loop ------------------
# for name in runs:
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#     np.random.seed(SEED) 
#     random.seed(SEED)
#     print("\n==============================")
#     print("Training:", name)
#     print("==============================")

#     model = build_model(name, 10 if args.dataset=="cifar10" else 100).to(DEVICE)
#     shape = name.startswith("shape")
#     if shape: model.alpha=0.0

#     opt = torch.optim.SGD(model.parameters(),0.1,momentum=0.9,weight_decay=1e-4,nesterov=True)
#     sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
#     crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

#     best=0
#     for ep in range(1,args.epochs+1):
#         if shape:
#             if ep<=8: model.alpha=0
#             else: model.alpha=min(1,(ep-8)/16)

#         model.train(); loss_sum=0
#         for x,y in trainloader:
#             x,y=x.to(DEVICE),y.to(DEVICE)
#             opt.zero_grad(set_to_none=True)
#             loss=crit(model(x),y); loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(),5)
#             opt.step(); loss_sum+=loss.item()

#         sch.step()
#         acc=test(model)
#         print(f"{ep:03d} | α={getattr(model,'alpha',1):.2f} | loss {loss_sum/len(trainloader):.3f} | acc {acc:.2f}")

#         if acc>best:
#             best=acc
#             torch.save(model.state_dict(), f"checkpoints/{name}_{args.dataset}.pt")

#     print("Best:",best)





# """
# train.py — Unified training script (CIFAR-10 / CIFAR-100 / ImageNet).

# Training protocol
# -----------------
# CLEAN DATA ONLY. No corruption augmentation, no AugMix, no RandAugment.
# - CIFAR : ToTensor + Normalize only (strictly no augmentation)
# - ImageNet: RandomResizedCrop + RandomHorizontalFlip + Normalize
#   (standard clean baseline — used by all comparison methods including
#   AugMix, DeepAugment etc. These two transforms are NOT considered
#   augmentation in the corruption-robustness literature)

# This clean-only protocol is the core experimental claim: robustness
# comes from the shape stream's inductive bias, not from data augmentation.

# Checkpointing
# -------------
# Saves full state (model + optimizer + scheduler + epoch) on every
# accuracy improvement. Auto-resumes from existing checkpoint on restart —
# critical for Nautilus jobs that may be preempted.

# Usage
# -----
#     # CIFAR
#     python train.py --model shape_res18 --dataset cifar10
#     python train.py --model shape_res50 --dataset cifar100
#     python train.py --dataset cifar10 --run_all

#     # ImageNet
#     python train.py --model shape_res18 --dataset imagenet \\
#                     --data_dir /pvc/imagenet --ckpt_dir /pvc/checkpoints
# """

# import argparse, os, random, time
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as T
# from torch.utils.data import DataLoader
# from models import build_model, MODEL_NAMES

# # ──────────────────────────────────────────────────────────────
# #  ARGS
# # ──────────────────────────────────────────────────────────────
# p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# p.add_argument("--model",       default="shape_res18", choices=MODEL_NAMES)
# p.add_argument("--dataset",     default="cifar10",     choices=["cifar10","cifar100","imagenet"])
# p.add_argument("--data_dir",    default="./data",
#                help="Data root. ImageNet expects <data_dir>/train and <data_dir>/val")
# p.add_argument("--ckpt_dir",    default="./checkpoints")
# p.add_argument("--epochs",      type=int,   default=None, help="Default: 40 CIFAR / 90 ImageNet")
# p.add_argument("--batch",       type=int,   default=None, help="Default: 128 CIFAR / 256 ImageNet")
# p.add_argument("--workers",     type=int,   default=None, help="Default: 2 CIFAR / 8 ImageNet")
# p.add_argument("--lr",          type=float, default=0.1)
# p.add_argument("--accum_steps", type=int,   default=1,
#                help="Gradient accumulation steps (for large effective batch sizes)")
# p.add_argument("--run_all",     action="store_true", help="Train all models sequentially")
# p.add_argument("--seed",        type=int,   default=42)
# args = p.parse_args()

# IS_IMAGENET = args.dataset == "imagenet"
# EPOCHS      = args.epochs  or (90  if IS_IMAGENET else 40)
# BATCH       = args.batch   or (256 if IS_IMAGENET else 128)
# WORKERS     = args.workers or (8   if IS_IMAGENET else 2)
# NUM_CLASSES = 1000 if IS_IMAGENET else (10 if args.dataset == "cifar10" else 100)
# DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# os.makedirs(args.ckpt_dir, exist_ok=True)


# # ──────────────────────────────────────────────────────────────
# #  REPRODUCIBILITY
# # ──────────────────────────────────────────────────────────────
# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True

# set_seed(args.seed)


# # ──────────────────────────────────────────────────────────────
# #  DATA  (clean only)
# # ──────────────────────────────────────────────────────────────
# MEAN = {"cifar10":  (0.4914, 0.4822, 0.4465),
#         "cifar100": (0.5071, 0.4867, 0.4408),
#         "imagenet": (0.485,  0.456,  0.406)}
# STD  = {"cifar10":  (0.247,  0.243,  0.261),
#         "cifar100": (0.2675, 0.2565, 0.2761),
#         "imagenet": (0.229,  0.224,  0.225)}
# mean, std = MEAN[args.dataset], STD[args.dataset]

# if IS_IMAGENET:
#     # Standard clean ImageNet protocol used by all baselines (AugMix, etc.)
#     # RandomResizedCrop + RandomHorizontalFlip are NOT corruption augmentation.
#     train_tf = T.Compose([
#         T.RandomResizedCrop(224),
#         T.RandomHorizontalFlip(),
#         T.ToTensor(),
#         T.Normalize(mean, std),
#     ])
#     val_tf = T.Compose([
#         T.Resize(256), T.CenterCrop(224),
#         T.ToTensor(), T.Normalize(mean, std),
#     ])
#     trainset = torchvision.datasets.ImageFolder(
#         os.path.join(args.data_dir, "train"), transform=train_tf)
#     testset  = torchvision.datasets.ImageFolder(
#         os.path.join(args.data_dir, "val"),   transform=val_tf)
# else:
#     # CIFAR: strictly no augmentation — ToTensor + Normalize only
#     tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
#     DS = (torchvision.datasets.CIFAR10 if args.dataset == "cifar10"
#           else torchvision.datasets.CIFAR100)
#     trainset = DS(args.data_dir, train=True,  download=True, transform=tf)
#     testset  = DS(args.data_dir, train=False, download=True, transform=tf)

# trainloader = DataLoader(trainset, BATCH,   shuffle=True,
#                          num_workers=WORKERS, pin_memory=True,
#                          persistent_workers=(WORKERS > 0))
# testloader  = DataLoader(testset,  BATCH*2, shuffle=False,
#                          num_workers=WORKERS, pin_memory=True,
#                          persistent_workers=(WORKERS > 0))


# # ──────────────────────────────────────────────────────────────
# #  CLEAN EVAL
# # ──────────────────────────────────────────────────────────────
# @torch.no_grad()
# def evaluate(model: nn.Module) -> float:
#     model.eval()
#     correct = total = 0
#     for x, y in testloader:
#         x, y     = x.to(DEVICE), y.to(DEVICE)
#         correct  += (model(x).argmax(1) == y).sum().item()
#         total    += y.size(0)
#     return 100.0 * correct / total


# # ──────────────────────────────────────────────────────────────
# #  TRAIN ONE MODEL
# # ──────────────────────────────────────────────────────────────
# def train_model(name: str) -> float:
#     set_seed(args.seed)

#     print(f"\n{'='*60}")
#     print(f"  Model   : {name}")
#     print(f"  Dataset : {args.dataset}  ({NUM_CLASSES} classes)")
#     print(f"  Epochs  : {EPOCHS}  |  Batch: {BATCH}  |  Workers: {WORKERS}")
#     print(f"  Device  : {DEVICE}")
#     print(f"  Train transform: {'clean only (ToTensor+Normalize)' if not IS_IMAGENET else 'clean (RandomResizedCrop+HFlip+Normalize)'}")
#     print(f"{'='*60}")

#     model = build_model(name, NUM_CLASSES, dataset=args.dataset).to(DEVICE)
#     opt   = torch.optim.SGD(model.parameters(), lr=args.lr,
#                             momentum=0.9, weight_decay=1e-4, nesterov=True)
#     sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
#     crit  = nn.CrossEntropyLoss(label_smoothing=0.1)

#     start_ep = 1
#     best_acc = 0.0
#     ckpt_path = os.path.join(args.ckpt_dir, f"{name}_{args.dataset}.pt")

#     # ── auto-resume on Nautilus preemption ──
#     if os.path.exists(ckpt_path):
#         print(f"  [Resume] Found checkpoint: {ckpt_path}")
#         state    = torch.load(ckpt_path, map_location="cpu")
#         model.load_state_dict(state["model"])
#         opt.load_state_dict(state["opt"])
#         sch.load_state_dict(state["sch"])
#         start_ep = state["epoch"] + 1
#         best_acc = state.get("best_acc", 0.0)
#         print(f"  [Resume] Continuing from epoch {start_ep}, best={best_acc:.2f}%")

#     if start_ep > EPOCHS:
#         print("  Already completed. Skipping training.")
#         return best_acc

#     # ── training loop ──
#     opt.zero_grad(set_to_none=True)
#     for ep in range(start_ep, EPOCHS + 1):
#         t0 = time.time()
#         model.train()
#         loss_sum = 0.0

#         for step, (x, y) in enumerate(trainloader):
#             x, y  = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
#             loss  = crit(model(x), y) / args.accum_steps
#             loss.backward()
#             loss_sum += loss.item() * args.accum_steps

#             if (step + 1) % args.accum_steps == 0:
#                 nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#                 opt.step()
#                 opt.zero_grad(set_to_none=True)

#         sch.step()
#         acc     = evaluate(model)
#         elapsed = time.time() - t0

#         print(f"  [{ep:03d}/{EPOCHS}] "
#               f"loss={loss_sum/len(trainloader):.4f} | "
#               f"clean_acc={acc:.2f}% | "
#               f"lr={sch.get_last_lr()[0]:.5f} | "
#               f"{elapsed:.0f}s")

#         # ── save on improvement (full state for resuming) ──
#         if acc > best_acc:
#             best_acc = acc
#             torch.save({
#                 "epoch":      ep,
#                 "model":      model.state_dict(),
#                 "opt":        opt.state_dict(),
#                 "sch":        sch.state_dict(),
#                 "best_acc":   best_acc,
#                 "model_name": name,
#                 "dataset":    args.dataset,
#             }, ckpt_path)
#             print(f"  [Saved] best={best_acc:.2f}% → {ckpt_path}")

#     print(f"\n  Best clean accuracy: {best_acc:.2f}%")
#     return best_acc


# # ──────────────────────────────────────────────────────────────
# #  MAIN
# # ──────────────────────────────────────────────────────────────
# runs = MODEL_NAMES if args.run_all else [args.model]
# for name in runs:
#     train_model(name)


# """
# train.py — Unified training script (CIFAR-10 / CIFAR-100 / ImageNet).

# Training protocol
# -----------------
# CLEAN DATA ONLY by default. No corruption augmentation, no AugMix.
# - CIFAR    : ToTensor + Normalize only (strictly no augmentation)
# - ImageNet : RandomResizedCrop + RandomHorizontalFlip + Normalize
#              (standard clean baseline — used by all comparison methods)

# AugMix (--augmix flag)
# ----------------------
# Optional. Only activates when --augmix is explicitly passed.
# Never interferes with clean training. Used only for the
# "ShapeBiasNet + AugMix" ablation experiment.
# Checkpoint saved separately as <model>_<dataset>_augmix.pt

# Experiment groups (--run_group)
# --------------------------------
# Predefined groups so you can run a full round with one command:

#     cifar_all        : all models on CIFAR
#     imagenet_core    : shape_res50 + baseline_res50            (Round 1)
#     imagenet_deep    : shape_res101 + baseline_res101          (Round 2)
#     imagenet_convnext: shape_convnext_tiny + baseline_convnext (Round 3)
#     imagenet_effnet  : shape_effnet_b4 + baseline_effnet       (Round 4)
#     imagenet_all     : all imagenet rounds combined

# Usage
# -----
#     # Single model
#     python train.py --model shape_res18 --dataset cifar10
#     python train.py --model shape_res50 --dataset imagenet --data_dir /pvc/imagenet

#     # Run a predefined group
#     python train.py --run_group imagenet_core   --dataset imagenet --data_dir /pvc/imagenet
#     python train.py --run_group imagenet_deep   --dataset imagenet --data_dir /pvc/imagenet
#     python train.py --run_group imagenet_all    --dataset imagenet --data_dir /pvc/imagenet
#     python train.py --run_group cifar_all       --dataset cifar10

#     # Run all models for a dataset
#     python train.py --run_all --dataset cifar10
#     python train.py --run_all --dataset imagenet --data_dir /pvc/imagenet

#     # AugMix ablation (time permitting)
#     python train.py --model shape_res50 --dataset imagenet --augmix --data_dir /pvc/imagenet

# Checkpointing
# -------------
# Saves full state (model + optimizer + scheduler + epoch) on every
# accuracy improvement. Auto-resumes from existing checkpoint on restart.
# """

# import argparse, os, random, time
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as T
# from torch.utils.data import DataLoader
# from models import build_model, MODEL_NAMES

# # ──────────────────────────────────────────────────────────────
# #  EXPERIMENT GROUPS
# #  Predefined pairs/sets for each review round.
# #  Each group runs baseline + shape variant together.
# # ──────────────────────────────────────────────────────────────
# GROUPS = {
#     # CIFAR
#     "cifar_all":          ["baseline_res18", "baseline_res50",
#                            "shape_custom", "shape_res18", "shape_res50"],
#     # ImageNet — Round 1: core ResNet-50 results
#     "imagenet_core":      ["baseline_res50",        "shape_res50"],
#     # ImageNet — Round 2: deeper ResNet
#     "imagenet_deep":      ["baseline_res101",        "shape_res101"],
#     # ImageNet — Round 3: ConvNeXt
#     "imagenet_convnext":  ["baseline_convnext_tiny", "shape_convnext_tiny"],
#     # ImageNet — Round 4: EfficientNet
#     "imagenet_effnet":    ["baseline_efficientnet_b4", "shape_effnet_b4"],
#     # ImageNet — all rounds combined
#     "imagenet_all":       ["baseline_res50",          "shape_res50",
#                            "baseline_res101",          "shape_res101",
#                            "baseline_convnext_tiny",   "shape_convnext_tiny",
#                            "baseline_efficientnet_b4", "shape_effnet_b4"],
# }

# # ──────────────────────────────────────────────────────────────
# #  ARGS
# # ──────────────────────────────────────────────────────────────
# p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# p.add_argument("--model",       default="shape_res18", choices=MODEL_NAMES)
# p.add_argument("--dataset",     default="cifar10",
#                choices=["cifar10", "cifar100", "imagenet"])
# p.add_argument("--data_dir",    default="./data",
#                help="Data root. ImageNet expects <data_dir>/train and <data_dir>/val")
# p.add_argument("--ckpt_dir",    default="./checkpoints")
# p.add_argument("--epochs",      type=int,   default=None,
#                help="Default: 40 CIFAR / 90 ImageNet")
# p.add_argument("--batch",       type=int,   default=None,
#                help="Default: 128 CIFAR / 256 ImageNet")
# p.add_argument("--workers",     type=int,   default=None,
#                help="Default: 2 CIFAR / 8 ImageNet")
# p.add_argument("--lr",          type=float, default=0.1)
# p.add_argument("--accum_steps", type=int,   default=1,
#                help="Gradient accumulation steps")
# p.add_argument("--seed",        type=int,   default=42)

# # ── what to run ──────────────────────────────────────────────
# run_group = p.add_mutually_exclusive_group()
# run_group.add_argument("--run_all",   action="store_true",
#                        help="Train all MODEL_NAMES for --dataset")
# run_group.add_argument("--run_group", choices=list(GROUPS.keys()),
#                        help="Train a predefined experiment group")

# # ── augmix ablation ──────────────────────────────────────────
# p.add_argument("--augmix", action="store_true",
#                help="Use AugMix augmentation (ablation only). "
#                     "Saves checkpoint as <model>_<dataset>_augmix.pt. "
#                     "Does NOT affect clean training runs.")

# args = p.parse_args()

# IS_IMAGENET = args.dataset == "imagenet"
# EPOCHS      = args.epochs  or (90  if IS_IMAGENET else 40)
# BATCH       = args.batch   or (256 if IS_IMAGENET else 128)
# WORKERS     = args.workers or (8   if IS_IMAGENET else 2)
# NUM_CLASSES = 1000 if IS_IMAGENET else (10 if args.dataset == "cifar10" else 100)
# DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# os.makedirs(args.ckpt_dir, exist_ok=True)

# # Determine which models to run
# if args.run_all:
#     runs = MODEL_NAMES
# elif args.run_group:
#     runs = GROUPS[args.run_group]
# else:
#     runs = [args.model]


# # ──────────────────────────────────────────────────────────────
# #  REPRODUCIBILITY
# # ──────────────────────────────────────────────────────────────
# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True

# set_seed(args.seed)


# # ──────────────────────────────────────────────────────────────
# #  DATA
# # ──────────────────────────────────────────────────────────────
# MEAN = {"cifar10":  (0.4914, 0.4822, 0.4465),
#         "cifar100": (0.5071, 0.4867, 0.4408),
#         "imagenet": (0.485,  0.456,  0.406)}
# STD  = {"cifar10":  (0.247,  0.243,  0.261),
#         "cifar100": (0.2675, 0.2565, 0.2761),
#         "imagenet": (0.229,  0.224,  0.225)}
# mean, std = MEAN[args.dataset], STD[args.dataset]


# def get_transforms():
#     """
#     Returns (train_tf, val_tf) based on dataset and --augmix flag.

#     Clean training (default):
#         CIFAR    → ToTensor + Normalize only
#         ImageNet → RandomResizedCrop + HFlip + Normalize

#     AugMix (--augmix, ImageNet only):
#         Replaces RandomResizedCrop+HFlip with AugMix pipeline.
#         Only used when --augmix is explicitly passed.
#         CIFAR AugMix is not implemented — raises error if attempted.

#     val_tf is always clean regardless of --augmix.
#     """
#     if args.augmix:
#         if not IS_IMAGENET:
#             raise ValueError("--augmix is only supported for ImageNet. "
#                              "CIFAR clean-only training is the contribution.")
#         # AugMix augmentation pipeline
#         # Severity and mixture_width follow the original AugMix paper defaults
#         train_tf = T.Compose([
#             T.RandomResizedCrop(224),
#             T.RandomHorizontalFlip(),
#             T.AugMix(),             # torchvision AugMix (severity=3, mixture_width=3)
#             T.ToTensor(),
#             T.Normalize(mean, std),
#         ])
#         val_tf = T.Compose([
#             T.Resize(256), T.CenterCrop(224),
#             T.ToTensor(), T.Normalize(mean, std),
#         ])
#         return train_tf, val_tf

#     # ── clean training (default) ──────────────────────────────
#     if IS_IMAGENET:
#         # Standard clean ImageNet protocol used by all baselines.
#         # RandomResizedCrop + HFlip are NOT corruption augmentation.
#         train_tf = T.Compose([
#             T.RandomResizedCrop(224),
#             T.RandomHorizontalFlip(),
#             T.ToTensor(),
#             T.Normalize(mean, std),
#         ])
#         val_tf = T.Compose([
#             T.Resize(256), T.CenterCrop(224),
#             T.ToTensor(), T.Normalize(mean, std),
#         ])
#     else:
#         # CIFAR: strictly no augmentation
#         train_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
#         val_tf   = train_tf

#     return train_tf, val_tf


# train_tf, val_tf = get_transforms()

# if IS_IMAGENET:
#     trainset = torchvision.datasets.ImageFolder(
#         os.path.join(args.data_dir, "train"), transform=train_tf)
#     testset  = torchvision.datasets.ImageFolder(
#         os.path.join(args.data_dir, "val"),   transform=val_tf)
# else:
#     DS = (torchvision.datasets.CIFAR10 if args.dataset == "cifar10"
#           else torchvision.datasets.CIFAR100)
#     trainset = DS(args.data_dir, train=True,  download=True, transform=train_tf)
#     testset  = DS(args.data_dir, train=False, download=True, transform=val_tf)

# trainloader = DataLoader(trainset, BATCH,   shuffle=True,
#                          num_workers=WORKERS, pin_memory=True,
#                          persistent_workers=(WORKERS > 0))
# testloader  = DataLoader(testset,  BATCH*2, shuffle=False,
#                          num_workers=WORKERS, pin_memory=True,
#                          persistent_workers=(WORKERS > 0))

# # ── describe training mode ────────────────────────────────────
# TRAIN_MODE = "augmix" if args.augmix else "clean"
# if IS_IMAGENET:
#     TRAIN_DESC = "AugMix (ablation)" if args.augmix else "clean (RandomResizedCrop+HFlip+Normalize)"
# else:
#     TRAIN_DESC = "clean only (ToTensor+Normalize)"


# # ──────────────────────────────────────────────────────────────
# #  CLEAN EVAL
# # ──────────────────────────────────────────────────────────────
# @torch.no_grad()
# def evaluate(model: nn.Module) -> float:
#     model.eval()
#     correct = total = 0
#     for x, y in testloader:
#         x, y    = x.to(DEVICE), y.to(DEVICE)
#         correct += (model(x).argmax(1) == y).sum().item()
#         total   += y.size(0)
#     return 100.0 * correct / total


# # ──────────────────────────────────────────────────────────────
# #  TRAIN ONE MODEL
# # ──────────────────────────────────────────────────────────────
# def train_model(name: str) -> float:
#     set_seed(args.seed)

#     # Checkpoint path — augmix runs saved separately to avoid
#     # overwriting clean checkpoints
#     ckpt_suffix = f"_{TRAIN_MODE}" if args.augmix else ""
#     ckpt_path   = os.path.join(args.ckpt_dir,
#                                f"{name}_{args.dataset}{ckpt_suffix}.pt")

#     print(f"\n{'='*60}")
#     print(f"  Model      : {name}")
#     print(f"  Dataset    : {args.dataset}  ({NUM_CLASSES} classes)")
#     print(f"  Epochs     : {EPOCHS}  |  Batch: {BATCH}  |  Workers: {WORKERS}")
#     print(f"  Device     : {DEVICE}")
#     print(f"  Train mode : {TRAIN_DESC}")
#     print(f"  Checkpoint : {ckpt_path}")
#     print(f"{'='*60}")

#     model = build_model(name, NUM_CLASSES, dataset=args.dataset).to(DEVICE)
#     opt   = torch.optim.SGD(model.parameters(), lr=args.lr,
#                             momentum=0.9, weight_decay=1e-4, nesterov=True)
#     sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
#     crit  = nn.CrossEntropyLoss(label_smoothing=0.1)

#     start_ep = 1
#     best_acc = 0.0

#     # ── auto-resume on Nautilus preemption ──
#     if os.path.exists(ckpt_path):
#         print(f"  [Resume] Found checkpoint: {ckpt_path}")
#         state    = torch.load(ckpt_path, map_location="cpu")
#         model.load_state_dict(state["model"])
#         opt.load_state_dict(state["opt"])
#         sch.load_state_dict(state["sch"])
#         start_ep = state["epoch"] + 1
#         best_acc = state.get("best_acc", 0.0)
#         print(f"  [Resume] Continuing from epoch {start_ep}, best={best_acc:.2f}%")

#     if start_ep > EPOCHS:
#         print("  Already completed. Skipping.")
#         return best_acc

#     # ── training loop ──
#     opt.zero_grad(set_to_none=True)
#     for ep in range(start_ep, EPOCHS + 1):
#         t0 = time.time()
#         model.train()
#         loss_sum = 0.0

#         for step, (x, y) in enumerate(trainloader):
#             x, y  = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
#             loss  = crit(model(x), y) / args.accum_steps
#             loss.backward()
#             loss_sum += loss.item() * args.accum_steps

#             if (step + 1) % args.accum_steps == 0:
#                 nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#                 opt.step()
#                 opt.zero_grad(set_to_none=True)

#         sch.step()
#         acc     = evaluate(model)
#         elapsed = time.time() - t0

#         print(f"  [{ep:03d}/{EPOCHS}] "
#               f"loss={loss_sum/len(trainloader):.4f} | "
#               f"clean_acc={acc:.2f}% | "
#               f"lr={sch.get_last_lr()[0]:.5f} | "
#               f"{elapsed:.0f}s")

#         # ── save on improvement ──
#         if acc > best_acc:
#             best_acc = acc
#             torch.save({
#                 "epoch":      ep,
#                 "model":      model.state_dict(),
#                 "opt":        opt.state_dict(),
#                 "sch":        sch.state_dict(),
#                 "best_acc":   best_acc,
#                 "model_name": name,
#                 "dataset":    args.dataset,
#                 "train_mode": TRAIN_MODE,
#             }, ckpt_path)
#             print(f"  [Saved] best={best_acc:.2f}% → {ckpt_path}")

#     print(f"\n  Best clean accuracy: {best_acc:.2f}%")
#     return best_acc


# # ──────────────────────────────────────────────────────────────
# #  MAIN
# # ──────────────────────────────────────────────────────────────
# print(f"\n  Running {len(runs)} model(s): {runs}")
# print(f"  Dataset: {args.dataset}  |  Mode: {TRAIN_DESC}")

# for name in runs:
#     train_model(name)




"""
train.py — Unified training script (CIFAR-10 / CIFAR-100 / ImageNet).

Training protocol
-----------------
CLEAN DATA ONLY by default. No corruption augmentation, no AugMix.
- CIFAR    : ToTensor + Normalize only (strictly no augmentation)
- ImageNet : RandomResizedCrop + RandomHorizontalFlip + Normalize
             (standard clean baseline — used by all comparison methods)

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

    cifar_all        : all models on CIFAR
    imagenet_core    : shape_res50 + baseline_res50            (Round 1)
    imagenet_deep    : shape_res101 + baseline_res101          (Round 2)
    imagenet_convnext: shape_convnext_tiny + baseline_convnext (Round 3)
    imagenet_effnet  : shape_effnet_b4 + baseline_effnet       (Round 4)
    imagenet_all     : all imagenet rounds combined

Usage
-----
    # Single model
    python train.py --model shape_res18 --dataset cifar10
    python train.py --model shape_res50 --dataset imagenet --data_dir /pvc/imagenet

    # Run a predefined group
    python train.py --run_group imagenet_core   --dataset imagenet --data_dir /pvc/imagenet
    python train.py --run_group imagenet_deep   --dataset imagenet --data_dir /pvc/imagenet
    python train.py --run_group imagenet_all    --dataset imagenet --data_dir /pvc/imagenet
    python train.py --run_group cifar_all       --dataset cifar10

    # Run all models for a dataset
    python train.py --run_all --dataset cifar10
    python train.py --run_all --dataset imagenet --data_dir /pvc/imagenet

    # AugMix ablation (time permitting)
    python train.py --model shape_res50 --dataset imagenet --augmix --data_dir /pvc/imagenet

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
    "cifar_all":          ["baseline_res18", "baseline_res50",
                           "shape_custom", "shape_res18", "shape_res50"],
    "imagenet_core":      ["baseline_res50",        "shape_res50"],
    "imagenet_deep":      ["baseline_res101",        "shape_res101"],
    "imagenet_convnext":  ["baseline_convnext_tiny", "shape_convnext_tiny"],
    "imagenet_effnet":    ["baseline_efficientnet_b4", "shape_effnet_b4"],
    "imagenet_all":       ["baseline_res50",          "shape_res50",
                           "baseline_res101",          "shape_res101",
                           "baseline_convnext_tiny",   "shape_convnext_tiny",
                           "baseline_efficientnet_b4", "shape_effnet_b4"],
}

# ──────────────────────────────────────────────────────────────
#  ARGS
# ──────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--model",       default="shape_res18", choices=MODEL_NAMES)
p.add_argument("--dataset",     default="cifar10",
               choices=["cifar10", "cifar100", "imagenet"])
p.add_argument("--data_dir",    default="./data",
               help="Data root. ImageNet expects <data_dir>/train and <data_dir>/val")
p.add_argument("--ckpt_dir",    default="./checkpoints")
p.add_argument("--epochs",      type=int,   default=None,
               help="Default: 40 CIFAR / 90 ImageNet")
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

IS_IMAGENET = args.dataset == "imagenet"
EPOCHS      = args.epochs  or (90  if IS_IMAGENET else 40)
BATCH       = args.batch   or (256 if IS_IMAGENET else 128)
WORKERS     = args.workers or (8   if IS_IMAGENET else 2)
NUM_CLASSES = 1000 if IS_IMAGENET else (10 if args.dataset == "cifar10" else 100)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS    = torch.cuda.device_count()

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
MEAN = {"cifar10":  (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "imagenet": (0.485,  0.456,  0.406)}
STD  = {"cifar10":  (0.247,  0.243,  0.261),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "imagenet": (0.229,  0.224,  0.225)}
mean, std = MEAN[args.dataset], STD[args.dataset]


def get_transforms():
    if args.augmix:
        if not IS_IMAGENET:
            raise ValueError("--augmix is only supported for ImageNet.")
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
    trainset = torchvision.datasets.ImageFolder(
        os.path.join(args.data_dir, "train"), transform=train_tf)
    testset  = torchvision.datasets.ImageFolder(
        os.path.join(args.data_dir, "val"),   transform=val_tf)
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
    model = build_model(name, NUM_CLASSES, dataset=args.dataset).to(DEVICE)

    # ── wrap with DataParallel if multiple GPUs available ────
    if NUM_GPUS > 1:
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
        # Load into underlying model (works whether wrapped or not)
        unwrap(model).load_state_dict(state["model"])
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
        # Always save unwrap(model).state_dict() so checkpoint is
        # portable — loadable with or without DataParallel.
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

for name in runs:
    train_model(name)