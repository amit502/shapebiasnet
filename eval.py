# import torch, argparse, numpy as np, torchvision, os
# import torchvision.transforms as T
# from models import build_model

# p=argparse.ArgumentParser()
# p.add_argument("--model", default="shape_custom")
# p.add_argument("--dataset", default="cifar10")
# p.add_argument("--ckpt", default=None)
# p.add_argument("--eval_all", action="store_true")
# args=p.parse_args()

# DEVICE="cuda" if torch.cuda.is_available() else "cpu"

# C = ['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur',
#      'motion_blur','zoom_blur','snow','frost','fog','brightness','contrast',
#      'elastic_transform','pixelate','jpeg_compression']

# ALL_MODELS = ["baseline_res18","baseline_res50","shape_custom","shape_res18","shape_res50"]

# mean10,std10=(0.4914,0.4822,0.4465),(0.247,0.243,0.261)
# mean100,std100=(0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)
# mean,std = (mean10,std10) if args.dataset=="cifar10" else (mean100,std100)

# # -------------------------------------------------
# # CIFAR-C folder root
# # -------------------------------------------------
# CROOT = "./data/CIFAR-10-C" if args.dataset=="cifar10" else "./data/CIFAR-100-C"

# # -------- clean ----------
# tf=T.Compose([T.ToTensor(),T.Normalize(mean,std)])
# DS = torchvision.datasets.CIFAR10 if args.dataset=="cifar10" else torchvision.datasets.CIFAR100
# clean = torch.utils.data.DataLoader(
#     DS("./data",False,download=True,transform=tf),
#     256,shuffle=False,num_workers=2)

# # -------- helpers ----------
# def prep(x):
#     x=torch.from_numpy(x).float().permute(0,3,1,2)/255.
#     m=torch.tensor(mean).view(1,3,1,1); s=torch.tensor(std).view(1,3,1,1)
#     return (x-m)/s

# @torch.no_grad()
# def clean_eval(model):
#     c=t=0
#     for x,y in clean:
#         x,y=x.to(DEVICE),y.to(DEVICE)
#         c+=(model(x).argmax(1)==y).sum().item(); t+=y.size(0)
#     return 100*c/t

# @torch.no_grad()
# def eval_sub(model,x,y):
#     c=t=0
#     for i in range(0,len(x),256):
#         xb=prep(x[i:i+256]).to(DEVICE)
#         yb=torch.from_numpy(y[i:i+256]).to(DEVICE)
#         c+=(model(xb).argmax(1)==yb).sum().item(); t+=yb.size(0)
#     return 100*c/t

# # -------------------------------------------------
# # Evaluation runner
# # -------------------------------------------------
# def run(model_name, ckpt_path):

#     print("\n========================================")
#     print("Model:", model_name)
#     print("Ckpt :", ckpt_path)
#     print("========================================")

#     num_classes = 10 if args.dataset=="cifar10" else 100
#     model = build_model(model_name, num_classes)
#     if model_name.startswith("shape"): model.alpha=1.0
#     model.load_state_dict(torch.load(ckpt_path,map_location="cpu"))
#     model=model.to(DEVICE).eval()

#     print("\nClean accuracy:", round(clean_eval(model),2))
#     print("\n--- CIFAR-C ---")

#     mca=[]
#     for c in C:
#         x=np.load(f"{CROOT}/{c}.npy"); y=np.load(f"{CROOT}/labels.npy")
#         sev=[]
#         for s in range(5):
#             sev.append(eval_sub(model,
#                         x[s*10000:(s+1)*10000],
#                         y[s*10000:(s+1)*10000]))
#         print(f"{c:18s} | {[round(a,2) for a in sev]} | mean {sum(sev)/5:.2f}")
#         mca.append(sum(sev)/5)

#     print("\n--------------------------------")
#     print("mCA:", round(sum(mca)/len(mca),2))
#     print("mCE:", round(100-sum(mca)/len(mca),2))
#     print("--------------------------------")

# # -------------------------------------------------
# # Main
# # -------------------------------------------------
# if args.eval_all:
#     for m in ALL_MODELS:
#         ckpt = f"checkpoints/{m}_{args.dataset}.pt"
#         if os.path.exists(ckpt):
#             run(m, ckpt)
#         else:
#             print(f"[SKIP] {ckpt} not found")

# else:
#     if args.ckpt is None:
#         raise ValueError("Provide --ckpt or use --eval_all")
#     run(args.model, args.ckpt)





# """
# eval.py — Evaluation on clean data and corruption benchmarks.

# Two evaluations, clearly separated:
#   1. CLEAN accuracy  — standard test set (CIFAR test / ImageNet val)
#   2. CORRUPT accuracy — CIFAR-C or ImageNet-C, all 15 corruptions × 5 severities

# Results printed in the same table format as the original script, and
# saved to --results_dir as JSON + txt so evaluation never needs re-running.

# CIFAR-C layout expected:
#     <data_dir>/CIFAR-10-C/<corruption>.npy
#     <data_dir>/CIFAR-10-C/labels.npy

# ImageNet-C layout expected:
#     <data_dir>/ImageNet-C/<corruption>/<severity 1-5>/<class>/<img>.JPEG

# Usage
# -----
#     # Single model
#     python eval.py --model shape_res18 --dataset cifar10
#     python eval.py --model shape_res18 --dataset imagenet --data_dir /pvc/imagenet

#     # All available checkpoints
#     python eval.py --dataset cifar10  --eval_all
#     python eval.py --dataset imagenet --eval_all --data_dir /pvc/imagenet
# """

# import argparse, json, os, time
# from datetime import datetime
# from pathlib import Path

# import numpy as np
# import torch
# import torchvision
# import torchvision.transforms as T
# from torch.utils.data import DataLoader, Dataset
# from models import build_model, MODEL_NAMES

# # ──────────────────────────────────────────────────────────────
# #  ARGS
# # ──────────────────────────────────────────────────────────────
# p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# p.add_argument("--model",       default="shape_res18", choices=MODEL_NAMES)
# p.add_argument("--dataset",     default="cifar10",     choices=["cifar10","cifar100","imagenet"])
# p.add_argument("--data_dir",    default="./data")
# p.add_argument("--ckpt_dir",    default="./checkpoints")
# p.add_argument("--results_dir", default="./results",
#                help="Where to save JSON + txt reports (persists to PVC)")
# p.add_argument("--ckpt",        default=None, help="Explicit checkpoint path")
# p.add_argument("--batch",       type=int, default=256)
# p.add_argument("--workers",     type=int, default=4)
# p.add_argument("--eval_all",    action="store_true",
#                help="Evaluate all checkpoints found in --ckpt_dir")
# p.add_argument("--clean_only",   action="store_true",
#                help="Run clean evaluation only (skip corruption eval)")
# p.add_argument("--corrupt_only", action="store_true",
#                help="Run corruption evaluation only (skip clean eval)")
# args = p.parse_args()

# IS_IMAGENET = args.dataset == "imagenet"
# DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_CLASSES = 1000 if IS_IMAGENET else (10 if args.dataset == "cifar10" else 100)

# os.makedirs(args.results_dir, exist_ok=True)

# # ──────────────────────────────────────────────────────────────
# #  NORMALIZATION STATS
# # ──────────────────────────────────────────────────────────────
# MEAN = {"cifar10":  (0.4914, 0.4822, 0.4465),
#         "cifar100": (0.5071, 0.4867, 0.4408),
#         "imagenet": (0.485,  0.456,  0.406)}
# STD  = {"cifar10":  (0.247,  0.243,  0.261),
#         "cifar100": (0.2675, 0.2565, 0.2761),
#         "imagenet": (0.229,  0.224,  0.225)}
# mean, std = MEAN[args.dataset], STD[args.dataset]

# CORRUPTIONS = [
#     "gaussian_noise", "shot_noise",       "impulse_noise",
#     "defocus_blur",   "glass_blur",       "motion_blur",    "zoom_blur",
#     "snow",           "frost",            "fog",            "brightness",
#     "contrast",       "elastic_transform","pixelate",       "jpeg_compression",
# ]

# # ──────────────────────────────────────────────────────────────
# #  CLEAN DATA LOADER
# # ──────────────────────────────────────────────────────────────
# if IS_IMAGENET:
#     clean_tf = T.Compose([T.Resize(256), T.CenterCrop(224),
#                           T.ToTensor(), T.Normalize(mean, std)])
#     clean_ds = torchvision.datasets.ImageFolder(
#         os.path.join(args.data_dir, "val"), transform=clean_tf)
# else:
#     clean_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
#     DS       = (torchvision.datasets.CIFAR10 if args.dataset == "cifar10"
#                 else torchvision.datasets.CIFAR100)
#     clean_ds = DS(args.data_dir, train=False, download=True, transform=clean_tf)

# clean_loader = DataLoader(clean_ds, args.batch, shuffle=False,
#                           num_workers=args.workers, pin_memory=True)

# # ──────────────────────────────────────────────────────────────
# #  IMAGENET-C DATASET WRAPPER
# # ──────────────────────────────────────────────────────────────
# class ImageNetCDataset(Dataset):
#     """Single (corruption, severity) folder as a standard Dataset."""
#     def __init__(self, root: str, corruption: str, severity: int, transform):
#         path     = os.path.join(root, "ImageNet-C", corruption, str(severity))
#         self.ds  = torchvision.datasets.ImageFolder(path, transform=transform)
#     def __len__(self):        return len(self.ds)
#     def __getitem__(self, i): return self.ds[i]


# # ──────────────────────────────────────────────────────────────
# #  EVAL HELPERS
# # ──────────────────────────────────────────────────────────────
# def prep_cifar_npy(x: np.ndarray) -> torch.Tensor:
#     """uint8 numpy (N,H,W,C) → normalised float tensor (N,C,H,W)."""
#     t = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 255.0
#     m = torch.tensor(mean).view(1, 3, 1, 1)
#     s = torch.tensor(std).view(1, 3, 1, 1)
#     return (t - m) / s

# @torch.no_grad()
# def eval_loader(model, loader) -> float:
#     correct = total = 0
#     for x, y in loader:
#         x, y    = x.to(DEVICE), y.to(DEVICE)
#         correct += (model(x).argmax(1) == y).sum().item()
#         total   += y.size(0)
#     return 100.0 * correct / total

# @torch.no_grad()
# def eval_cifar_npy(model, x_np: np.ndarray, y_np: np.ndarray) -> float:
#     correct = total = 0
#     for i in range(0, len(x_np), args.batch):
#         xb = prep_cifar_npy(x_np[i:i+args.batch]).to(DEVICE)
#         yb = torch.from_numpy(y_np[i:i+args.batch]).to(DEVICE)
#         correct += (model(xb).argmax(1) == yb).sum().item()
#         total   += yb.size(0)
#     return 100.0 * correct / total


# # ──────────────────────────────────────────────────────────────
# #  SAVE RESULTS
# # ──────────────────────────────────────────────────────────────
# def save_results(model_name: str, results: dict, lines: list):
#     """
#     Save to results_dir:
#       <model>_<dataset>_results.json  — full structured data
#       <model>_<dataset>_report.txt    — identical to stdout output
#     """
#     stem     = f"{model_name}_{args.dataset}"
#     json_path = os.path.join(args.results_dir, f"{stem}_results.json")
#     txt_path  = os.path.join(args.results_dir, f"{stem}_report.txt")

#     with open(json_path, "w") as f:
#         json.dump(results, f, indent=2)
#     with open(txt_path, "w") as f:
#         f.write("\n".join(lines) + "\n")

#     print(f"  [Saved] {json_path}")
#     print(f"  [Saved] {txt_path}")


# # ──────────────────────────────────────────────────────────────
# #  MAIN EVAL RUNNER
# # ──────────────────────────────────────────────────────────────
# def run(model_name: str, ckpt_path: str) -> dict:
#     lines = []
#     def out(s: str = ""):
#         print(s); lines.append(s)

#     out(f"\n{'='*60}")
#     out(f"  Model   : {model_name}")
#     out(f"  Dataset : {args.dataset}")
#     out(f"  Ckpt    : {ckpt_path}")
#     out(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     out(f"{'='*60}")

#     # ── load model ──
#     model = build_model(model_name, NUM_CLASSES, dataset=args.dataset)
#     state = torch.load(ckpt_path, map_location="cpu")
#     model.load_state_dict(state["model"] if "model" in state else state)
#     model = model.to(DEVICE).eval()

#     # ──────────────────────────────────────────────────────────
#     #  CLEAN EVALUATION
#     # ──────────────────────────────────────────────────────────
#     clean_acc      = None
#     per_corruption = {}
#     mca = mce      = None

#     if not args.corrupt_only:
#         out(f"\n  {'─'*40}")
#         out(f"  CLEAN EVALUATION")
#         out(f"  {'─'*40}")
#         t0        = time.time()
#         clean_acc = eval_loader(model, clean_loader)
#         out(f"  Clean accuracy : {clean_acc:.2f}%   ({time.time()-t0:.1f}s)")

#     # ──────────────────────────────────────────────────────────
#     #  CORRUPTION EVALUATION
#     # ──────────────────────────────────────────────────────────
#     if not args.clean_only:
#         benchmark = ("CIFAR-10-C"  if args.dataset == "cifar10"  else
#                      "CIFAR-100-C" if args.dataset == "cifar100" else
#                      "ImageNet-C")
#         out(f"\n  {'─'*40}")
#         out(f"  CORRUPTION EVALUATION  ({benchmark})")
#         out(f"  {'─'*40}")
#         out(f"  {'Corruption':<22} | {'s1':>5} {'s2':>5} {'s3':>5} {'s4':>5} {'s5':>5} | {'mean':>6}")
#         out(f"  {'-'*58}")

#         mca_vals = []
#         for corr in CORRUPTIONS:
#             sev_accs = []
#             t0 = time.time()

#             if IS_IMAGENET:
#                 for sev in range(1, 6):
#                     try:
#                         ds  = ImageNetCDataset(args.data_dir, corr, sev, clean_tf)
#                         ldr = DataLoader(ds, args.batch, shuffle=False,
#                                          num_workers=args.workers, pin_memory=True)
#                         sev_accs.append(eval_loader(model, ldr))
#                     except (FileNotFoundError, RuntimeError):
#                         out(f"  [WARN] Missing: ImageNet-C/{corr}/{sev}")
#                         sev_accs.append(float("nan"))
#             else:
#                 croot = (f"{args.data_dir}/CIFAR-10-C"  if args.dataset == "cifar10"
#                          else f"{args.data_dir}/CIFAR-100-C")
#                 x_all = np.load(f"{croot}/{corr}.npy")
#                 y_all = np.load(f"{croot}/labels.npy")
#                 for sev in range(5):
#                     sev_accs.append(eval_cifar_npy(
#                         model,
#                         x_all[sev*10000:(sev+1)*10000],
#                         y_all[sev*10000:(sev+1)*10000],
#                     ))

#             valid    = [a for a in sev_accs if not np.isnan(a)]
#             mean_acc = float(np.mean(valid)) if valid else float("nan")
#             mca_vals.append(mean_acc)

#             sev_str = " ".join(f"{a:5.1f}" for a in sev_accs)
#             out(f"  {corr:<22} | {sev_str} | {mean_acc:6.2f}   ({time.time()-t0:.1f}s)")

#             per_corruption[corr] = {
#                 **{f"s{i+1}": round(sev_accs[i], 4) for i in range(5)},
#                 "mean": round(mean_acc, 4),
#             }

#         valid_mca = [a for a in mca_vals if not np.isnan(a)]
#         mca       = float(np.mean(valid_mca))
#         mce       = 100.0 - mca

#     # ── summary ──
#     out(f"\n  {'='*58}")
#     if clean_acc is not None:
#         out(f"  Clean accuracy : {clean_acc:.2f}%")
#     if mca is not None:
#         out(f"  mCA            : {mca:.2f}%   (mean Corruption Accuracy)")
#         out(f"  mCE            : {mce:.2f}%   (mean Corruption Error = 100 - mCA)")
#     out(f"  {'='*58}")

#     results = {
#         "model":           model_name,
#         "dataset":         args.dataset,
#         "checkpoint":      ckpt_path,
#         "timestamp":       datetime.now().isoformat(),
#         "clean_acc":       round(clean_acc, 4) if clean_acc is not None else None,
#         "mCA":             round(mca, 4)       if mca       is not None else None,
#         "mCE":             round(mce, 4)       if mce       is not None else None,
#         "per_corruption":  per_corruption,
#     }
#     save_results(model_name, results, lines)
#     return results


# # ──────────────────────────────────────────────────────────────
# #  MAIN
# # ──────────────────────────────────────────────────────────────
# if args.eval_all:
#     all_results = {}
#     for m in MODEL_NAMES:
#         ckpt = os.path.join(args.ckpt_dir, f"{m}_{args.dataset}.pt")
#         if os.path.exists(ckpt):
#             all_results[m] = run(m, ckpt)
#         else:
#             print(f"  [SKIP] {ckpt} not found")

#     # ── summary table ──
#     print(f"\n{'='*62}")
#     print(f"  SUMMARY — {args.dataset.upper()}")
#     print(f"  {'Model':<22} | {'Clean':>6} | {'mCA':>6} | {'mCE':>6}")
#     print(f"  {'-'*60}")
#     for m, r in all_results.items():
#         print(f"  {m:<22} | {r['clean_acc']:6.2f} | {r['mCA']:6.2f} | {r['mCE']:6.2f}")
#     print(f"{'='*62}")

#     # save summary
#     summary_path = os.path.join(args.results_dir, f"summary_{args.dataset}.json")
#     with open(summary_path, "w") as f:
#         json.dump(all_results, f, indent=2)
#     print(f"\n  [Saved] {summary_path}")

# else:
#     ckpt = args.ckpt or os.path.join(args.ckpt_dir, f"{args.model}_{args.dataset}.pt")
#     if not os.path.exists(ckpt):
#         raise FileNotFoundError(
#             f"No checkpoint at {ckpt}\n"
#             "Provide --ckpt <path> or use --eval_all"
#         )
#     run(args.model, ckpt)




# """
# eval.py — Evaluation on clean data and corruption benchmarks.

# Two evaluations, clearly separated:
#   1. CLEAN accuracy  — standard test set (CIFAR test / ImageNet val)
#   2. CORRUPT accuracy — CIFAR-C or ImageNet-C, 15 corruptions × 5 severities

# Results printed in the same table format as the original script, and
# saved to --results_dir as JSON + txt so evaluation never needs re-running.

# Evaluation groups (--eval_group)
# ---------------------------------
# Mirror train.py groups so you can eval a round's results in one command:

#     cifar_all         : all CIFAR models
#     imagenet_core     : shape_res50 + baseline_res50
#     imagenet_deep     : shape_res101 + baseline_res101
#     imagenet_convnext : shape_convnext_tiny + baseline_convnext_tiny
#     imagenet_effnet   : shape_effnet_b4 + baseline_efficientnet_b4
#     imagenet_all      : all imagenet rounds combined

# CIFAR-C layout expected:
#     <data_dir>/CIFAR-10-C/<corruption>.npy
#     <data_dir>/CIFAR-10-C/labels.npy

# ImageNet-C layout expected:
#     <data_dir>/ImageNet-C/<corruption>/<severity 1-5>/<class>/<img>.JPEG

# Usage
# -----
#     # Single model
#     python eval.py --model shape_res18 --dataset cifar10
#     python eval.py --model shape_res50 --dataset imagenet --data_dir /pvc/imagenet

#     # Eval a predefined group
#     python eval.py --eval_group imagenet_core --dataset imagenet --data_dir /pvc/imagenet
#     python eval.py --eval_group cifar_all     --dataset cifar10

#     # All available checkpoints
#     python eval.py --eval_all --dataset cifar10
#     python eval.py --eval_all --dataset imagenet --data_dir /pvc/imagenet

#     # AugMix ablation checkpoint
#     python eval.py --model shape_res50 --dataset imagenet --augmix --data_dir /pvc/imagenet

#     # Clean only / corrupt only
#     python eval.py --eval_group imagenet_core --dataset imagenet --clean_only
#     python eval.py --eval_group imagenet_core --dataset imagenet --corrupt_only
# """

# import argparse, json, os, time
# from datetime import datetime

# import numpy as np
# import torch
# import torchvision
# import torchvision.transforms as T
# from torch.utils.data import DataLoader, Dataset
# from models import build_model, MODEL_NAMES

# # ──────────────────────────────────────────────────────────────
# #  EVAL GROUPS  (mirrors train.py GROUPS)
# # ──────────────────────────────────────────────────────────────
# GROUPS = {
#     "cifar_all":          ["baseline_res18", "baseline_res50",
#                            "shape_custom", "shape_res18", "shape_res50"],
#     "imagenet_core":      ["baseline_res50",           "shape_res50"],
#     "imagenet_deep":      ["baseline_res101",           "shape_res101"],
#     "imagenet_convnext":  ["baseline_convnext_tiny",    "shape_convnext_tiny"],
#     "imagenet_effnet":    ["baseline_efficientnet_b4",  "shape_effnet_b4"],
#     "imagenet_all":       ["baseline_res50",            "shape_res50",
#                            "baseline_res101",           "shape_res101",
#                            "baseline_convnext_tiny",    "shape_convnext_tiny",
#                            "baseline_efficientnet_b4",  "shape_effnet_b4"],
# }

# # ──────────────────────────────────────────────────────────────
# #  ARGS
# # ──────────────────────────────────────────────────────────────
# p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# p.add_argument("--model",       default="shape_res18", choices=MODEL_NAMES)
# p.add_argument("--dataset",     default="cifar10",
#                choices=["cifar10", "cifar100", "imagenet"])
# p.add_argument("--data_dir",    default="./data")
# p.add_argument("--ckpt_dir",    default="./checkpoints")
# p.add_argument("--results_dir", default="./results",
#                help="Where to save JSON + txt reports")
# p.add_argument("--ckpt",        default=None,
#                help="Explicit checkpoint path (overrides --ckpt_dir lookup)")
# p.add_argument("--batch",       type=int, default=256)
# p.add_argument("--workers",     type=int, default=4)

# # ── what to eval ─────────────────────────────────────────────
# eval_group = p.add_mutually_exclusive_group()
# eval_group.add_argument("--eval_all",   action="store_true",
#                         help="Evaluate all MODEL_NAMES checkpoints in --ckpt_dir")
# eval_group.add_argument("--eval_group", choices=list(GROUPS.keys()),
#                         help="Evaluate a predefined experiment group")

# # ── eval scope ───────────────────────────────────────────────
# p.add_argument("--clean_only",   action="store_true",
#                help="Run clean evaluation only (skip corruption eval)")
# p.add_argument("--corrupt_only", action="store_true",
#                help="Run corruption evaluation only (skip clean eval)")

# # ── augmix checkpoint ─────────────────────────────────────────
# p.add_argument("--augmix", action="store_true",
#                help="Load AugMix checkpoint (<model>_<dataset>_augmix.pt) "
#                     "instead of clean checkpoint.")

# args = p.parse_args()

# IS_IMAGENET = args.dataset == "imagenet"
# DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_CLASSES = 1000 if IS_IMAGENET else (10 if args.dataset == "cifar10" else 100)
# CKPT_SUFFIX = "_augmix" if args.augmix else ""

# os.makedirs(args.results_dir, exist_ok=True)

# # ──────────────────────────────────────────────────────────────
# #  NORMALIZATION STATS
# # ──────────────────────────────────────────────────────────────
# MEAN = {"cifar10":  (0.4914, 0.4822, 0.4465),
#         "cifar100": (0.5071, 0.4867, 0.4408),
#         "imagenet": (0.485,  0.456,  0.406)}
# STD  = {"cifar10":  (0.247,  0.243,  0.261),
#         "cifar100": (0.2675, 0.2565, 0.2761),
#         "imagenet": (0.229,  0.224,  0.225)}
# mean, std = MEAN[args.dataset], STD[args.dataset]

# CORRUPTIONS = [
#     "gaussian_noise", "shot_noise",        "impulse_noise",
#     "defocus_blur",   "glass_blur",        "motion_blur",    "zoom_blur",
#     "snow",           "frost",             "fog",            "brightness",
#     "contrast",       "elastic_transform", "pixelate",       "jpeg_compression",
# ]

# # ──────────────────────────────────────────────────────────────
# #  CLEAN DATA LOADER
# # ──────────────────────────────────────────────────────────────
# if IS_IMAGENET:
#     clean_tf = T.Compose([T.Resize(256), T.CenterCrop(224),
#                           T.ToTensor(), T.Normalize(mean, std)])
#     clean_ds = torchvision.datasets.ImageFolder(
#         os.path.join(args.data_dir, "val"), transform=clean_tf)
# else:
#     clean_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
#     DS       = (torchvision.datasets.CIFAR10 if args.dataset == "cifar10"
#                 else torchvision.datasets.CIFAR100)
#     clean_ds = DS(args.data_dir, train=False, download=True, transform=clean_tf)

# clean_loader = DataLoader(clean_ds, args.batch, shuffle=False,
#                           num_workers=args.workers, pin_memory=True)


# # ──────────────────────────────────────────────────────────────
# #  IMAGENET-C DATASET WRAPPER
# # ──────────────────────────────────────────────────────────────
# class ImageNetCDataset(Dataset):
#     """Single (corruption, severity) folder as a standard Dataset."""
#     def __init__(self, root: str, corruption: str, severity: int, transform):
#         path    = os.path.join(root, "ImageNet-C", corruption, str(severity))
#         self.ds = torchvision.datasets.ImageFolder(path, transform=transform)

#     def __len__(self):        return len(self.ds)
#     def __getitem__(self, i): return self.ds[i]


# # ──────────────────────────────────────────────────────────────
# #  EVAL HELPERS
# # ──────────────────────────────────────────────────────────────
# def prep_cifar_npy(x: np.ndarray) -> torch.Tensor:
#     """uint8 numpy (N,H,W,C) → normalised float tensor (N,C,H,W)."""
#     t = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 255.0
#     m = torch.tensor(mean).view(1, 3, 1, 1)
#     s = torch.tensor(std).view(1, 3, 1, 1)
#     return (t - m) / s


# @torch.no_grad()
# def eval_loader(model, loader) -> float:
#     correct = total = 0
#     for x, y in loader:
#         x, y    = x.to(DEVICE), y.to(DEVICE)
#         correct += (model(x).argmax(1) == y).sum().item()
#         total   += y.size(0)
#     return 100.0 * correct / total


# @torch.no_grad()
# def eval_cifar_npy(model, x_np: np.ndarray, y_np: np.ndarray) -> float:
#     correct = total = 0
#     for i in range(0, len(x_np), args.batch):
#         xb = prep_cifar_npy(x_np[i:i+args.batch]).to(DEVICE)
#         yb = torch.from_numpy(y_np[i:i+args.batch]).to(DEVICE)
#         correct += (model(xb).argmax(1) == yb).sum().item()
#         total   += yb.size(0)
#     return 100.0 * correct / total


# # ──────────────────────────────────────────────────────────────
# #  SAVE RESULTS
# # ──────────────────────────────────────────────────────────────
# def save_results(model_name: str, results: dict, lines: list):
#     """
#     Saves two files to --results_dir:
#       <model>_<dataset>[_augmix]_results.json  — full structured data
#       <model>_<dataset>[_augmix]_report.txt    — identical to stdout
#     """
#     stem      = f"{model_name}_{args.dataset}{CKPT_SUFFIX}"
#     json_path = os.path.join(args.results_dir, f"{stem}_results.json")
#     txt_path  = os.path.join(args.results_dir, f"{stem}_report.txt")

#     with open(json_path, "w") as f:
#         json.dump(results, f, indent=2)
#     with open(txt_path, "w") as f:
#         f.write("\n".join(lines) + "\n")

#     print(f"  [Saved] {json_path}")
#     print(f"  [Saved] {txt_path}")


# # ──────────────────────────────────────────────────────────────
# #  CHECKPOINT LOOKUP
# # ──────────────────────────────────────────────────────────────
# def ckpt_path_for(model_name: str) -> str:
#     """Return the checkpoint path for a model, respecting --augmix flag."""
#     return os.path.join(args.ckpt_dir,
#                         f"{model_name}_{args.dataset}{CKPT_SUFFIX}.pt")


# # ──────────────────────────────────────────────────────────────
# #  MAIN EVAL RUNNER
# # ──────────────────────────────────────────────────────────────
# def run(model_name: str, ckpt_path: str) -> dict:
#     lines = []

#     def out(s: str = ""):
#         print(s)
#         lines.append(s)

#     out(f"\n{'='*65}")
#     out(f"  Model      : {model_name}")
#     out(f"  Dataset    : {args.dataset}")
#     out(f"  Train mode : {'augmix' if args.augmix else 'clean'}")
#     out(f"  Ckpt       : {ckpt_path}")
#     out(f"  Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     out(f"{'='*65}")

#     # ── load model ──
#     model = build_model(model_name, NUM_CLASSES, dataset=args.dataset)
#     state = torch.load(ckpt_path, map_location="cpu")
#     model.load_state_dict(state["model"] if "model" in state else state)
#     model = model.to(DEVICE).eval()

#     clean_acc      = None
#     per_corruption = {}
#     mca = mce      = None

#     # ──────────────────────────────────────────────────────────
#     #  CLEAN EVALUATION
#     # ──────────────────────────────────────────────────────────
#     if not args.corrupt_only:
#         out(f"\n  {'─'*45}")
#         out(f"  CLEAN EVALUATION")
#         out(f"  {'─'*45}")
#         t0        = time.time()
#         clean_acc = eval_loader(model, clean_loader)
#         out(f"  Clean accuracy : {clean_acc:.2f}%   ({time.time()-t0:.1f}s)")

#     # ──────────────────────────────────────────────────────────
#     #  CORRUPTION EVALUATION
#     # ──────────────────────────────────────────────────────────
#     if not args.clean_only:
#         benchmark = ("CIFAR-10-C"  if args.dataset == "cifar10"  else
#                      "CIFAR-100-C" if args.dataset == "cifar100" else
#                      "ImageNet-C")
#         out(f"\n  {'─'*45}")
#         out(f"  CORRUPTION EVALUATION  ({benchmark})")
#         out(f"  {'─'*45}")
#         out(f"  {'Corruption':<22} | {'s1':>5} {'s2':>5} {'s3':>5} {'s4':>5} {'s5':>5} | {'mean':>6}")
#         out(f"  {'-'*60}")

#         mca_vals = []
#         for corr in CORRUPTIONS:
#             sev_accs = []
#             t0 = time.time()

#             if IS_IMAGENET:
#                 for sev in range(1, 6):
#                     try:
#                         ds  = ImageNetCDataset(args.data_dir, corr, sev, clean_tf)
#                         ldr = DataLoader(ds, args.batch, shuffle=False,
#                                          num_workers=args.workers, pin_memory=True)
#                         sev_accs.append(eval_loader(model, ldr))
#                     except (FileNotFoundError, RuntimeError):
#                         out(f"  [WARN] Missing: ImageNet-C/{corr}/{sev}")
#                         sev_accs.append(float("nan"))
#             else:
#                 croot = (f"{args.data_dir}/CIFAR-10-C"  if args.dataset == "cifar10"
#                          else f"{args.data_dir}/CIFAR-100-C")
#                 x_all = np.load(f"{croot}/{corr}.npy")
#                 y_all = np.load(f"{croot}/labels.npy")
#                 for sev in range(5):
#                     sev_accs.append(eval_cifar_npy(
#                         model,
#                         x_all[sev*10000:(sev+1)*10000],
#                         y_all[sev*10000:(sev+1)*10000],
#                     ))

#             valid    = [a for a in sev_accs if not np.isnan(a)]
#             mean_acc = float(np.mean(valid)) if valid else float("nan")
#             mca_vals.append(mean_acc)

#             sev_str = " ".join(f"{a:5.1f}" for a in sev_accs)
#             out(f"  {corr:<22} | {sev_str} | {mean_acc:6.2f}   ({time.time()-t0:.1f}s)")

#             per_corruption[corr] = {
#                 **{f"s{i+1}": round(sev_accs[i], 4) for i in range(5)},
#                 "mean": round(mean_acc, 4),
#             }

#         valid_mca = [a for a in mca_vals if not np.isnan(a)]
#         mca       = float(np.mean(valid_mca))
#         mce       = 100.0 - mca

#     # ── summary ──────────────────────────────────────────────
#     out(f"\n  {'='*63}")
#     if clean_acc is not None:
#         out(f"  Clean accuracy : {clean_acc:.2f}%")
#     if mca is not None:
#         out(f"  mCA            : {mca:.2f}%   (mean Corruption Accuracy)")
#         out(f"  mCE            : {mce:.2f}%   (mean Corruption Error = 100 - mCA)")
#     out(f"  {'='*63}")

#     results = {
#         "model":          model_name,
#         "dataset":        args.dataset,
#         "train_mode":     "augmix" if args.augmix else "clean",
#         "checkpoint":     ckpt_path,
#         "timestamp":      datetime.now().isoformat(),
#         "clean_acc":      round(clean_acc, 4) if clean_acc is not None else None,
#         "mCA":            round(mca, 4)       if mca       is not None else None,
#         "mCE":            round(mce, 4)       if mce       is not None else None,
#         "per_corruption": per_corruption,
#     }
#     save_results(model_name, results, lines)
#     return results


# # ──────────────────────────────────────────────────────────────
# #  DETERMINE WHICH MODELS TO EVAL
# # ──────────────────────────────────────────────────────────────
# if args.eval_all:
#     eval_models = MODEL_NAMES
# elif args.eval_group:
#     eval_models = GROUPS[args.eval_group]
# else:
#     eval_models = None   # single model mode


# # ──────────────────────────────────────────────────────────────
# #  MAIN
# # ──────────────────────────────────────────────────────────────
# if eval_models is not None:
#     # ── multi-model mode (eval_all or eval_group) ─────────────
#     all_results = {}
#     for m in eval_models:
#         ckpt = ckpt_path_for(m)
#         if os.path.exists(ckpt):
#             all_results[m] = run(m, ckpt)
#         else:
#             print(f"  [SKIP] {ckpt} not found")

#     # ── summary table ─────────────────────────────────────────
#     W = max(len(m) for m in eval_models) + 2   # dynamic width for model names
#     print(f"\n{'='*70}")
#     print(f"  SUMMARY — {args.dataset.upper()}"
#           + (f" (augmix)" if args.augmix else " (clean)"))
#     print(f"  {'Model':<{W}} | {'Clean':>6} | {'mCA':>6} | {'mCE':>6}")
#     print(f"  {'-'*68}")
#     for m, r in all_results.items():
#         ca  = f"{r['clean_acc']:6.2f}" if r['clean_acc'] is not None else "  n/a "
#         mca = f"{r['mCA']:6.2f}"       if r['mCA']       is not None else "  n/a "
#         mce = f"{r['mCE']:6.2f}"       if r['mCE']       is not None else "  n/a "
#         print(f"  {m:<{W}} | {ca} | {mca} | {mce}")
#     print(f"{'='*70}")

#     # ── save summary ──────────────────────────────────────────
#     suffix       = f"_augmix" if args.augmix else ""
#     summary_path = os.path.join(args.results_dir,
#                                 f"summary_{args.dataset}{suffix}.json")
#     with open(summary_path, "w") as f:
#         json.dump(all_results, f, indent=2)
#     print(f"\n  [Saved] {summary_path}")

# else:
#     # ── single model mode ─────────────────────────────────────
#     ckpt = args.ckpt or ckpt_path_for(args.model)
#     if not os.path.exists(ckpt):
#         raise FileNotFoundError(
#             f"No checkpoint at {ckpt}\n"
#             "Provide --ckpt <path>, use --eval_all, or --eval_group"
#         )
#     run(args.model, ckpt)


# """
# eval.py — Evaluation on clean data and corruption benchmarks.

# Two evaluations, clearly separated:
#   1. CLEAN accuracy  — standard test set (CIFAR test / ImageNet val)
#   2. CORRUPT accuracy — CIFAR-C or ImageNet-C, 15 corruptions × 5 severities

# Results printed in the same table format as the original script, and
# saved to --results_dir as JSON + txt so evaluation never needs re-running.

# Evaluation groups (--eval_group)
# ---------------------------------
# Mirror train.py groups so you can eval a round's results in one command:

#     cifar_all         : all CIFAR models
#     imagenet_core     : shape_res50 + baseline_res50
#     imagenet_deep     : shape_res101 + baseline_res101
#     imagenet_convnext : shape_convnext_tiny + baseline_convnext_tiny
#     imagenet_effnet   : shape_effnet_b4 + baseline_efficientnet_b4
#     imagenet_all      : all imagenet rounds combined

# CIFAR-C layout expected:
#     <data_dir>/CIFAR-10-C/<corruption>.npy
#     <data_dir>/CIFAR-10-C/labels.npy

# ImageNet-C layout expected:
#     <data_dir>/ImageNet-C/<corruption>/<severity 1-5>/<class>/<img>.JPEG

# Usage
# -----
#     # Single model
#     python eval.py --model shape_res18 --dataset cifar10
#     python eval.py --model shape_res50 --dataset imagenet --data_dir /pvc/imagenet

#     # Eval a predefined group
#     python eval.py --eval_group imagenet_core --dataset imagenet --data_dir /pvc/imagenet
#     python eval.py --eval_group cifar_all     --dataset cifar10

#     # All available checkpoints
#     python eval.py --eval_all --dataset cifar10
#     python eval.py --eval_all --dataset imagenet --data_dir /pvc/imagenet

#     # AugMix ablation checkpoint
#     python eval.py --model shape_res50 --dataset imagenet --augmix --data_dir /pvc/imagenet

#     # Clean only / corrupt only
#     python eval.py --eval_group imagenet_core --dataset imagenet --clean_only
#     python eval.py --eval_group imagenet_core --dataset imagenet --corrupt_only
# """

# import argparse, json, os, time
# from datetime import datetime

# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as T
# from torch.utils.data import DataLoader, Dataset
# from models import build_model, MODEL_NAMES

# # ──────────────────────────────────────────────────────────────
# #  EVAL GROUPS  (mirrors train.py GROUPS)
# # ──────────────────────────────────────────────────────────────
# GROUPS = {
#     "cifar_all":          ["baseline_res18", "baseline_res50",
#                            "shape_custom", "shape_res18", "shape_res50"],
#     "imagenet_core":      ["baseline_res50",           "shape_res50"],
#     "imagenet_deep":      ["baseline_res101",           "shape_res101"],
#     "imagenet_convnext":  ["baseline_convnext_tiny",    "shape_convnext_tiny"],
#     "imagenet_effnet":    ["baseline_efficientnet_b4",  "shape_effnet_b4"],
#     "imagenet_all":       ["baseline_res50",            "shape_res50",
#                            "baseline_res101",           "shape_res101",
#                            "baseline_convnext_tiny",    "shape_convnext_tiny",
#                            "baseline_efficientnet_b4",  "shape_effnet_b4"],
# }

# # ──────────────────────────────────────────────────────────────
# #  ARGS
# # ──────────────────────────────────────────────────────────────
# p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# p.add_argument("--model",       default="shape_res18", choices=MODEL_NAMES)
# p.add_argument("--dataset",     default="cifar10",
#                choices=["cifar10", "cifar100", "imagenet"])
# p.add_argument("--data_dir",    default="./data")
# p.add_argument("--ckpt_dir",    default="./checkpoints")
# p.add_argument("--results_dir", default="./results",
#                help="Where to save JSON + txt reports")
# p.add_argument("--ckpt",        default=None,
#                help="Explicit checkpoint path (overrides --ckpt_dir lookup)")
# p.add_argument("--batch",       type=int, default=256)
# p.add_argument("--workers",     type=int, default=4)

# # ── what to eval ─────────────────────────────────────────────
# eval_group = p.add_mutually_exclusive_group()
# eval_group.add_argument("--eval_all",   action="store_true",
#                         help="Evaluate all MODEL_NAMES checkpoints in --ckpt_dir")
# eval_group.add_argument("--eval_group", choices=list(GROUPS.keys()),
#                         help="Evaluate a predefined experiment group")

# # ── eval scope ───────────────────────────────────────────────
# p.add_argument("--clean_only",   action="store_true",
#                help="Run clean evaluation only (skip corruption eval)")
# p.add_argument("--corrupt_only", action="store_true",
#                help="Run corruption evaluation only (skip clean eval)")

# # ── augmix checkpoint ─────────────────────────────────────────
# p.add_argument("--augmix", action="store_true",
#                help="Load AugMix checkpoint (<model>_<dataset>_augmix.pt) "
#                     "instead of clean checkpoint.")

# args = p.parse_args()

# IS_IMAGENET = args.dataset == "imagenet"
# DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_CLASSES = 1000 if IS_IMAGENET else (10 if args.dataset == "cifar10" else 100)
# CKPT_SUFFIX = "_augmix" if args.augmix else ""
# NUM_GPUS    = torch.cuda.device_count()

# os.makedirs(args.results_dir, exist_ok=True)

# # ──────────────────────────────────────────────────────────────
# #  NORMALIZATION STATS
# # ──────────────────────────────────────────────────────────────
# MEAN = {"cifar10":  (0.4914, 0.4822, 0.4465),
#         "cifar100": (0.5071, 0.4867, 0.4408),
#         "imagenet": (0.485,  0.456,  0.406)}
# STD  = {"cifar10":  (0.247,  0.243,  0.261),
#         "cifar100": (0.2675, 0.2565, 0.2761),
#         "imagenet": (0.229,  0.224,  0.225)}
# mean, std = MEAN[args.dataset], STD[args.dataset]

# CORRUPTIONS = [
#     "gaussian_noise", "shot_noise",        "impulse_noise",
#     "defocus_blur",   "glass_blur",        "motion_blur",    "zoom_blur",
#     "snow",           "frost",             "fog",            "brightness",
#     "contrast",       "elastic_transform", "pixelate",       "jpeg_compression",
# ]

# # ──────────────────────────────────────────────────────────────
# #  CLEAN DATA LOADER
# # ──────────────────────────────────────────────────────────────
# if IS_IMAGENET:
#     clean_tf = T.Compose([T.Resize(256), T.CenterCrop(224),
#                           T.ToTensor(), T.Normalize(mean, std)])
#     clean_ds = torchvision.datasets.ImageFolder(
#         os.path.join(args.data_dir, "val"), transform=clean_tf)
# else:
#     clean_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
#     DS       = (torchvision.datasets.CIFAR10 if args.dataset == "cifar10"
#                 else torchvision.datasets.CIFAR100)
#     clean_ds = DS(args.data_dir, train=False, download=True, transform=clean_tf)

# clean_loader = DataLoader(clean_ds, args.batch, shuffle=False,
#                           num_workers=args.workers, pin_memory=True)


# # ──────────────────────────────────────────────────────────────
# #  IMAGENET-C DATASET WRAPPER
# # ──────────────────────────────────────────────────────────────
# class ImageNetCDataset(Dataset):
#     """Single (corruption, severity) folder as a standard Dataset."""
#     def __init__(self, root: str, corruption: str, severity: int, transform):
#         path    = os.path.join(root, "ImageNet-C", corruption, str(severity))
#         self.ds = torchvision.datasets.ImageFolder(path, transform=transform)

#     def __len__(self):        return len(self.ds)
#     def __getitem__(self, i): return self.ds[i]


# # ──────────────────────────────────────────────────────────────
# #  EVAL HELPERS
# # ──────────────────────────────────────────────────────────────
# def prep_cifar_npy(x: np.ndarray) -> torch.Tensor:
#     """uint8 numpy (N,H,W,C) → normalised float tensor (N,C,H,W)."""
#     t = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 255.0
#     m = torch.tensor(mean).view(1, 3, 1, 1)
#     s = torch.tensor(std).view(1, 3, 1, 1)
#     return (t - m) / s


# @torch.no_grad()
# def eval_loader(model, loader) -> float:
#     correct = total = 0
#     for x, y in loader:
#         x, y    = x.to(DEVICE), y.to(DEVICE)
#         correct += (model(x).argmax(1) == y).sum().item()
#         total   += y.size(0)
#     return 100.0 * correct / total


# @torch.no_grad()
# def eval_cifar_npy(model, x_np: np.ndarray, y_np: np.ndarray) -> float:
#     correct = total = 0
#     for i in range(0, len(x_np), args.batch):
#         xb = prep_cifar_npy(x_np[i:i+args.batch]).to(DEVICE)
#         yb = torch.from_numpy(y_np[i:i+args.batch]).to(DEVICE)
#         correct += (model(xb).argmax(1) == yb).sum().item()
#         total   += yb.size(0)
#     return 100.0 * correct / total


# # ──────────────────────────────────────────────────────────────
# #  SAVE RESULTS
# # ──────────────────────────────────────────────────────────────
# def save_results(model_name: str, results: dict, lines: list):
#     """
#     Saves two files to --results_dir:
#       <model>_<dataset>[_augmix]_results.json  — full structured data
#       <model>_<dataset>[_augmix]_report.txt    — identical to stdout
#     """
#     stem      = f"{model_name}_{args.dataset}{CKPT_SUFFIX}"
#     json_path = os.path.join(args.results_dir, f"{stem}_results.json")
#     txt_path  = os.path.join(args.results_dir, f"{stem}_report.txt")

#     with open(json_path, "w") as f:
#         json.dump(results, f, indent=2)
#     with open(txt_path, "w") as f:
#         f.write("\n".join(lines) + "\n")

#     print(f"  [Saved] {json_path}")
#     print(f"  [Saved] {txt_path}")


# # ──────────────────────────────────────────────────────────────
# #  CHECKPOINT LOOKUP
# # ──────────────────────────────────────────────────────────────
# def ckpt_path_for(model_name: str) -> str:
#     """Return the checkpoint path for a model, respecting --augmix flag."""
#     return os.path.join(args.ckpt_dir,
#                         f"{model_name}_{args.dataset}{CKPT_SUFFIX}.pt")


# # ──────────────────────────────────────────────────────────────
# #  MAIN EVAL RUNNER
# # ──────────────────────────────────────────────────────────────
# def run(model_name: str, ckpt_path: str) -> dict:
#     lines = []

#     def out(s: str = ""):
#         print(s)
#         lines.append(s)

#     out(f"\n{'='*65}")
#     out(f"  Model      : {model_name}")
#     out(f"  Dataset    : {args.dataset}")
#     out(f"  Train mode : {'augmix' if args.augmix else 'clean'}")
#     out(f"  Ckpt       : {ckpt_path}")
#     out(f"  GPUs       : {NUM_GPUS}")
#     out(f"  Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     out(f"{'='*65}")

#     # ── load model ──
#     # Checkpoints are always saved without DataParallel wrapper (model.module)
#     # so load_state_dict works cleanly regardless of how many GPUs were used.
#     model = build_model(model_name, NUM_CLASSES, dataset=args.dataset)
#     state = torch.load(ckpt_path, map_location="cpu")
#     model.load_state_dict(state["model"] if "model" in state else state)
#     model = model.to(DEVICE)

#     # Wrap with DataParallel if multiple GPUs available
#     if NUM_GPUS > 1:
#         print(f"  [DataParallel] Using {NUM_GPUS} GPUs for eval")
#         model = nn.DataParallel(model)

#     model = model.eval()

#     clean_acc      = None
#     per_corruption = {}
#     mca = mce      = None

#     # ──────────────────────────────────────────────────────────
#     #  CLEAN EVALUATION
#     # ──────────────────────────────────────────────────────────
#     if not args.corrupt_only:
#         out(f"\n  {'─'*45}")
#         out(f"  CLEAN EVALUATION")
#         out(f"  {'─'*45}")
#         t0        = time.time()
#         clean_acc = eval_loader(model, clean_loader)
#         out(f"  Clean accuracy : {clean_acc:.2f}%   ({time.time()-t0:.1f}s)")

#     # ──────────────────────────────────────────────────────────
#     #  CORRUPTION EVALUATION
#     # ──────────────────────────────────────────────────────────
#     if not args.clean_only:
#         benchmark = ("CIFAR-10-C"  if args.dataset == "cifar10"  else
#                      "CIFAR-100-C" if args.dataset == "cifar100" else
#                      "ImageNet-C")
#         out(f"\n  {'─'*45}")
#         out(f"  CORRUPTION EVALUATION  ({benchmark})")
#         out(f"  {'─'*45}")
#         out(f"  {'Corruption':<22} | {'s1':>5} {'s2':>5} {'s3':>5} {'s4':>5} {'s5':>5} | {'mean':>6}")
#         out(f"  {'-'*60}")

#         mca_vals = []
#         for corr in CORRUPTIONS:
#             sev_accs = []
#             t0 = time.time()

#             if IS_IMAGENET:
#                 for sev in range(1, 6):
#                     try:
#                         ds  = ImageNetCDataset(args.data_dir, corr, sev, clean_tf)
#                         ldr = DataLoader(ds, args.batch, shuffle=False,
#                                          num_workers=args.workers, pin_memory=True)
#                         sev_accs.append(eval_loader(model, ldr))
#                     except (FileNotFoundError, RuntimeError):
#                         out(f"  [WARN] Missing: ImageNet-C/{corr}/{sev}")
#                         sev_accs.append(float("nan"))
#             else:
#                 croot = (f"{args.data_dir}/CIFAR-10-C"  if args.dataset == "cifar10"
#                          else f"{args.data_dir}/CIFAR-100-C")
#                 x_all = np.load(f"{croot}/{corr}.npy")
#                 y_all = np.load(f"{croot}/labels.npy")
#                 for sev in range(5):
#                     sev_accs.append(eval_cifar_npy(
#                         model,
#                         x_all[sev*10000:(sev+1)*10000],
#                         y_all[sev*10000:(sev+1)*10000],
#                     ))

#             valid    = [a for a in sev_accs if not np.isnan(a)]
#             mean_acc = float(np.mean(valid)) if valid else float("nan")
#             mca_vals.append(mean_acc)

#             sev_str = " ".join(f"{a:5.1f}" for a in sev_accs)
#             out(f"  {corr:<22} | {sev_str} | {mean_acc:6.2f}   ({time.time()-t0:.1f}s)")

#             per_corruption[corr] = {
#                 **{f"s{i+1}": round(sev_accs[i], 4) for i in range(5)},
#                 "mean": round(mean_acc, 4),
#             }

#         valid_mca = [a for a in mca_vals if not np.isnan(a)]
#         mca       = float(np.mean(valid_mca))
#         mce       = 100.0 - mca

#     # ── summary ──────────────────────────────────────────────
#     out(f"\n  {'='*63}")
#     if clean_acc is not None:
#         out(f"  Clean accuracy : {clean_acc:.2f}%")
#     if mca is not None:
#         out(f"  mCA            : {mca:.2f}%   (mean Corruption Accuracy)")
#         out(f"  mCE            : {mce:.2f}%   (mean Corruption Error = 100 - mCA)")
#     out(f"  {'='*63}")

#     results = {
#         "model":          model_name,
#         "dataset":        args.dataset,
#         "train_mode":     "augmix" if args.augmix else "clean",
#         "checkpoint":     ckpt_path,
#         "timestamp":      datetime.now().isoformat(),
#         "clean_acc":      round(clean_acc, 4) if clean_acc is not None else None,
#         "mCA":            round(mca, 4)       if mca       is not None else None,
#         "mCE":            round(mce, 4)       if mce       is not None else None,
#         "per_corruption": per_corruption,
#     }
#     save_results(model_name, results, lines)
#     return results


# # ──────────────────────────────────────────────────────────────
# #  DETERMINE WHICH MODELS TO EVAL
# # ──────────────────────────────────────────────────────────────
# if args.eval_all:
#     eval_models = MODEL_NAMES
# elif args.eval_group:
#     eval_models = GROUPS[args.eval_group]
# else:
#     eval_models = None   # single model mode


# # ──────────────────────────────────────────────────────────────
# #  MAIN
# # ──────────────────────────────────────────────────────────────
# if eval_models is not None:
#     # ── multi-model mode (eval_all or eval_group) ─────────────
#     all_results = {}
#     for m in eval_models:
#         ckpt = ckpt_path_for(m)
#         if os.path.exists(ckpt):
#             all_results[m] = run(m, ckpt)
#         else:
#             print(f"  [SKIP] {ckpt} not found")

#     # ── summary table ─────────────────────────────────────────
#     W = max(len(m) for m in eval_models) + 2
#     print(f"\n{'='*70}")
#     print(f"  SUMMARY — {args.dataset.upper()}"
#           + (f" (augmix)" if args.augmix else " (clean)"))
#     print(f"  {'Model':<{W}} | {'Clean':>6} | {'mCA':>6} | {'mCE':>6}")
#     print(f"  {'-'*68}")
#     for m, r in all_results.items():
#         ca  = f"{r['clean_acc']:6.2f}" if r['clean_acc'] is not None else "  n/a "
#         mca = f"{r['mCA']:6.2f}"       if r['mCA']       is not None else "  n/a "
#         mce = f"{r['mCE']:6.2f}"       if r['mCE']       is not None else "  n/a "
#         print(f"  {m:<{W}} | {ca} | {mca} | {mce}")
#     print(f"{'='*70}")

#     suffix       = f"_augmix" if args.augmix else ""
#     summary_path = os.path.join(args.results_dir,
#                                 f"summary_{args.dataset}{suffix}.json")
#     with open(summary_path, "w") as f:
#         json.dump(all_results, f, indent=2)
#     print(f"\n  [Saved] {summary_path}")

# else:
#     # ── single model mode ─────────────────────────────────────
#     ckpt = args.ckpt or ckpt_path_for(args.model)
#     if not os.path.exists(ckpt):
#         raise FileNotFoundError(
#             f"No checkpoint at {ckpt}\n"
#             "Provide --ckpt <path>, use --eval_all, or --eval_group"
#         )
#     run(args.model, ckpt)


"""
eval.py — Evaluation on clean data and corruption benchmarks.

Two evaluations, clearly separated:
  1. CLEAN accuracy  — standard test set (CIFAR test / ImageNet val)
  2. CORRUPT accuracy — CIFAR-C or ImageNet-C, 15 corruptions × 5 severities

Results printed in the same table format as the original script, and
saved to --results_dir as JSON + txt so evaluation never needs re-running.

Evaluation groups (--eval_group)
---------------------------------
Mirror train.py groups so you can eval a round's results in one command:

    cifar_all         : all CIFAR models
    imagenet_core     : shape_res50 + baseline_res50
    imagenet_deep     : shape_res101 + baseline_res101
    imagenet_convnext : shape_convnext_tiny + baseline_convnext_tiny
    imagenet_effnet   : shape_effnet_b4 + baseline_efficientnet_b4
    imagenet_all      : all imagenet rounds combined

CIFAR-C layout expected:
    <data_dir>/CIFAR-10-C/<corruption>.npy
    <data_dir>/CIFAR-10-C/labels.npy

ImageNet-C layout expected:
    <data_dir>/ImageNet-C/<corruption>/<severity 1-5>/<class>/<img>.JPEG

Usage
-----
    # Single model
    python eval.py --model shape_res18 --dataset cifar10
    python eval.py --model shape_res50 --dataset imagenet --data_dir /pvc/imagenet

    # Eval a predefined group
    python eval.py --eval_group imagenet_core --dataset imagenet --data_dir /pvc/imagenet
    python eval.py --eval_group cifar_all     --dataset cifar10

    # All available checkpoints
    python eval.py --eval_all --dataset cifar10
    python eval.py --eval_all --dataset imagenet --data_dir /pvc/imagenet

    # AugMix ablation checkpoint
    python eval.py --model shape_res50 --dataset imagenet --augmix --data_dir /pvc/imagenet

    # Clean only / corrupt only
    python eval.py --eval_group imagenet_core --dataset imagenet --clean_only
    python eval.py --eval_group imagenet_core --dataset imagenet --corrupt_only
"""

import argparse, json, os, time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from models import build_model, MODEL_NAMES

# ──────────────────────────────────────────────────────────────
#  EVAL GROUPS  (mirrors train.py GROUPS)
# ──────────────────────────────────────────────────────────────
GROUPS = {
    "cifar_all":          ["baseline_res18", "baseline_res50",
                           "shape_custom", "shape_res18", "shape_res50"],
    "imagenet_core":      ["baseline_res50",           "shape_res50"],
    "imagenet_deep":      ["baseline_res101",           "shape_res101"],
    "imagenet_convnext":  ["baseline_convnext_tiny",    "shape_convnext_tiny"],
    "imagenet_effnet":    ["baseline_efficientnet_b4",  "shape_effnet_b4"],
    "imagenet_all":       ["baseline_res50",            "shape_res50",
                           "baseline_res101",           "shape_res101",
                           "baseline_convnext_tiny",    "shape_convnext_tiny",
                           "baseline_efficientnet_b4",  "shape_effnet_b4"],
}

# ──────────────────────────────────────────────────────────────
#  ARGS
# ──────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--model",       default="shape_res18", choices=MODEL_NAMES)
p.add_argument("--dataset",     default="cifar10",
               choices=["cifar10", "cifar100", "imagenet"])
p.add_argument("--data_dir",    default="./data")
p.add_argument("--ckpt_dir",    default="./checkpoints")
p.add_argument("--results_dir", default="./results",
               help="Where to save JSON + txt reports")
p.add_argument("--ckpt",        default=None,
               help="Explicit checkpoint path (overrides --ckpt_dir lookup)")
p.add_argument("--batch",       type=int, default=256)
p.add_argument("--workers",     type=int, default=4)

# ── what to eval ─────────────────────────────────────────────
eval_group = p.add_mutually_exclusive_group()
eval_group.add_argument("--eval_all",   action="store_true",
                        help="Evaluate all MODEL_NAMES checkpoints in --ckpt_dir")
eval_group.add_argument("--eval_group", choices=list(GROUPS.keys()),
                        help="Evaluate a predefined experiment group")

# ── eval scope ───────────────────────────────────────────────
p.add_argument("--clean_only",   action="store_true",
               help="Run clean evaluation only (skip corruption eval)")
p.add_argument("--corrupt_only", action="store_true",
               help="Run corruption evaluation only (skip clean eval)")

# ── augmix checkpoint ─────────────────────────────────────────
p.add_argument("--augmix", action="store_true",
               help="Load AugMix checkpoint (<model>_<dataset>_augmix.pt) "
                    "instead of clean checkpoint.")

args = p.parse_args()

IS_IMAGENET = args.dataset == "imagenet"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 1000 if IS_IMAGENET else (10 if args.dataset == "cifar10" else 100)
CKPT_SUFFIX = "_augmix" if args.augmix else ""
NUM_GPUS    = torch.cuda.device_count()

os.makedirs(args.results_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────
#  NORMALIZATION STATS
# ──────────────────────────────────────────────────────────────
MEAN = {"cifar10":  (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "imagenet": (0.485,  0.456,  0.406)}
STD  = {"cifar10":  (0.247,  0.243,  0.261),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "imagenet": (0.229,  0.224,  0.225)}
mean, std = MEAN[args.dataset], STD[args.dataset]

CORRUPTIONS = [
    "gaussian_noise", "shot_noise",        "impulse_noise",
    "defocus_blur",   "glass_blur",        "motion_blur",    "zoom_blur",
    "snow",           "frost",             "fog",            "brightness",
    "contrast",       "elastic_transform", "pixelate",       "jpeg_compression",
]

# ──────────────────────────────────────────────────────────────
#  CLEAN DATA LOADER
# ──────────────────────────────────────────────────────────────
if IS_IMAGENET:
    clean_tf = T.Compose([T.Resize(256), T.CenterCrop(224),
                          T.ToTensor(), T.Normalize(mean, std)])
    import pickle
    _val_ds = torchvision.datasets.ImageFolder(
        os.path.join(args.data_dir, "val"), transform=clean_tf)
    _val_cache = os.path.join(args.data_dir, "imagenet_val_cache.pkl")
    if os.path.exists(_val_cache):
        print(f"  [Cache] Loading val index from {_val_cache}...")
        with open(_val_cache, "rb") as f:
            _cached = pickle.load(f)
        _val_ds.samples = _cached["samples"]
        _val_ds.targets = _cached["targets"]
        _val_ds.imgs    = _val_ds.samples
        print(f"  [Cache] Loaded {len(_val_ds.samples)} val samples")
    else:
        print("  [Cache] Building val index (first time)...")
        with open(_val_cache, "wb") as f:
            pickle.dump({"samples": _val_ds.samples, "targets": _val_ds.targets}, f)
        print(f"  [Cache] Saved val index to {_val_cache}")
    clean_ds = _val_ds
else:
    clean_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    DS       = (torchvision.datasets.CIFAR10 if args.dataset == "cifar10"
                else torchvision.datasets.CIFAR100)
    clean_ds = DS(args.data_dir, train=False, download=True, transform=clean_tf)

clean_loader = DataLoader(clean_ds, args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)


# ──────────────────────────────────────────────────────────────
#  IMAGENET-C DATASET WRAPPER
# ──────────────────────────────────────────────────────────────
class ImageNetCDataset(Dataset):
    """Single (corruption, severity) folder as a standard Dataset."""
    def __init__(self, root: str, corruption: str, severity: int, transform):
        path    = os.path.join(root, "ImageNet-C", corruption, str(severity))
        self.ds = torchvision.datasets.ImageFolder(path, transform=transform)

    def __len__(self):        return len(self.ds)
    def __getitem__(self, i): return self.ds[i]


# ──────────────────────────────────────────────────────────────
#  EVAL HELPERS
# ──────────────────────────────────────────────────────────────
def prep_cifar_npy(x: np.ndarray) -> torch.Tensor:
    """uint8 numpy (N,H,W,C) → normalised float tensor (N,C,H,W)."""
    t = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 255.0
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (t - m) / s


@torch.no_grad()
def eval_loader(model, loader) -> float:
    correct = total = 0
    for x, y in loader:
        x, y    = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def eval_cifar_npy(model, x_np: np.ndarray, y_np: np.ndarray) -> float:
    correct = total = 0
    for i in range(0, len(x_np), args.batch):
        xb = prep_cifar_npy(x_np[i:i+args.batch]).to(DEVICE)
        yb = torch.from_numpy(y_np[i:i+args.batch]).to(DEVICE)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total   += yb.size(0)
    return 100.0 * correct / total


# ──────────────────────────────────────────────────────────────
#  SAVE RESULTS
# ──────────────────────────────────────────────────────────────
def save_results(model_name: str, results: dict, lines: list):
    """
    Saves two files to --results_dir:
      <model>_<dataset>[_augmix]_results.json  — full structured data
      <model>_<dataset>[_augmix]_report.txt    — identical to stdout
    """
    stem      = f"{model_name}_{args.dataset}{CKPT_SUFFIX}"
    json_path = os.path.join(args.results_dir, f"{stem}_results.json")
    txt_path  = os.path.join(args.results_dir, f"{stem}_report.txt")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  [Saved] {json_path}")
    print(f"  [Saved] {txt_path}")


# ──────────────────────────────────────────────────────────────
#  CHECKPOINT LOOKUP
# ──────────────────────────────────────────────────────────────
def ckpt_path_for(model_name: str) -> str:
    """Return the checkpoint path for a model, respecting --augmix flag."""
    return os.path.join(args.ckpt_dir,
                        f"{model_name}_{args.dataset}{CKPT_SUFFIX}.pt")


# ──────────────────────────────────────────────────────────────
#  MAIN EVAL RUNNER
# ──────────────────────────────────────────────────────────────
def run(model_name: str, ckpt_path: str) -> dict:
    lines = []

    def out(s: str = ""):
        print(s)
        lines.append(s)

    out(f"\n{'='*65}")
    out(f"  Model      : {model_name}")
    out(f"  Dataset    : {args.dataset}")
    out(f"  Train mode : {'augmix' if args.augmix else 'clean'}")
    out(f"  Ckpt       : {ckpt_path}")
    out(f"  GPUs       : {NUM_GPUS}")
    out(f"  Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"{'='*65}")

    # ── load model ──
    # Checkpoints are always saved without DataParallel wrapper (model.module)
    # so load_state_dict works cleanly regardless of how many GPUs were used.
    model = build_model(model_name, NUM_CLASSES, dataset=args.dataset)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model = model.to(DEVICE)

    # Wrap with DataParallel if multiple GPUs available
    if NUM_GPUS > 1:
        print(f"  [DataParallel] Using {NUM_GPUS} GPUs for eval")
        model = nn.DataParallel(model)

    model = model.eval()

    clean_acc      = None
    per_corruption = {}
    mca = mce      = None

    # ──────────────────────────────────────────────────────────
    #  CLEAN EVALUATION
    # ──────────────────────────────────────────────────────────
    if not args.corrupt_only:
        out(f"\n  {'─'*45}")
        out(f"  CLEAN EVALUATION")
        out(f"  {'─'*45}")
        t0        = time.time()
        clean_acc = eval_loader(model, clean_loader)
        out(f"  Clean accuracy : {clean_acc:.2f}%   ({time.time()-t0:.1f}s)")

    # ──────────────────────────────────────────────────────────
    #  CORRUPTION EVALUATION
    # ──────────────────────────────────────────────────────────
    if not args.clean_only:
        benchmark = ("CIFAR-10-C"  if args.dataset == "cifar10"  else
                     "CIFAR-100-C" if args.dataset == "cifar100" else
                     "ImageNet-C")
        out(f"\n  {'─'*45}")
        out(f"  CORRUPTION EVALUATION  ({benchmark})")
        out(f"  {'─'*45}")
        out(f"  {'Corruption':<22} | {'s1':>5} {'s2':>5} {'s3':>5} {'s4':>5} {'s5':>5} | {'mean':>6}")
        out(f"  {'-'*60}")

        mca_vals = []
        for corr in CORRUPTIONS:
            sev_accs = []
            t0 = time.time()

            if IS_IMAGENET:
                for sev in range(1, 6):
                    try:
                        ds  = ImageNetCDataset(args.data_dir, corr, sev, clean_tf)
                        ldr = DataLoader(ds, args.batch, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
                        sev_accs.append(eval_loader(model, ldr))
                    except (FileNotFoundError, RuntimeError):
                        out(f"  [WARN] Missing: ImageNet-C/{corr}/{sev}")
                        sev_accs.append(float("nan"))
            else:
                croot = (f"{args.data_dir}/CIFAR-10-C"  if args.dataset == "cifar10"
                         else f"{args.data_dir}/CIFAR-100-C")
                x_all = np.load(f"{croot}/{corr}.npy")
                y_all = np.load(f"{croot}/labels.npy")
                for sev in range(5):
                    sev_accs.append(eval_cifar_npy(
                        model,
                        x_all[sev*10000:(sev+1)*10000],
                        y_all[sev*10000:(sev+1)*10000],
                    ))

            valid    = [a for a in sev_accs if not np.isnan(a)]
            mean_acc = float(np.mean(valid)) if valid else float("nan")
            mca_vals.append(mean_acc)

            sev_str = " ".join(f"{a:5.1f}" for a in sev_accs)
            out(f"  {corr:<22} | {sev_str} | {mean_acc:6.2f}   ({time.time()-t0:.1f}s)")

            per_corruption[corr] = {
                **{f"s{i+1}": round(sev_accs[i], 4) for i in range(5)},
                "mean": round(mean_acc, 4),
            }

        valid_mca = [a for a in mca_vals if not np.isnan(a)]
        mca       = float(np.mean(valid_mca))
        mce       = 100.0 - mca

    # ── summary ──────────────────────────────────────────────
    out(f"\n  {'='*63}")
    if clean_acc is not None:
        out(f"  Clean accuracy : {clean_acc:.2f}%")
    if mca is not None:
        out(f"  mCA            : {mca:.2f}%   (mean Corruption Accuracy)")
        out(f"  mCE            : {mce:.2f}%   (mean Corruption Error = 100 - mCA)")
    out(f"  {'='*63}")

    results = {
        "model":          model_name,
        "dataset":        args.dataset,
        "train_mode":     "augmix" if args.augmix else "clean",
        "checkpoint":     ckpt_path,
        "timestamp":      datetime.now().isoformat(),
        "clean_acc":      round(clean_acc, 4) if clean_acc is not None else None,
        "mCA":            round(mca, 4)       if mca       is not None else None,
        "mCE":            round(mce, 4)       if mce       is not None else None,
        "per_corruption": per_corruption,
    }
    save_results(model_name, results, lines)
    return results


# ──────────────────────────────────────────────────────────────
#  DETERMINE WHICH MODELS TO EVAL
# ──────────────────────────────────────────────────────────────
if args.eval_all:
    eval_models = MODEL_NAMES
elif args.eval_group:
    eval_models = GROUPS[args.eval_group]
else:
    eval_models = None   # single model mode


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
if eval_models is not None:
    # ── multi-model mode (eval_all or eval_group) ─────────────
    all_results = {}
    for m in eval_models:
        ckpt = ckpt_path_for(m)
        if os.path.exists(ckpt):
            all_results[m] = run(m, ckpt)
        else:
            print(f"  [SKIP] {ckpt} not found")

    # ── summary table ─────────────────────────────────────────
    W = max(len(m) for m in eval_models) + 2
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {args.dataset.upper()}"
          + (f" (augmix)" if args.augmix else " (clean)"))
    print(f"  {'Model':<{W}} | {'Clean':>6} | {'mCA':>6} | {'mCE':>6}")
    print(f"  {'-'*68}")
    for m, r in all_results.items():
        ca  = f"{r['clean_acc']:6.2f}" if r['clean_acc'] is not None else "  n/a "
        mca = f"{r['mCA']:6.2f}"       if r['mCA']       is not None else "  n/a "
        mce = f"{r['mCE']:6.2f}"       if r['mCE']       is not None else "  n/a "
        print(f"  {m:<{W}} | {ca} | {mca} | {mce}")
    print(f"{'='*70}")

    suffix       = f"_augmix" if args.augmix else ""
    summary_path = os.path.join(args.results_dir,
                                f"summary_{args.dataset}{suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  [Saved] {summary_path}")

else:
    # ── single model mode ─────────────────────────────────────
    ckpt = args.ckpt or ckpt_path_for(args.model)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}\n"
            "Provide --ckpt <path>, use --eval_all, or --eval_group"
        )
    run(args.model, ckpt)