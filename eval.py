"""
eval.py — Evaluation on clean data and corruption benchmarks.

Two evaluations:
  1. CLEAN accuracy  — standard test set (CIFAR test / ImageNet val)
  2. CORRUPT accuracy — CIFAR-C or ImageNet-C, 15 corruptions × 5 severities

Results saved as JSON + txt to --results_dir.

CIFAR-C layout:
    <data_dir>/CIFAR-10-C/<corruption>.npy
    <data_dir>/CIFAR-10-C/labels.npy

ImageNet-C layout:
    <data_dir>/ImageNet-C/<corruption>/<severity>/<class>/<img>.JPEG
    <data_dir>/ImageNet-100-C/<corruption>/<severity>/<class>/<img>.JPEG

Usage
-----
    python eval.py --model shape_res18 --dataset cifar10
    python eval.py --model shape_res50 --dataset imagenet100 --data_dir /pvc/imagenet100
    python eval.py --eval_group cifar_all --dataset cifar10
    python eval.py --eval_group imagenet_core --dataset imagenet100 --data_dir /pvc/imagenet100
    python eval.py --model shape_res18 --dataset cifar10 --clean_only
"""

import argparse, json, os, time
from datetime import datetime

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from models import build_model, MODEL_NAMES

# ──────────────────────────────────────────────────────────────
#  EVAL GROUPS
# ──────────────────────────────────────────────────────────────
GROUPS = {
    "cifar_all":        ["baseline_res18", "baseline_res50",
                         "shape_res18",    "shape_res50"],
    "imagenet_core":    ["baseline_res50",  "shape_res50"],
    "imagenet_deep":    ["baseline_res101", "shape_res101"],
    "imagenet_all":     ["baseline_res50",  "shape_res50",
                         "baseline_res101", "shape_res101"],
    "imagenet100_core": ["baseline_res50",  "shape_res50"],
    "imagenet100_all":  ["baseline_res50",  "shape_res50",
                         "baseline_res101", "shape_res101"],
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
p.add_argument("--results_dir", default="./results")
p.add_argument("--ckpt",        default=None,
               help="Explicit checkpoint path (overrides --ckpt_dir lookup)")
p.add_argument("--batch",       type=int, default=256)
p.add_argument("--workers",     type=int, default=4)

eval_group = p.add_mutually_exclusive_group()
eval_group.add_argument("--eval_all",   action="store_true")
eval_group.add_argument("--eval_group", choices=list(GROUPS.keys()))

p.add_argument("--clean_only",   action="store_true")
p.add_argument("--corrupt_only", action="store_true")

args = p.parse_args()

IS_IMAGENET    = args.dataset in ("imagenet", "imagenet100")
IS_IMAGENET100 = args.dataset == "imagenet100"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES    = (100  if IS_IMAGENET100 else
                  1000 if IS_IMAGENET    else
                  10   if args.dataset == "cifar10" else 100)
NUM_GPUS       = torch.cuda.device_count()
MODEL_DATASET  = "imagenet" if IS_IMAGENET else args.dataset
IMAGENET_C_DIR = "ImageNet-100-C" if IS_IMAGENET100 else "ImageNet-C"

os.makedirs(args.results_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────
#  NORMALIZATION STATS
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
    _cache_prefix = "imagenet100" if IS_IMAGENET100 else "imagenet"
    _val_cache = os.path.join(args.data_dir, f"{_cache_prefix}_val_cache.pkl")
    if os.path.exists(_val_cache):
        with open(_val_cache, "rb") as f:
            _cached = pickle.load(f)
        _val_ds.samples = _cached["samples"]
        _val_ds.targets = _cached["targets"]
        _val_ds.imgs    = _val_ds.samples
    else:
        with open(_val_cache, "wb") as f:
            pickle.dump({"samples": _val_ds.samples, "targets": _val_ds.targets}, f)
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
    def __init__(self, root, corruption, severity, transform, c_dir="ImageNet-C"):
        path    = os.path.join(root, c_dir, corruption, str(severity))
        self.ds = torchvision.datasets.ImageFolder(path, transform=transform)

    def __len__(self):        return len(self.ds)
    def __getitem__(self, i): return self.ds[i]


# ──────────────────────────────────────────────────────────────
#  EVAL HELPERS
# ──────────────────────────────────────────────────────────────
def prep_cifar_npy(x: np.ndarray) -> torch.Tensor:
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
def eval_cifar_npy(model, x_np, y_np) -> float:
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
def save_results(model_name, results, lines):
    stem      = f"{model_name}_{args.dataset}"
    json_path = os.path.join(args.results_dir, f"{stem}_results.json")
    txt_path  = os.path.join(args.results_dir, f"{stem}_report.txt")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [Saved] {json_path}")
    print(f"  [Saved] {txt_path}")


def ckpt_path_for(model_name):
    return os.path.join(args.ckpt_dir, f"{model_name}_{args.dataset}.pt")


# ──────────────────────────────────────────────────────────────
#  MAIN EVAL RUNNER
# ──────────────────────────────────────────────────────────────
def run(model_name, ckpt_path) -> dict:
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    out(f"\n{'='*65}")
    out(f"  Model      : {model_name}")
    out(f"  Dataset    : {args.dataset}")
    out(f"  Ckpt       : {ckpt_path}")
    out(f"  GPUs       : {NUM_GPUS}")
    out(f"  Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"{'='*65}")

    model = build_model(model_name, NUM_CLASSES, dataset=MODEL_DATASET)
    state = torch.load(ckpt_path, map_location="cpu")
    sd    = state["model"] if "model" in state else state
    # Strip training-only heads (e.g. aux_head) that are absent at eval time
    sd    = {k: v for k, v in sd.items() if not k.startswith("aux_head.")}
    model.load_state_dict(sd)
    model = model.to(DEVICE).eval()

    clean_acc      = None
    per_corruption = {}
    mca = mce      = None

    # ── Clean eval ───────────────────────────────────────────
    if not args.corrupt_only:
        out(f"\n  {'─'*45}")
        out(f"  CLEAN EVALUATION")
        out(f"  {'─'*45}")
        t0        = time.time()
        clean_acc = eval_loader(model, clean_loader)
        out(f"  Clean accuracy : {clean_acc:.2f}%   ({time.time()-t0:.1f}s)")

    # ── Corruption eval ──────────────────────────────────────
    if not args.clean_only:
        benchmark = ("CIFAR-10-C"     if args.dataset == "cifar10"  else
                     "CIFAR-100-C"    if args.dataset == "cifar100" else
                     IMAGENET_C_DIR)
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
                        ds  = ImageNetCDataset(args.data_dir, corr, sev,
                                               clean_tf, c_dir=IMAGENET_C_DIR)
                        ldr = DataLoader(ds, args.batch, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
                        sev_accs.append(eval_loader(model, ldr))
                    except (FileNotFoundError, RuntimeError):
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
            per_corruption[corr] = {
                **{f"s{i+1}": round(sev_accs[i], 4) for i in range(5)},
                "mean": round(mean_acc, 4),
            }
            sev_str = " ".join(f"{a:5.1f}" for a in sev_accs)
            out(f"  {corr:<22} | {sev_str} | {mean_acc:6.2f}   ({time.time()-t0:.1f}s)")

        valid_mca = [a for a in mca_vals if not np.isnan(a)]
        mca       = float(np.mean(valid_mca))
        mce       = 100.0 - mca

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
#  MAIN
# ──────────────────────────────────────────────────────────────
if args.eval_all:
    eval_models = MODEL_NAMES
elif args.eval_group:
    eval_models = GROUPS[args.eval_group]
else:
    eval_models = None

if eval_models is not None:
    all_results = {}
    for m in eval_models:
        ckpt = ckpt_path_for(m)
        if os.path.exists(ckpt):
            all_results[m] = run(m, ckpt)
            torch.cuda.empty_cache()
            import gc; gc.collect()
        else:
            print(f"  [SKIP] {ckpt} not found")

    W = max(len(m) for m in eval_models) + 2
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {args.dataset.upper()}")
    print(f"  {'Model':<{W}} | {'Clean':>6} | {'mCA':>6} | {'mCE':>6}")
    print(f"  {'-'*68}")
    for m, r in all_results.items():
        ca  = f"{r['clean_acc']:6.2f}" if r['clean_acc'] is not None else "  n/a "
        mca = f"{r['mCA']:6.2f}"       if r['mCA']       is not None else "  n/a "
        mce = f"{r['mCE']:6.2f}"       if r['mCE']       is not None else "  n/a "
        print(f"  {m:<{W}} | {ca} | {mca} | {mce}")
    print(f"{'='*70}")

    summary_path = os.path.join(args.results_dir, f"summary_{args.dataset}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  [Saved] {summary_path}")

else:
    ckpt = args.ckpt or ckpt_path_for(args.model)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}\n"
            "Provide --ckpt <path>, use --eval_all, or --eval_group")
    run(args.model, ckpt)
