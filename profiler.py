"""
profiler.py — Efficiency profiling for ShapeBiasNet models.

Reports per model:
  - Parameter count: total, trainable, fixed (orientation bank buffers)
  - GFLOPs (via fvcore; falls back to torch.profiler)
  - GPU peak memory (MB)
  - Inference latency: mean ± std, min, max (ms) — CUDA events on GPU
  - Throughput (images / second)

Also prints a paired comparison table (baseline vs shape per backbone),
showing param reduction %, FLOP reduction %, latency delta %, and memory delta %.

Outputs: JSON + TXT + CSV saved to --results_dir.

Usage
-----
    python profiler.py --dataset cifar10
    python profiler.py --dataset imagenet100
    python profiler.py --dataset cifar10 --models baseline_res18,shape_res18
    python profiler.py --dataset imagenet --device cpu

Requirements
------------
    pip install fvcore   # for accurate FLOPs; fallback available without it
"""

import argparse
import csv
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from models import build_model

# ── CLI ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--dataset",     default="cifar10",
               choices=["cifar10", "cifar100", "imagenet100", "imagenet"])
p.add_argument("--models",      default=None,
               help="Comma-separated model names (default: all for dataset)")
p.add_argument("--batch",       type=int, default=1,
               help="Batch size (1 = single-sample latency)")
p.add_argument("--warmup",      type=int, default=20)
p.add_argument("--runs",        type=int, default=100)
p.add_argument("--device",      default=None, help="cuda or cpu (auto-detected)")
p.add_argument("--results_dir", default="./results")
args = p.parse_args()

# ── Constants ─────────────────────────────────────────────────────────────────
IS_IMAGENET  = args.dataset in ("imagenet", "imagenet100")
NUM_CLASSES  = {
    "cifar10":     10,
    "cifar100":    100,
    "imagenet100": 100,
    "imagenet":    1000,
}[args.dataset]
INPUT_SIZE   = (args.batch, 3, 224, 224) if IS_IMAGENET else (args.batch, 3, 32, 32)
DEVICE       = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

# Models actually run for each dataset
DATASET_MODELS = {
    "cifar10":     ["baseline_res18", "shape_res18", "baseline_res50", "shape_res50"],
    "cifar100":    ["baseline_res18", "shape_res18", "baseline_res50", "shape_res50"],
    "imagenet100": ["baseline_res50", "shape_res50", "baseline_res101", "shape_res101"],
    "imagenet":    ["baseline_res50", "shape_res50", "baseline_res101", "shape_res101"],
}

if args.models:
    MODELS = [m.strip() for m in args.models.split(",")]
else:
    MODELS = DATASET_MODELS.get(args.dataset, [])

# Backbone pairs for comparison table
BACKBONE_PAIRS = [
    ("baseline_res18",  "shape_res18"),
    ("baseline_res50",  "shape_res50"),
    ("baseline_res101", "shape_res101"),
]

os.makedirs(args.results_dir, exist_ok=True)


# ── FLOPs ────────────────────────────────────────────────────────────────────
def count_flops(model: nn.Module, input_size: tuple) -> float:
    """Returns GFLOPs. Uses fvcore if available, falls back to torch.profiler."""
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.zeros(1, *input_size[1:]).to(next(model.parameters()).device)
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9
    except ImportError:
        return _flops_via_profiler(model, input_size)


def _flops_via_profiler(model: nn.Module, input_size: tuple) -> float:
    dummy = torch.zeros(*input_size).to(next(model.parameters()).device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as prof:
        model(dummy)
    return sum(e.flops for e in prof.key_averages() if e.flops > 0) / 1e9


# ── Params ───────────────────────────────────────────────────────────────────
def count_params(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed     = total - trainable
    return {
        "total_M":     round(total     / 1e6, 3),
        "trainable_M": round(trainable / 1e6, 3),
        "fixed_M":     round(fixed     / 1e6, 3),
    }


# ── Latency ──────────────────────────────────────────────────────────────────
def measure_latency(model: nn.Module, input_size: tuple,
                    warmup: int, runs: int, device: str) -> dict:
    dummy = torch.zeros(*input_size).to(device)
    model = model.to(device).eval()

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        timings = []
        if device == "cuda":
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            for _ in range(runs):
                start_ev.record()
                model(dummy)
                end_ev.record()
                torch.cuda.synchronize()
                timings.append(start_ev.elapsed_time(end_ev))
        else:
            for _ in range(runs):
                t0 = time.perf_counter()
                model(dummy)
                timings.append((time.perf_counter() - t0) * 1000)

    mean_ms = statistics.mean(timings)
    return {
        "mean_ms":      round(mean_ms,                    3),
        "std_ms":       round(statistics.stdev(timings),  3),
        "min_ms":       round(min(timings),               3),
        "max_ms":       round(max(timings),               3),
        "throughput":   round(input_size[0] / (mean_ms / 1000), 1),  # img/s
        "device":       device,
        "batch_size":   input_size[0],
    }


# ── Memory ───────────────────────────────────────────────────────────────────
def measure_memory(model: nn.Module, input_size: tuple) -> dict:
    if not torch.cuda.is_available():
        return {"peak_mb": 0.0, "note": "CPU only"}
    model = model.cuda().eval()
    dummy = torch.zeros(*input_size).cuda()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        model(dummy)
    torch.cuda.synchronize()
    return {"peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2)}


# ── Profile one model ─────────────────────────────────────────────────────────
def profile_model(name: str) -> dict:
    print(f"  profiling {name} ...")
    # imagenet100 uses the imagenet stem (no 'cifar' in name) — pass correctly
    ds_for_build = args.dataset
    model = build_model(name, NUM_CLASSES, dataset=ds_for_build)
    model = model.to(DEVICE).eval()

    params  = count_params(model)
    gflops  = count_flops(model, INPUT_SIZE)
    latency = measure_latency(model, INPUT_SIZE, args.warmup, args.runs, DEVICE)
    memory  = measure_memory(model, INPUT_SIZE)

    return {
        "model":   name,
        "dataset": args.dataset,
        "input":   list(INPUT_SIZE),
        "params":  params,
        "gflops":  round(gflops, 3),
        "latency": latency,
        "memory":  memory,
    }


# ── Run all models ────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  Efficiency Profiler — {args.dataset.upper()}  "
      f"input {INPUT_SIZE}  device {DEVICE}")
print(f"  Warmup: {args.warmup}  Runs: {args.runs}  Batch: {args.batch}")
print(f"{'='*72}")

all_profiles: dict = {}
for name in MODELS:
    try:
        all_profiles[name] = profile_model(name)
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")


# ── Per-model summary table ───────────────────────────────────────────────────
W = 20
HDR = (f"  {'Model':<{W}}  {'Params(M)':>9}  {'Train(M)':>8}  "
       f"{'Fixed(M)':>8}  {'GFLOPs':>7}  "
       f"{'Lat mean':>9}  {'Lat std':>8}  "
       f"{'Tput(img/s)':>11}  {'PeakMem(MB)':>11}")

print(f"\n{'─'*len(HDR)}")
print(f"  Per-model efficiency  [{DEVICE.upper()}  batch={args.batch}]")
print(f"{'─'*len(HDR)}")
print(HDR)
print(f"{'─'*len(HDR)}")

for name, prof in all_profiles.items():
    pr  = prof["params"]
    lat = prof["latency"]
    mem = prof["memory"]
    print(f"  {name:<{W}}  {pr['total_M']:>9.3f}  {pr['trainable_M']:>8.3f}  "
          f"{pr['fixed_M']:>8.3f}  {prof['gflops']:>7.3f}  "
          f"{lat['mean_ms']:>8.2f}ms  {lat['std_ms']:>7.2f}ms  "
          f"{lat['throughput']:>11.1f}  {mem.get('peak_mb', 0):>11.2f}")

print(f"{'─'*len(HDR)}")


# ── Paired comparison table ───────────────────────────────────────────────────
pairs_available = [
    (b, s) for b, s in BACKBONE_PAIRS
    if b in all_profiles and s in all_profiles
]

if pairs_available:
    print(f"\n{'─'*72}")
    print(f"  Shape vs. baseline comparison")
    print(f"{'─'*72}")
    print(f"  {'Backbone':<14}  {'Metric':<14}  "
          f"{'Baseline':>10}  {'Shape':>10}  {'Delta':>10}  {'Relative':>10}")
    print(f"{'─'*72}")

    for base_name, shape_name in pairs_available:
        b = all_profiles[base_name]
        s = all_profiles[shape_name]

        backbone = base_name.replace("baseline_", "")

        def row(label, bval, sval, unit=""):
            delta    = sval - bval
            rel      = (delta / bval * 100) if bval != 0 else 0.0
            sign     = "▼" if delta < 0 else "▲"
            rel_str  = f"{sign}{abs(rel):.1f}%"
            print(f"  {backbone:<14}  {label:<14}  "
                  f"{bval:>9.3f}{unit}  {sval:>9.3f}{unit}  "
                  f"{delta:>+9.3f}{unit}  {rel_str:>10}")

        row("Params (M)",    b["params"]["total_M"],           s["params"]["total_M"])
        row("Trainable (M)", b["params"]["trainable_M"],       s["params"]["trainable_M"])
        row("GFLOPs",        b["gflops"],                      s["gflops"])
        row("Latency (ms)",  b["latency"]["mean_ms"],          s["latency"]["mean_ms"])
        row("Tput (img/s)",  b["latency"]["throughput"],       s["latency"]["throughput"])
        row("PeakMem (MB)",  b["memory"].get("peak_mb", 0.0), s["memory"].get("peak_mb", 0.0))
        print(f"{'─'*72}")


# ── Save results ──────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
stem      = Path(args.results_dir) / f"profile_{args.dataset}_{timestamp}"

# JSON
with open(f"{stem}.json", "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "dataset":   args.dataset,
        "device":    DEVICE,
        "input":     list(INPUT_SIZE),
        "warmup":    args.warmup,
        "runs":      args.runs,
        "profiles":  all_profiles,
    }, f, indent=2)

# CSV (flat, one row per model — easy to paste into Excel / paper tables)
csv_fields = [
    "model", "dataset", "params_total_M", "params_trainable_M", "params_fixed_M",
    "gflops", "latency_mean_ms", "latency_std_ms", "latency_min_ms", "latency_max_ms",
    "throughput_img_s", "peak_mem_mb",
]
with open(f"{stem}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=csv_fields)
    w.writeheader()
    for name, prof in all_profiles.items():
        w.writerow({
            "model":               name,
            "dataset":             prof["dataset"],
            "params_total_M":      prof["params"]["total_M"],
            "params_trainable_M":  prof["params"]["trainable_M"],
            "params_fixed_M":      prof["params"]["fixed_M"],
            "gflops":              prof["gflops"],
            "latency_mean_ms":     prof["latency"]["mean_ms"],
            "latency_std_ms":      prof["latency"]["std_ms"],
            "latency_min_ms":      prof["latency"]["min_ms"],
            "latency_max_ms":      prof["latency"]["max_ms"],
            "throughput_img_s":    prof["latency"]["throughput"],
            "peak_mem_mb":         prof["memory"].get("peak_mb", 0.0),
        })

# TXT (human-readable, mirrors console output)
with open(f"{stem}.txt", "w") as f:
    f.write(f"Efficiency Profile — {args.dataset.upper()}\n")
    f.write(f"Timestamp : {datetime.now().isoformat()}\n")
    f.write(f"Device    : {DEVICE}  |  Input: {INPUT_SIZE}  |  "
            f"Warmup: {args.warmup}  Runs: {args.runs}\n\n")
    for name, prof in all_profiles.items():
        pr  = prof["params"]
        lat = prof["latency"]
        mem = prof["memory"]
        f.write(f"{name}\n")
        f.write(f"  Params total      : {pr['total_M']:.3f} M\n")
        f.write(f"  Params trainable  : {pr['trainable_M']:.3f} M\n")
        f.write(f"  Params fixed      : {pr['fixed_M']:.3f} M\n")
        f.write(f"  GFLOPs            : {prof['gflops']:.3f}\n")
        f.write(f"  Latency mean      : {lat['mean_ms']:.3f} ms\n")
        f.write(f"  Latency std       : {lat['std_ms']:.3f} ms\n")
        f.write(f"  Latency min/max   : {lat['min_ms']:.3f} / {lat['max_ms']:.3f} ms\n")
        f.write(f"  Throughput        : {lat['throughput']:.1f} img/s\n")
        f.write(f"  Peak GPU memory   : {mem.get('peak_mb', 0):.2f} MB\n\n")

print(f"\n  Saved → {stem}.json")
print(f"  Saved → {stem}.csv")
print(f"  Saved → {stem}.txt")
print(f"\n  Tip: install fvcore for accurate FLOP counts:  pip install fvcore")
