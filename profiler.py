"""
profiler.py — Efficiency profiling for all ShapeBiasNet models.

Reports per model:
  - Parameter count (M)
  - FLOPs / GFLOPs (via torch.profiler + manual FLOP count using fvcore)
  - GPU memory usage (peak, MB)
  - Inference latency (mean ± std over N warmup + timed runs, ms)

Results are printed as a formatted table AND saved to --results_dir as
JSON + txt so you never need to re-run.

Usage
------
    # Profile all models for CIFAR (32x32 input)
    python profiler.py --dataset cifar10

    # Profile all models for ImageNet (224x224 input)
    python profiler.py --dataset imagenet

    # Single model
    python profiler.py --dataset imagenet --model shape_res18

    # No GPU? Falls back to CPU latency measurement
    python profiler.py --dataset cifar10 --device cpu

Requirements
------------
    pip install fvcore   # for FLOPs counting
    (everything else is standard torch/torchvision)
"""

import argparse, json, os, time
from datetime import datetime

import torch
import torch.nn as nn
from models import build_model, MODEL_NAMES

# ──────────────────────────────────────────────────────────────
#  ARGS
# ──────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--dataset",     default="cifar10", choices=["cifar10","cifar100","imagenet"])
p.add_argument("--model",       default=None,      help="Single model (default: all)")
p.add_argument("--batch",       type=int, default=1,  help="Batch size for profiling (default 1 = single-sample latency)")
p.add_argument("--warmup",      type=int, default=20, help="Warmup iterations before timing")
p.add_argument("--runs",        type=int, default=100,help="Timed iterations for latency")
p.add_argument("--device",      default=None,      help="cuda or cpu (default: auto)")
p.add_argument("--results_dir", default="./results")
args = p.parse_args()

IS_IMAGENET  = args.dataset == "imagenet"
NUM_CLASSES  = 1000 if IS_IMAGENET else (10 if args.dataset == "cifar10" else 100)
INPUT_SIZE   = (args.batch, 3, 224, 224) if IS_IMAGENET else (args.batch, 3, 32, 32)
DEVICE       = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
MODELS       = [args.model] if args.model else MODEL_NAMES

os.makedirs(args.results_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────
#  FLOP COUNTING
# ──────────────────────────────────────────────────────────────
def count_flops(model: nn.Module, input_size: tuple) -> float:
    """
    Returns GFLOPs using fvcore's FlopCountAnalysis.
    Falls back to a manual estimate if fvcore is not installed.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.zeros(1, *input_size[1:]).to(next(model.parameters()).device)
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9  # GFLOPs
    except ImportError:
        # Fallback: rough estimate via torch.profiler
        return _flops_via_profiler(model, input_size)

def _flops_via_profiler(model: nn.Module, input_size: tuple) -> float:
    """Rough FLOP estimate using torch.profiler when fvcore unavailable."""
    dummy = torch.zeros(*input_size).to(next(model.parameters()).device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as prof:
        model(dummy)
    total = sum(e.flops for e in prof.key_averages() if e.flops > 0)
    return total / 1e9


# ──────────────────────────────────────────────────────────────
#  PARAM COUNT
# ──────────────────────────────────────────────────────────────
def count_params(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed     = total - trainable   # e.g. OrientationBank buffers
    return {
        "total_M":     round(total     / 1e6, 3),
        "trainable_M": round(trainable / 1e6, 3),
        "fixed_M":     round(fixed     / 1e6, 3),
    }


# ──────────────────────────────────────────────────────────────
#  LATENCY
# ──────────────────────────────────────────────────────────────
def measure_latency(model: nn.Module, input_size: tuple,
                    warmup: int, runs: int, device: str) -> dict:
    """
    Measures per-sample inference latency in milliseconds.
    Uses CUDA events for GPU (most accurate), time.perf_counter for CPU.
    """
    dummy = torch.zeros(*input_size).to(device)
    model = model.to(device).eval()

    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        if device == "cuda":
            # GPU timing via CUDA events
            timings = []
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            for _ in range(runs):
                start_ev.record()
                model(dummy)
                end_ev.record()
                torch.cuda.synchronize()
                timings.append(start_ev.elapsed_time(end_ev))  # ms
        else:
            # CPU timing
            timings = []
            for _ in range(runs):
                t0 = time.perf_counter()
                model(dummy)
                timings.append((time.perf_counter() - t0) * 1000)

    import statistics
    return {
        "mean_ms":   round(statistics.mean(timings),   3),
        "std_ms":    round(statistics.stdev(timings),  3),
        "min_ms":    round(min(timings),               3),
        "max_ms":    round(max(timings),               3),
        "device":    device,
        "batch_size": input_size[0],
    }


# ──────────────────────────────────────────────────────────────
#  MEMORY
# ──────────────────────────────────────────────────────────────
def measure_memory(model: nn.Module, input_size: tuple) -> dict:
    """
    Measures peak GPU memory allocated during a forward pass (MB).
    Returns zeros on CPU-only machines.
    """
    if not torch.cuda.is_available():
        return {"peak_mb": 0.0, "note": "CPU only — no GPU memory measured"}

    model = model.cuda().eval()
    dummy = torch.zeros(*input_size).cuda()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return {"peak_mb": round(peak_mb, 2)}


# ──────────────────────────────────────────────────────────────
#  PROFILE ONE MODEL
# ──────────────────────────────────────────────────────────────
def profile_model(name: str) -> dict:
    print(f"\n  Profiling: {name}  [{args.dataset}  {INPUT_SIZE}]")
    model = build_model(name, NUM_CLASSES, dataset=args.dataset)
    if name.startswith("shape"):
        model.alpha = 1.0
    model = model.to(DEVICE).eval()

    params  = count_params(model)
    gflops  = count_flops(model, INPUT_SIZE)
    latency = measure_latency(model, INPUT_SIZE, args.warmup, args.runs, DEVICE)
    memory  = measure_memory(model, INPUT_SIZE)

    return {
        "model":    name,
        "dataset":  args.dataset,
        "input":    list(INPUT_SIZE),
        "params":   params,
        "gflops":   round(gflops, 3),
        "latency":  latency,
        "memory":   memory,
    }


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
all_profiles = {}

for name in MODELS:
    try:
        profile = profile_model(name)
        all_profiles[name] = profile
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        continue

# ── formatted table ──
W = 22
print(f"\n{'='*90}")
print(f"  Efficiency Profile — {args.dataset.upper()}  (input {INPUT_SIZE})")
print(f"  Device: {DEVICE}  |  Latency batch size: {args.batch}  |  Runs: {args.runs}")
print(f"{'='*90}")
print(f"  {'Model':<{W}} | {'Params(M)':>9} | {'Trainable(M)':>12} | "
      f"{'Fixed(M)':>8} | {'GFLOPs':>7} | {'Lat mean(ms)':>12} | "
      f"{'Lat std(ms)':>11} | {'PeakMem(MB)':>11}")
print(f"  {'-'*88}")

for name, p in all_profiles.items():
    pr  = p["params"]
    lat = p["latency"]
    mem = p["memory"]
    print(f"  {name:<{W}} | {pr['total_M']:>9.3f} | {pr['trainable_M']:>12.3f} | "
          f"{pr['fixed_M']:>8.3f} | {p['gflops']:>7.3f} | "
          f"{lat['mean_ms']:>12.3f} | {lat['std_ms']:>11.3f} | "
          f"{mem.get('peak_mb',0):>11.2f}")

print(f"{'='*90}")

# ── save results ──
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
stem = f"profile_{args.dataset}_{timestamp}"

json_path = os.path.join(args.results_dir, f"{stem}.json")
with open(json_path, "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "dataset":   args.dataset,
        "device":    DEVICE,
        "input":     list(INPUT_SIZE),
        "profiles":  all_profiles,
    }, f, indent=2)

# plain text (capture table from above)
txt_path = os.path.join(args.results_dir, f"{stem}.txt")
with open(txt_path, "w") as f:
    f.write(f"Efficiency Profile — {args.dataset.upper()}\n")
    f.write(f"Timestamp : {datetime.now().isoformat()}\n")
    f.write(f"Device    : {DEVICE}\n")
    f.write(f"Input     : {INPUT_SIZE}\n\n")
    for name, p in all_profiles.items():
        pr  = p["params"]
        lat = p["latency"]
        mem = p["memory"]
        f.write(f"{name}\n")
        f.write(f"  Params (total)    : {pr['total_M']:.3f} M\n")
        f.write(f"  Params (trainable): {pr['trainable_M']:.3f} M\n")
        f.write(f"  Params (fixed)    : {pr['fixed_M']:.3f} M\n")
        f.write(f"  GFLOPs            : {p['gflops']:.3f}\n")
        f.write(f"  Latency mean      : {lat['mean_ms']:.3f} ms\n")
        f.write(f"  Latency std       : {lat['std_ms']:.3f} ms\n")
        f.write(f"  Latency min/max   : {lat['min_ms']:.3f} / {lat['max_ms']:.3f} ms\n")
        f.write(f"  Peak GPU memory   : {mem.get('peak_mb', 0):.2f} MB\n\n")

print(f"\n  [Saved] {json_path}")
print(f"  [Saved] {txt_path}")
print(f"\n  Tip: install fvcore for accurate FLOP counts:")
print(f"       pip install fvcore")