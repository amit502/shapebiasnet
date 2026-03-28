# Shape-ResNet

This repository contains code for training and evaluating dual-stream CNNs that integrate a **shape-biased pathway** into standard ResNet architectures. The models are designed to improve robustness under common corruptions (CIFAR-10-C and CIFAR-100-C) while maintaining competitive clean accuracy.

---

## Quick Start

1. **Place datasets** in `./data`:
   - CIFAR-10, CIFAR-10-C
   - CIFAR-100, CIFAR-100-C
2. **Install dependencies** :
   - create env
   - Install dependencies

   ```bash
       pip install -r requirements.txt

   ```

3. **Train models**:

```bash
# Train all models on CIFAR-10 for 40 epochs and run all configs
python train.py --dataset cifar10 --epochs 40 --run_all

# You may train a particular model like Baseline ResNet-50 on CIFAR-10 for 80 epochs
python train.py --dataset cifar10 --epochs 80 --model baseline_res50

# You may train a particular model like Shape-ResNet50 on CIFAR-10 for 80 epochs
python train.py --dataset cifar10 --epochs 80 --model shape_res50
```

# Evaluate all models on CIFAR-10

```bash
python eval.py --dataset cifar10 --eval_all
```

# Evaluate specific models using checkpoints

```bash

python eval.py --dataset cifar10 --model baseline_res50 --ckpt checkpoints/baseline_res50_cifar10.pt
python eval.py --dataset cifar10 --model shape_res50 --ckpt checkpoints/shape_res50_cifar10.pt
```

all models = ["baseline_res18","baseline_res50","shape_custom","shape_res18","shape_res50"]

./data # CIFAR-10, CIFAR-10-C, CIFAR-100, CIFAR-100-C datasets.\
./checkpoints # Saved model checkpoints.\
train.py # Training script.\
eval.py # Evaluation script.\
requirements.txt # Requirements.\
README.md # This file.

# ShapeBiasNet — Setup & Run Guide

## Repository Layout

```
.
├── models.py              # All model definitions (CIFAR + ImageNet)
├── train.py               # Training — auto-resumes on preemption
├── eval.py                # Corruption robustness eval — saves results to PVC
├── profiler.py            # FLOPs, latency, memory, param count
├── k8s/
│   ├── data-pvc.yaml      # One 300Gi PVC for everything
│   ├── downloader-pod.yaml# Interactive pod for data download
│   └── combined-job.yaml  # Train → Eval → Profile in one job
└── setup/
    ├── download_imagenet.sh   # Downloads + extracts ImageNet
    └── download_imagenet_c.sh # Downloads ImageNet-C from Zenodo
```

## Everything Lives on One PVC

```
shape-resnet-pvc  (300Gi, rook-cephfs-central)
└── /pvc/
    ├── imagenet/
    │   ├── train/          ImageNet train (1,281,167 images)
    │   ├── val/            ImageNet val   (50,000 images)
    │   └── ImageNet-C/     15 corruptions × 5 severities
    ├── checkpoints/        Model .pt files
    └── results/            Eval JSON/txt + profiler JSON/txt
```

The combined job mounts the whole PVC at `/pvc` and uses
`--data_dir /pvc/imagenet`, `--ckpt_dir /pvc/checkpoints`,
`--results_dir /pvc/results`. No path confusion.

---

## Step 1 — Create the PVC

```bash
kubectl apply -f k8s/data-pvc.yaml -n <your-namespace>

# Confirm it binds (takes ~30s)
kubectl get pvc shape-resnet-pvc -n <your-namespace>
# STATUS should be Bound
```

> **If you already have a 250GB+ PVC**: skip this step and replace
> `claimName: shape-resnet-pvc` with your existing PVC name in
> `downloader-pod.yaml` and `combined-job.yaml`.

---

## Step 2 — Upload Download Scripts

```bash
kubectl create configmap imagenet-download-scripts \
  --from-file=setup/download_imagenet.sh \
  --from-file=setup/download_imagenet_c.sh \
  -n <your-namespace>
```

---

## Step 3 — Download ImageNet + ImageNet-C

```bash
# Launch the downloader pod
kubectl apply -f k8s/downloader-pod.yaml -n <your-namespace>

# Wait for Running
kubectl get pod imagenet-downloader -n <your-namespace> -w

# Shell in
kubectl exec -it imagenet-downloader -n <your-namespace> -- bash
```

Inside the pod:

```bash
# Install tmux — survives disconnects
apt-get install -y tmux && tmux new -s dl

# Download ImageNet (~138GB train + 6.3GB val, ~2-4 hours)
bash /scripts/download_imagenet.sh

# Download ImageNet-C (~80GB, ~1-2 hours)
bash /scripts/download_imagenet_c.sh

# Detach: Ctrl+B, D  |  Reattach: tmux attach -t dl
```

Both scripts use `aria2c --continue=true` — fully resumable if interrupted.

Verify the final layout:

```bash
ls /pvc/imagenet/                  # train/  val/  ImageNet-C/
ls /pvc/imagenet/ImageNet-C/       # 15 corruption folders
ls /pvc/imagenet/ImageNet-C/gaussian_noise/   # 1  2  3  4  5
```

When done, delete the pod (data stays on PVC):

```bash
kubectl delete pod imagenet-downloader -n <your-namespace>
```

---

## Step 4 — Upload Your Code

```bash
kubectl create configmap shape-resnet-code \
  --from-file=models.py \
  --from-file=train.py \
  --from-file=eval.py \
  --from-file=profiler.py \
  -n <your-namespace>

# To update after code changes:
kubectl delete configmap shape-resnet-code -n <your-namespace>
kubectl create configmap shape-resnet-code \
  --from-file=models.py --from-file=train.py \
  --from-file=eval.py   --from-file=profiler.py \
  -n <your-namespace>
```

---

## Step 5 — Run the Pipeline

One command. Train → Eval → Profile. All saved to PVC.

```bash
kubectl apply -f k8s/combined-job.yaml -n <your-namespace>

# Stream logs live
kubectl logs -f job/shape-resnet-pipeline -n <your-namespace>

# Check status
kubectl get job shape-resnet-pipeline -n <your-namespace>
```

**What happens inside:**

1. `train.py` trains the model (90 epochs). Auto-resumes if the job was
   previously preempted — it detects the existing checkpoint and continues.
2. `eval.py --eval_all` evaluates all checkpoints on ImageNet-C and prints
   the same per-corruption table as your original script.
3. `profiler.py` measures FLOPs, latency, peak memory, and param counts.

**All output files land on the PVC at `/pvc/results/`:**

```
/pvc/results/
├── shape_res18_imagenet_results.json    full per-corruption accuracy data
├── shape_res18_imagenet_report.txt      printable table (same as stdout)
├── summary_imagenet.json                all models side by side
├── profile_imagenet_<timestamp>.json    FLOPs / latency / memory
└── profile_imagenet_<timestamp>.txt     human-readable profile table
```

---

## Running Locally (CIFAR, no Kubernetes)

```bash
pip install torch torchvision fvcore

# Train
python train.py --model shape_res18 --dataset cifar10
python train.py --model shape_res18 --dataset cifar100

# Eval (put CIFAR-10-C in ./data/CIFAR-10-C/)
python eval.py --dataset cifar10 --eval_all

# Profile
python profiler.py --dataset cifar10
```

---

## Nautilus Resource Policy

| Resource           | Request | Limit | Ratio |
| ------------------ | ------- | ----- | ----- |
| CPU (pipeline job) | 16      | 20    | 1.25  |
| RAM (pipeline job) | 64Gi    | 80Gi  | 1.25  |
| GPU (pipeline job) | 4       | 4     | 1.0   |
| CPU (downloader)   | 8       | 10    | 1.25  |
| RAM (downloader)   | 8Gi     | 10Gi  | 1.25  |

- CPU values are integers ✓
- GPU limits == GPU requests ✓
- No GPU on downloader pod ✓
- `backoffLimit: 4` for automatic retry on preemption ✓

---

## Useful Commands

```bash
# Check everything
kubectl get pods,jobs,pvc -n <your-namespace>

# Disk usage on PVC (exec into any running pod)
kubectl exec -it imagenet-downloader -n <your-namespace> -- df -h /pvc

# List result files
kubectl exec -it <any-pod> -n <your-namespace> -- ls -lh /pvc/results/

# Describe job (useful if it won't schedule)
kubectl describe job shape-resnet-pipeline -n <your-namespace>

# Kill and restart pipeline (train.py will resume)
kubectl delete job shape-resnet-pipeline -n <your-namespace>
kubectl apply  -f k8s/combined-job.yaml  -n <your-namespace>
```
