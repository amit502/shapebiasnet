# ShapeBiasNet

This repository contains code for training and evaluating dual-stream CNNs that integrate a **shape-biased pathway** into standard ResNet architectures. The models are designed to improve robustness under common corruptions (CIFAR-10-C and CIFAR-100-C) while maintaining competitive clean accuracy.

---

## Quick Start

1. **Place datasets** in `./data`:
   - CIFAR-10, CIFAR-10-C
   - CIFAR-100, CIFAR-100-C

2. **Train models**:

```bash
# Train all models on CIFAR-10 for 40 epochs and run all configs
python train.py --dataset cifar10 --epochs 40 --run_all

# Train Baseline ResNet-50 on CIFAR-10 for 80 epochs
python train.py --dataset cifar10 --epochs 80 --model baseline_res50

# Train Shape-ResNet50 on CIFAR-10 for 80 epochs
python train.py --dataset cifar10 --epochs 80 --model shape_res50
```

# Evaluate all models on CIFAR-10

python eval.py --dataset cifar10 --eval_all

# Evaluate specific models using checkpoints

python eval.py --dataset cifar10 --model baseline_res50 --ckpt checkpoints/baseline_res50_cifar10.pt
python eval.py --dataset cifar10 --model shape_res50 --ckpt checkpoints/shape_res50_cifar10.pt

./data # CIFAR-10, CIFAR-10-C, CIFAR-100, CIFAR-100-C datasets
./checkpoints # Saved model checkpoints
train.py # Training script
eval.py # Evaluation script
README.md # This file
