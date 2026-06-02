"""
models.py — ShapeBiasNet + ResNet baselines + gating ablations.

Models
------
Baselines  : baseline_res18 / res34 / res50 / res101
Main models: shape_res18 / res34 / res50 / res101
Ablations  : shape_res18_early_gate / late_gate / gate_only / early_gate_nofuse
             (gating mechanism ablations, ResNet-18 / CIFAR only)

Usage
-----
    from models import build_model, MODEL_NAMES
    model = build_model("shape_res18", num_classes=10, dataset="cifar10")
    model = build_model("shape_res50", num_classes=100, dataset="cifar100")
    model = build_model("shape_res18", num_classes=1000, dataset="imagenet")
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50


# ─────────────────────────────────────────────────────────────
#  BASELINES
# ─────────────────────────────────────────────────────────────

class BaselineResNet18(nn.Module):
    """ResNet-18 with dataset-aware stem. CIFAR: 3×3 stem, no maxpool."""
    def __init__(self, num_classes: int = 10, dataset: str = "cifar10"):
        super().__init__()
        net = resnet18(weights=None)
        if "cifar" in dataset:
            net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512, num_classes)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class BaselineResNet34(nn.Module):
    """ResNet-34 with dataset-aware stem."""
    def __init__(self, num_classes: int = 10, dataset: str = "cifar10"):
        super().__init__()
        from torchvision.models import resnet34
        net = resnet34(weights=None)
        if "cifar" in dataset:
            net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512, num_classes)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class BaselineResNet50(nn.Module):
    """ResNet-50 with dataset-aware stem."""
    def __init__(self, num_classes: int = 10, dataset: str = "cifar10"):
        super().__init__()
        net = resnet50(weights=None)
        if "cifar" in dataset:
            net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(2048, num_classes)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class BaselineResNet101(nn.Module):
    """ResNet-101 with dataset-aware stem."""
    def __init__(self, num_classes: int = 10, dataset: str = "cifar10"):
        super().__init__()
        from torchvision.models import resnet101
        net = resnet101(weights=None)
        if "cifar" in dataset:
            net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(2048, num_classes)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────
#  SHAPE STREAM COMPONENTS
# ─────────────────────────────────────────────────────────────

class OrientationBank(nn.Module):
    """
    16 oriented Sobel edge detectors uniformly spaced over [0, π).
    Applied to grayscale input. Output normalised per spatial location.

    Downsamples input to target_h before edge detection so shape stream
    spatial resolution always matches the RGB backbone stage it fuses with:
        CIFAR    (32×32 input) : target_h=32 → no-op   → shape: 32→16→8
        ImageNet (224×224)     : target_h=56 → /4      → shape: 56→28→14
    """
    def __init__(self, out_ch: int = 16):
        super().__init__()
        gx = torch.tensor([[1, 0,-1],[2, 0,-2],[1, 0,-1]], dtype=torch.float32)
        gy = torch.tensor([[1, 2, 1],[0, 0, 0],[-1,-2,-1]], dtype=torch.float32)
        kernels = []
        for k in range(out_ch):
            theta = math.pi * k / out_ch
            kernels.append(math.cos(theta) * gx + math.sin(theta) * gy)
        self.register_buffer("weight", torch.stack(kernels).unsqueeze(1))

    def forward(self, x: torch.Tensor, target_h: int = 32) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        s = max(1, gray.shape[2] // target_h)
        if s > 1:
            gray = F.avg_pool2d(gray, kernel_size=s, stride=s)
        e = F.conv2d(gray, self.weight, padding=1).abs()
        return e / (e.mean(dim=(2, 3), keepdim=True) + 1e-6)


class ShapeDiffusion(nn.Module):
    """
    Laplacian-guided diffusion block.

    Applies one step of discrete Laplacian diffusion (coefficient 0.12)
    to spread edge information spatially, then depthwise-separable conv
    with a residual connection. Suppresses texture noise while preserving
    boundaries.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        lap = torch.tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]])
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lap_x = F.conv2d(x, self.lap.expand(x.size(1), 1, 3, 3),
                         padding=1, groups=x.size(1))
        y = self.pw(self.dw(x + 0.12 * lap_x))
        return F.relu(self.bn(y) + x)


class ShapeEncoder(nn.Module):
    """
    Hierarchical shape feature extractor.

    out_ch   : final-stage channels; earlier stages are out_ch//4 and out_ch//2.
    n_blocks : ShapeDiffusion blocks per stage (b1, b2, b3).

    Spatial resolutions:
        CIFAR    : 32×32 → 16×16 → 8×8
        ImageNet : 56×56 → 28×28 → 14×14
    """
    def __init__(self, out_ch: int = 256, n_blocks: tuple = (2, 2, 1)):
        super().__init__()
        c1, c2, c3 = max(16, out_ch // 4), max(32, out_ch // 2), out_ch
        self.out_ch = c3
        self.edge   = OrientationBank(out_ch=16)
        self.stage1 = nn.Sequential(
            nn.Conv2d(16, c1, kernel_size=1),
            nn.BatchNorm2d(c1), nn.ReLU(),
            *[ShapeDiffusion(c1) for _ in range(n_blocks[0])],
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2), nn.ReLU(),
            *[ShapeDiffusion(c2) for _ in range(n_blocks[1])],
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3), nn.ReLU(),
            *[ShapeDiffusion(c3) for _ in range(n_blocks[2])],
        )

    def forward(self, x: torch.Tensor, target_h: int = 32):
        s1 = self.stage1(self.edge(x, target_h))
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return s1, s2, s3


# ─────────────────────────────────────────────────────────────
#  RGB BACKBONE
# ─────────────────────────────────────────────────────────────

class RGBResNet(nn.Module):
    """
    ResNet-18/34/50/101 with dataset-aware stem.
    Returns features from layer1, layer2, layer3 only (layer4 excluded).

    Output channels (self.out_ch = [l1, l2, l3]):
        ResNet-18/34  : [64,   128,  256]
        ResNet-50/101 : [256,  512, 1024]
    """
    def __init__(self, depth: str = "18", dataset: str = "cifar10"):
        super().__init__()
        from torchvision.models import resnet34, resnet101
        nets = {
            "18":  resnet18(weights=None),
            "34":  resnet34(weights=None),
            "50":  resnet50(weights=None),
            "101": resnet101(weights=None),
        }
        assert depth in nets, f"depth must be one of {list(nets.keys())}"
        net = nets[depth]
        if "cifar" in dataset:
            net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        self.stem   = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.l1, self.l2, self.l3 = net.layer1, net.layer2, net.layer3
        self.out_ch = [64, 128, 256] if depth in ("18", "34") else [256, 512, 1024]

    def forward_until_l2(self, x: torch.Tensor):
        r1 = self.l1(self.stem(x))
        return r1, self.l2(r1)

    def forward(self, x: torch.Tensor):
        r1, r2 = self.forward_until_l2(x)
        return r1, r2, self.l3(r2)


# ─────────────────────────────────────────────────────────────
#  MAIN MODEL
# ─────────────────────────────────────────────────────────────

class ShapeBiasNet(nn.Module):
    """
    Dual-stream network: shape stream + ResNet backbone → late fusion.

    Input ──┬── ShapeEncoder ──────────────────────► s3
            └── RGBResNet (layers 1-3) ────────────► r3
                           ↓
              concat(r3, s3) → Conv1×1 → BN → ReLU
                             → Conv3×3 → BN → ReLU
                             → GAP → Linear(256, num_classes)

    No gating. Robustness comes from the shape stream's inductive bias,
    not data augmentation.
    """
    def __init__(self, rgb_type: str = "18", num_classes: int = 10,
                 dataset: str = "cifar10"):
        super().__init__()
        assert rgb_type in ("18", "34", "50", "101"), \
            f"rgb_type must be one of ('18','34','50','101')"

        self.rgb     = RGBResNet(depth=rgb_type, dataset=dataset)
        rgb_out_ch   = self.rgb.out_ch[2]

        _NBLOCKS = {"18": (2,2,1), "34": (2,2,1), "50": (2,2,1), "101": (2,2,1)}
        self.shape   = ShapeEncoder(out_ch=256, n_blocks=_NBLOCKS[rgb_type])

        fusion_in_ch  = rgb_out_ch + 256
        fusion_mid_ch = max(320, fusion_in_ch // 4)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_ch, fusion_mid_ch, kernel_size=1),
            nn.BatchNorm2d(fusion_mid_ch), nn.ReLU(),
            nn.Conv2d(fusion_mid_ch, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, r3 = self.rgb(x)
        _, _, s3 = self.shape(x, target_h=r3.shape[2] * 4)
        return self.head(self.fusion(torch.cat([r3, s3], dim=1)))


# ─────────────────────────────────────────────────────────────
#  GATING ABLATIONS  (ResNet-18 / CIFAR only)
# ─────────────────────────────────────────────────────────────

class _LearnedGate(nn.Module):
    """
    Shape-guided multiplicative gate.
    R̃ = (1-α)·R + α·(R ⊙ σ(G(S))), α ∈ (0,1) learned per gate.
    Raw parameter initialised to 0  →  effective α = sigmoid(0) = 0.5.
    """
    def __init__(self, rgb_ch: int, shape_ch: int):
        super().__init__()
        self.proj  = nn.Conv2d(shape_ch, rgb_ch, kernel_size=1)
        self.bn    = nn.BatchNorm2d(rgb_ch)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, rgb: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        mask  = torch.sigmoid(self.bn(self.proj(shape)))
        alpha = torch.sigmoid(self.alpha)
        return (1.0 - alpha) * rgb + alpha * (rgb * mask)


class ShapeBiasNetAblation(nn.Module):
    """
    Gating ablations of ShapeBiasNet — ResNet-18 / CIFAR only.

    Variants
    --------
    early_gate           : gate r1,r2        → concat(r3,s3) → fusion head
    late_gate            : gate r3           → concat(r3,s3) → fusion head
    gate_only            : gate r3, no concat → classify from gated r3
    early_gate_nofuse    : gate r1,r2, no concat → classify from r3
    early_fuse           : inject s2 at r2   → l3 → classify
    early_fuse_early_gate: gate r1,r2 + inject s2 at r2 → l3 → classify
    early_fuse_late_gate : inject s2 at r2 + gate r3 → classify

    Seed 42 / CIFAR-10-C results:
        early_gate           : clean=86.54%  mCA=75.75%
        late_gate            : clean=86.25%  mCA=76.24%
        gate_only            : clean=86.89%  mCA=71.42%
        early_gate_nofuse    : clean=87.39%  mCA=71.89%
        early_fuse           : clean=87.49%  mCA=76.56%  (seed 42 only — inconsistent)
        early_fuse_early_gate: clean=87.34%  mCA=74.80%
    vs main model (shape_res18): mCA=75.97±0.23%
    """
    VARIANTS = (
        "early_gate", "late_gate", "gate_only", "early_gate_nofuse",
        "early_fuse", "early_fuse_early_gate", "early_fuse_late_gate",
    )

    def __init__(self, num_classes: int, dataset: str, variant: str):
        super().__init__()
        assert "cifar" in dataset, "ShapeBiasNetAblation is CIFAR-only"
        assert variant in self.VARIANTS, f"variant must be one of {self.VARIANTS}"
        self.variant = variant

        self.rgb   = RGBResNet(depth="18", dataset=dataset)
        self.shape = ShapeEncoder(out_ch=256, n_blocks=(2, 2, 1))

        r_ch = self.rgb.out_ch   # [64, 128, 256]
        s_ch = [64, 128, 256]    # ShapeEncoder stage channels

        if variant in ("early_gate", "early_gate_nofuse", "early_fuse_early_gate"):
            self.gate1 = _LearnedGate(r_ch[0], s_ch[0])
            self.gate2 = _LearnedGate(r_ch[1], s_ch[1])
        if variant in ("late_gate", "gate_only", "early_fuse_late_gate"):
            self.gate3 = _LearnedGate(r_ch[2], s_ch[2])

        if variant in ("early_fuse", "early_fuse_early_gate", "early_fuse_late_gate"):
            self.early_proj = nn.Sequential(
                nn.Conv2d(r_ch[1] + s_ch[1], r_ch[1], kernel_size=1, bias=False),
                nn.BatchNorm2d(r_ch[1]), nn.ReLU(),
            )

        if variant in ("early_gate", "late_gate"):
            fusion_in  = r_ch[2] + 256
            fusion_mid = max(320, fusion_in // 4)
            self.fusion = nn.Sequential(
                nn.Conv2d(fusion_in, fusion_mid, kernel_size=1),
                nn.BatchNorm2d(fusion_mid), nn.ReLU(),
                nn.Conv2d(fusion_mid, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(256, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(r_ch[2], num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_h = x.shape[2]
        s1, s2, s3 = self.shape(x, target_h=target_h)

        if self.variant == "early_gate":
            r1 = self.gate1(self.rgb.l1(self.rgb.stem(x)), s1)
            r2 = self.gate2(self.rgb.l2(r1), s2)
            r3 = self.rgb.l3(r2)
            return self.head(self.fusion(torch.cat([r3, s3], dim=1)))

        if self.variant == "late_gate":
            _, _, r3_raw = self.rgb(x)
            r3 = self.gate3(r3_raw, s3)
            return self.head(self.fusion(torch.cat([r3, s3], dim=1)))

        if self.variant == "gate_only":
            _, _, r3_raw = self.rgb(x)
            return self.head(self.gate3(r3_raw, s3))

        if self.variant == "early_gate_nofuse":
            r1 = self.gate1(self.rgb.l1(self.rgb.stem(x)), s1)
            r2 = self.gate2(self.rgb.l2(r1), s2)
            r3 = self.rgb.l3(r2)
            return self.head(r3)

        if self.variant == "early_fuse":
            r1, r2 = self.rgb.forward_until_l2(x)
            r3 = self.rgb.l3(self.early_proj(torch.cat([r2, s2], dim=1)))
            return self.head(r3)

        if self.variant == "early_fuse_early_gate":
            r1 = self.gate1(self.rgb.l1(self.rgb.stem(x)), s1)
            r2 = self.gate2(self.rgb.l2(r1), s2)
            r3 = self.rgb.l3(self.early_proj(torch.cat([r2, s2], dim=1)))
            return self.head(r3)

        # early_fuse_late_gate
        r1, r2 = self.rgb.forward_until_l2(x)
        r3_raw = self.rgb.l3(self.early_proj(torch.cat([r2, s2], dim=1)))
        return self.head(self.gate3(r3_raw, s3))


# ─────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────

MODEL_NAMES = [
    # ── Baselines ─────────────────────────────────────────────
    "baseline_res18",
    "baseline_res34",
    "baseline_res50",
    "baseline_res101",
    # ── ShapeBiasNet ──────────────────────────────────────────
    "shape_res18",
    "shape_res34",
    "shape_res50",
    "shape_res101",
    # ── Ablations (ResNet-18 / CIFAR only) ────────────────────
    "shape_res18_early_gate",
    "shape_res18_late_gate",
    "shape_res18_gate_only",
    "shape_res18_early_gate_nofuse",
    "shape_res18_early_fuse",
    "shape_res18_early_fuse_early_gate",
    "shape_res18_early_fuse_late_gate",
]


def build_model(name: str, num_classes: int,
                dataset: str = "cifar10") -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        name        : one of MODEL_NAMES
        num_classes : 10 (CIFAR-10), 100 (CIFAR-100), 1000 (ImageNet)
        dataset     : "cifar10" | "cifar100" | "imagenet" | "imagenet100"

    Returns:
        nn.Module, randomly initialised
    """
    dataset = dataset.lower()
    kw      = dict(num_classes=num_classes, dataset=dataset)

    if name == "baseline_res18":  return BaselineResNet18(**kw)
    if name == "baseline_res34":  return BaselineResNet34(**kw)
    if name == "baseline_res50":  return BaselineResNet50(**kw)
    if name == "baseline_res101": return BaselineResNet101(**kw)

    if name == "shape_res18":  return ShapeBiasNet("18",  **kw)
    if name == "shape_res34":  return ShapeBiasNet("34",  **kw)
    if name == "shape_res50":  return ShapeBiasNet("50",  **kw)
    if name == "shape_res101": return ShapeBiasNet("101", **kw)

    _abl = ShapeBiasNetAblation
    if name == "shape_res18_early_gate":      return _abl(variant="early_gate",      **kw)
    if name == "shape_res18_late_gate":       return _abl(variant="late_gate",       **kw)
    if name == "shape_res18_gate_only":       return _abl(variant="gate_only",       **kw)
    if name == "shape_res18_early_gate_nofuse":     return _abl(variant="early_gate_nofuse",     **kw)
    if name == "shape_res18_early_fuse":            return _abl(variant="early_fuse",            **kw)
    if name == "shape_res18_early_fuse_early_gate": return _abl(variant="early_fuse_early_gate", **kw)
    if name == "shape_res18_early_fuse_late_gate":  return _abl(variant="early_fuse_late_gate",  **kw)

    raise ValueError(f"Unknown model '{name}'. Choose from: {MODEL_NAMES}")
