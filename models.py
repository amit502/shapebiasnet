# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from torchvision.models import resnet18, resnet50

# # ============================================================
# # ------------------ BASELINE MODELS -------------------------
# # ============================================================

# class BaselineResNet18(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.model = resnet18(weights=None)
#         self.model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
#         self.model.maxpool = nn.Identity()
#         self.model.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         return self.model(x)


# class BaselineResNet50(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.model = resnet50(weights=None)
#         self.model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
#         self.model.maxpool = nn.Identity()
#         self.model.fc = nn.Linear(2048, num_classes)

#     def forward(self, x):
#         return self.model(x)

# # ============================================================
# # ------------------ SHAPE COMPONENTS ------------------------
# # ============================================================

# class OrientationBank(nn.Module):
#     def __init__(self, out_ch=16):
#         super().__init__()
#         kernels = []
#         for k in range(out_ch):
#             theta = math.pi * k / out_ch
#             gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
#             gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
#             kernel = math.cos(theta)*gx + math.sin(theta)*gy
#             kernels.append(kernel)
#         weight = torch.stack(kernels).unsqueeze(1)
#         self.register_buffer("weight", weight)

#     def forward(self, x):
#         x = x.mean(1, keepdim=True)
#         e = F.conv2d(x, self.weight, padding=1)
#         e = torch.abs(e)
#         e = e / (e.mean(dim=(2,3), keepdim=True) + 1e-6)
#         return e


# class ShapeDiffusion(nn.Module):
#     def __init__(self, ch):
#         super().__init__()
#         self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
#         self.pw = nn.Conv2d(ch, ch, 1, bias=False)
#         self.bn = nn.BatchNorm2d(ch)
#         lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
#         self.register_buffer("lap", lap.view(1,1,3,3))

#     def forward(self, x):
#         lap = F.conv2d(x, self.lap.repeat(x.size(1),1,1,1),
#                        padding=1, groups=x.size(1))
#         y = x + 0.12 * lap
#         y = self.dw(y)
#         y = self.pw(y)
#         return F.relu(self.bn(y) + x)


# class ShapeEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.edge = OrientationBank(16)

#         self.stage1 = nn.Sequential(
#             nn.Conv2d(16, 64, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             ShapeDiffusion(64),
#             ShapeDiffusion(64),
#         )

#         self.stage2 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             ShapeDiffusion(128),
#             ShapeDiffusion(128),
#         )

#         self.stage3 = nn.Sequential(
#             nn.Conv2d(128, 256, 3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             ShapeDiffusion(256),
#         )

#     def forward(self, x):
#         s1 = self.stage1(self.edge(x))
#         s2 = self.stage2(s1)
#         s3 = self.stage3(s2)
#         return s1, s2, s3


# class ShapeGate(nn.Module):
#     def __init__(self, rgb_ch, shape_ch):
#         super().__init__()
#         self.proj = nn.Conv2d(shape_ch, rgb_ch, 1)
#         self.bn = nn.BatchNorm2d(rgb_ch)

#     def forward(self, rgb, shape, alpha):
#         g = torch.sigmoid(self.bn(self.proj(shape)))
#         return rgb * (1 - alpha) + rgb * g * alpha

# # ============================================================
# # ---------------- RGB BACKBONES -----------------------------
# # ============================================================

# class RGBCustom(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.r1 = nn.Sequential(nn.Conv2d(3,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU())
#         self.r2 = nn.Sequential(nn.Conv2d(64,128,3,2,1), nn.BatchNorm2d(128), nn.ReLU())
#         self.r3 = nn.Sequential(
#             nn.Conv2d(128,256,3,2,1), nn.BatchNorm2d(256), nn.ReLU(),
#             nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU()
#         )

#     def forward(self,x):
#         r1 = self.r1(x)
#         r2 = self.r2(r1)
#         r3 = self.r3(r2)
#         return r1,r2,r3


# class RGBResNet(nn.Module):
#     def __init__(self, depth="18"):
#         super().__init__()
#         net = resnet18(weights=None) if depth=="18" else resnet50(weights=None)
#         net.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
#         net.maxpool = nn.Identity()
#         self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
#         self.l1, self.l2, self.l3 = net.layer1, net.layer2, net.layer3

#         # 🔹 store output channels for gates
#         if depth=="18":
#             self.out_ch = [64, 128, 256]
#         else:
#             self.out_ch = [256, 512, 1024]

#     def forward(self,x):
#         r1 = self.l1(self.stem(x))
#         r2 = self.l2(r1)
#         r3 = self.l3(r2)
#         return r1,r2,r3

# # ============================================================
# # ---------------- SHAPE-BIAS NET ----------------------------
# # ============================================================

# class ShapeBiasNet(nn.Module):
#     def __init__(self, rgb_type="custom", num_classes=10):
#         super().__init__()
#         self.shape = ShapeEncoder()
#         self.rgb   = RGBCustom() if rgb_type=="custom" else RGBResNet(rgb_type)

#         # 🔹 set gate channels correctly based on RGB backbone
#         if rgb_type=="custom" or rgb_type=="18":
#             # self.g1 = ShapeGate(64, 64)
#             # self.g2 = ShapeGate(128, 128)
#             # self.g3 = ShapeGate(256, 256)
#             fusion_in_ch = 512
#         elif rgb_type=="50":
#             # self.g1 = ShapeGate(256, 64)
#             # self.g2 = ShapeGate(512, 128)
#             # self.g3 = ShapeGate(1024, 256)
#             fusion_in_ch = 1024 + 256

#         self.fusion = nn.Sequential(
#             nn.Conv2d(fusion_in_ch, 320,1), nn.BatchNorm2d(320), nn.ReLU(),
#             nn.Conv2d(320,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU()
#         )

#         self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
#                                   nn.Linear(256, num_classes))
#         self.alpha = 0.0

#     def forward(self,x):
#         s1,s2,s3 = self.shape(x)
#         r1,r2,r3 = self.rgb(x)

#         # r1 = self.g1(r1, F.interpolate(s1, r1.shape[2:]), self.alpha)
#         # r2 = self.g2(r2, F.interpolate(s2, r2.shape[2:]), self.alpha)
#         # r3 = self.g3(r3, F.interpolate(s3, r3.shape[2:]), self.alpha)

#         f = self.fusion(torch.cat([r3, F.interpolate(s3, r3.shape[2:])],1))
#         return self.head(f)

# # ============================================================
# # ----------------- MODEL REGISTRY ---------------------------
# # ============================================================

# def build_model(name, num_classes):
#     if name=="baseline_res18": return BaselineResNet18(num_classes)
#     if name=="baseline_res50": return BaselineResNet50(num_classes)
#     if name=="shape_custom":   return ShapeBiasNet("custom", num_classes)
#     if name=="shape_res18":    return ShapeBiasNet("18", num_classes)
#     if name=="shape_res50":    return ShapeBiasNet("50", num_classes)
#     raise ValueError("Unknown model name")






# """
# models.py — ShapeBiasNet + Baselines
# =====================================
# Dual-stream architecture combining a fixed orientation-based shape stream
# with a standard RGB backbone via late fusion. Supports CIFAR-10, CIFAR-100,
# and ImageNet out of the box.

# Design philosophy
# -----------------
# The shape stream uses a fixed (non-learnable) oriented edge filter bank
# followed by learnable Laplacian diffusion blocks. It runs in parallel with
# a standard RGB backbone. Their deepest features are concatenated and passed
# through a small fusion head — no gating, no attention, just late fusion.

# Ablation note
# -------------
# ShapeGate (alpha-based gating) was evaluated and showed no improvement over
# plain late fusion. It is preserved in the codebase for reference but is NOT
# used in any of the main models. The main contribution is the shape stream
# itself, not any gating mechanism.

# Quick start
# -----------
#     from models import build_model

#     model = build_model("shape_res18", num_classes=10,   dataset="cifar10")
#     model = build_model("shape_res18", num_classes=100,  dataset="cifar100")
#     model = build_model("shape_res18", num_classes=1000, dataset="imagenet")
#     model = build_model("baseline_res18", num_classes=10, dataset="cifar10")

# Available model names
# ---------------------
#     baseline_res18  — ResNet-18, no shape stream
#     baseline_res50  — ResNet-50, no shape stream
#     shape_custom    — ShapeBiasNet + lightweight custom RGB (CIFAR only)
#     shape_res18     — ShapeBiasNet + ResNet-18
#     shape_res50     — ShapeBiasNet + ResNet-50

# Extending
# ---------
# To plug in a new RGB backbone: create an nn.Module that returns
# (r1, r2, r3) from forward() and sets self.out_ch = [l1_ch, l2_ch, l3_ch].
# Register it in build_model().
# """

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet18, resnet50


# # ─────────────────────────────────────────────────────────────
# #  BASELINE MODELS
# # ─────────────────────────────────────────────────────────────

# class BaselineResNet18(nn.Module):
#     """
#     Standard ResNet-18 with dataset-aware stem.
#     CIFAR  : 3×3 conv, stride 1, no maxpool  (preserves 32×32 spatial)
#     ImageNet: 7×7 conv, stride 2, maxpool    (standard)
#     """
#     def __init__(self, num_classes: int = 10, dataset: str = "cifar10"):
#         super().__init__()
#         net = resnet18(weights=None)
#         if "cifar" in dataset:
#             net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#             net.maxpool = nn.Identity()
#         net.fc = nn.Linear(512, num_classes)
#         self.model = net

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.model(x)


# class BaselineResNet50(nn.Module):
#     """
#     Standard ResNet-50 with dataset-aware stem.
#     CIFAR  : 3×3 conv, stride 1, no maxpool
#     ImageNet: 7×7 conv, stride 2, maxpool
#     """
#     def __init__(self, num_classes: int = 10, dataset: str = "cifar10"):
#         super().__init__()
#         net = resnet50(weights=None)
#         if "cifar" in dataset:
#             net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#             net.maxpool = nn.Identity()
#         net.fc = nn.Linear(2048, num_classes)
#         self.model = net

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.model(x)


# # ─────────────────────────────────────────────────────────────
# #  SHAPE STREAM COMPONENTS
# # ─────────────────────────────────────────────────────────────

# class OrientationBank(nn.Module):
#     """
#     Fixed (non-learnable) bank of oriented edge detectors.

#     Builds `out_ch` Sobel-based kernels uniformly spaced over [0, pi).
#     Applied to grayscale (mean of RGB) — colour invariant by design.
#     Output normalised per spatial location for scale invariance.
#     Resolution-agnostic: identical behaviour on 32×32 and 224×224.
#     """
#     def __init__(self, out_ch: int = 16):
#         super().__init__()
#         gx = torch.tensor([[1, 0,-1],[2, 0,-2],[1, 0,-1]], dtype=torch.float32)
#         gy = torch.tensor([[1, 2, 1],[0, 0, 0],[-1,-2,-1]], dtype=torch.float32)
#         kernels = []
#         for k in range(out_ch):
#             theta = math.pi * k / out_ch
#             kernels.append(math.cos(theta) * gx + math.sin(theta) * gy)
#         # shape: (out_ch, 1, 3, 3) — register as buffer (not a parameter)
#         self.register_buffer("weight", torch.stack(kernels).unsqueeze(1))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         gray = x.mean(dim=1, keepdim=True)
#         e    = F.conv2d(gray, self.weight, padding=1).abs()
#         return e / (e.mean(dim=(2, 3), keepdim=True) + 1e-6)


# class ShapeDiffusion(nn.Module):
#     """
#     Laplacian-guided diffusion block.

#     Applies a fixed discrete Laplacian (coefficient 0.12) to spread edge
#     information spatially along iso-contour lines, then a depthwise-separable
#     conv with a residual connection. This acts as an anisotropic diffusion
#     prior: suppresses high-frequency texture noise while preserving boundaries.
#     """
#     def __init__(self, ch: int):
#         super().__init__()
#         self.dw = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)
#         self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(ch)
#         lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
#         self.register_buffer("lap", lap.view(1, 1, 3, 3))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         lap_x = F.conv2d(x, self.lap.expand(x.size(1), 1, 3, 3),
#                          padding=1, groups=x.size(1))
#         y = self.pw(self.dw(x + 0.12 * lap_x))
#         return F.relu(self.bn(y) + x)


# class ShapeEncoder(nn.Module):
#     """
#     Hierarchical shape feature extractor.

#     Three progressive downsampling stages for CIFAR; four for ImageNet,
#     to keep the shape stream spatially compatible with the RGB backbone's
#     layer3 output before fusion.

#     Spatial resolutions:
#         CIFAR   (32×32)  : stage1=32×32, stage2=16×16, stage3=8×8
#         ImageNet(224×224): stage1=224×224, stage2=112×112, stage3=56×56

#     F.interpolate in ShapeBiasNet.forward() aligns s3 to r3's spatial size
#     (e.g. 14×14 for ImageNet ResNet layer3) — no extra stages needed.

#     Output: always 256 channels at s3.
#     """
#     def __init__(self):
#         super().__init__()
#         self.edge = OrientationBank(out_ch=16)

#         self.stage1 = nn.Sequential(
#             nn.Conv2d(16, 64, kernel_size=1),
#             nn.BatchNorm2d(64), nn.ReLU(),
#             ShapeDiffusion(64),
#             ShapeDiffusion(64),
#         )
#         self.stage2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128), nn.ReLU(),
#             ShapeDiffusion(128),
#             ShapeDiffusion(128),
#         )
#         self.stage3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256), nn.ReLU(),
#             ShapeDiffusion(256),
#         )
#         # No extra stages for ImageNet — F.interpolate in ShapeBiasNet.forward()
#         # aligns s3 to r3's spatial size regardless of input resolution.

#     def forward(self, x: torch.Tensor):
#         s1 = self.stage1(self.edge(x))
#         s2 = self.stage2(s1)
#         s3 = self.stage3(s2)
#         return s1, s2, s3


# # ─────────────────────────────────────────────────────────────
# #  ABLATION ONLY — not used in main models
# # ─────────────────────────────────────────────────────────────

# class ShapeGate(nn.Module):
#     """
#     [ABLATION ONLY — not used in main ShapeBiasNet]

#     Multiplicative gate that modulates RGB features by shape features.
#     Evaluated and found to offer no improvement over plain late fusion.
#     Kept for reference.

#     Usage (ablation only):
#         gate = ShapeGate(rgb_ch=256, shape_ch=256)
#         r3_gated = gate(r3, s3, alpha=1.0)
#     """
#     def __init__(self, rgb_ch: int, shape_ch: int):
#         super().__init__()
#         self.proj = nn.Conv2d(shape_ch, rgb_ch, kernel_size=1)
#         self.bn   = nn.BatchNorm2d(rgb_ch)

#     def forward(self, rgb: torch.Tensor,
#                 shape: torch.Tensor,
#                 alpha: float = 1.0) -> torch.Tensor:
#         g = torch.sigmoid(self.bn(self.proj(shape)))
#         return rgb * (1 - alpha) + rgb * g * alpha


# # ─────────────────────────────────────────────────────────────
# #  RGB BACKBONES
# # ─────────────────────────────────────────────────────────────

# class RGBCustom(nn.Module):
#     """
#     Lightweight 3-stage custom RGB backbone (CIFAR-scale only).
#     No pretrained weights. Used for ablations and fast experiments.
#     Output channels: [64, 128, 256]
#     """
#     def __init__(self):
#         super().__init__()
#         self.stage1 = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
#         self.stage2 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
#         self.stage3 = nn.Sequential(
#             nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
#             nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
#         self.out_ch = [64, 128, 256]

#     def forward(self, x: torch.Tensor):
#         r1 = self.stage1(x)
#         r2 = self.stage2(r1)
#         r3 = self.stage3(r2)
#         return r1, r2, r3


# class RGBResNet(nn.Module):
#     """
#     ResNet-18 or ResNet-50 with dataset-aware stem.
#     Returns features from layer1, layer2, layer3 (layer4 excluded —
#     fusion happens at intermediate depth to preserve spatial resolution).

#     Output channels (self.out_ch = [l1, l2, l3]):
#         ResNet-18 : [64,   128,  256]
#         ResNet-50 : [256,  512, 1024]
#     """
#     def __init__(self, depth: str = "18", dataset: str = "cifar10"):
#         super().__init__()
#         net = resnet18(weights=None) if depth == "18" else resnet50(weights=None)

#         if "cifar" in dataset:
#             net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#             net.maxpool = nn.Identity()
#         # ImageNet: keep original 7×7 stem + maxpool untouched

#         self.stem   = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
#         self.l1, self.l2, self.l3 = net.layer1, net.layer2, net.layer3
#         self.out_ch = [64, 128, 256] if depth == "18" else [256, 512, 1024]

#     def forward(self, x: torch.Tensor):
#         r1 = self.l1(self.stem(x))
#         r2 = self.l2(r1)
#         r3 = self.l3(r2)
#         return r1, r2, r3


# # ─────────────────────────────────────────────────────────────
# #  SHAPE-BIAS NET  (main model)
# # ─────────────────────────────────────────────────────────────

# class ShapeBiasNet(nn.Module):
#     """
#     Dual-stream network: shape stream + RGB backbone → late fusion.

#     Architecture
#     ────────────
#     Input ──┬── ShapeEncoder ─────────────────────────► s3 (256ch)
#             └── RGBBackbone (custom / ResNet-18 / -50) ► r3 (256 or 1024ch)
#                                    ↓
#                      concat(r3, interpolate(s3 → r3 size))
#                                    ↓
#                      Fusion: Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU
#                                    ↓
#                      GlobalAvgPool → Linear(256, num_classes)

#     No gating, no alpha, no attention — pure late fusion.
#     The shape stream provides structural/boundary features; the RGB stream
#     provides texture/colour/semantic features. Fusion learns to combine them.

#     Training protocol
#     ─────────────────
#     Trained on CLEAN data only (no corruption augmentation).
#     Robustness comes from the inductive bias of the shape stream, not
#     from data augmentation. This is the core contribution.

#     Args:
#         rgb_type    : "custom" | "18" | "50"
#         num_classes : 10, 100, or 1000
#         dataset     : "cifar10" | "cifar100" | "imagenet"
#     """
#     def __init__(self,
#                  rgb_type: str    = "custom",
#                  num_classes: int = 10,
#                  dataset: str     = "cifar10"):
#         super().__init__()

#         self.shape = ShapeEncoder()

#         if rgb_type == "custom":
#             self.rgb   = RGBCustom()
#             rgb_out_ch = 256
#         else:
#             self.rgb   = RGBResNet(depth=rgb_type, dataset=dataset)
#             rgb_out_ch = self.rgb.out_ch[2]   # layer3 channels

#         # shape stream always outputs 256ch at s3
#         fusion_in_ch = rgb_out_ch + 256

#         self.fusion = nn.Sequential(
#             nn.Conv2d(fusion_in_ch, 320, kernel_size=1),
#             nn.BatchNorm2d(320), nn.ReLU(),
#             nn.Conv2d(320, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256), nn.ReLU(),
#         )
#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(256, num_classes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         _, _, s3 = self.shape(x)
#         _, _, r3 = self.rgb(x)
#         # align shape spatial size to RGB (may differ for ImageNet)
#         s3 = F.interpolate(s3, size=r3.shape[2:], mode="bilinear", align_corners=False)
#         return self.head(self.fusion(torch.cat([r3, s3], dim=1)))


# # ─────────────────────────────────────────────────────────────
# #  PUBLIC API
# # ─────────────────────────────────────────────────────────────

# MODEL_NAMES = [
#     "baseline_res18",
#     "baseline_res50",
#     "shape_custom",
#     "shape_res18",
#     "shape_res50",
# ]


# def build_model(name: str,
#                 num_classes: int,
#                 dataset: str = "cifar10") -> nn.Module:
#     """
#     Instantiate a model by name.

#     Args:
#         name        : One of MODEL_NAMES.
#         num_classes : 10 (CIFAR-10), 100 (CIFAR-100), 1000 (ImageNet).
#         dataset     : "cifar10" | "cifar100" | "imagenet"

#     Returns:
#         nn.Module, randomly initialised.

#     Raises:
#         ValueError for unrecognised names.

#     Example:
#         model = build_model("shape_res18", num_classes=1000, dataset="imagenet")
#     """
#     dataset = dataset.lower()
#     kw = dict(num_classes=num_classes, dataset=dataset)

#     if name == "baseline_res18": return BaselineResNet18(**kw)
#     if name == "baseline_res50": return BaselineResNet50(**kw)
#     if name == "shape_custom":   return ShapeBiasNet("custom", **kw)
#     if name == "shape_res18":    return ShapeBiasNet("18",     **kw)
#     if name == "shape_res50":    return ShapeBiasNet("50",     **kw)

#     raise ValueError(f"Unknown model '{name}'. Choose from: {MODEL_NAMES}")



"""
models.py — ShapeBiasNet + Baselines
=====================================
Dual-stream architecture combining a fixed orientation-based shape stream
with a standard RGB backbone via late fusion. Supports CIFAR-10, CIFAR-100,
and ImageNet out of the box.

Design philosophy
-----------------
The shape stream uses a fixed (non-learnable) oriented edge filter bank
followed by learnable Laplacian diffusion blocks. It runs in parallel with
a standard RGB backbone. Their deepest features are concatenated and passed
through a small fusion head — no gating, no attention, just late fusion.

Ablation note
-------------
ShapeGate (alpha-based gating) was evaluated and showed no improvement over
plain late fusion. It is preserved in the codebase for reference but is NOT
used in any of the main models. The main contribution is the shape stream
itself, not any gating mechanism.

Quick start
-----------
    from models import build_model

    # CIFAR
    model = build_model("shape_res18", num_classes=10,   dataset="cifar10")
    model = build_model("shape_res50", num_classes=100,  dataset="cifar100")

    # ImageNet — ResNet
    model = build_model("shape_res50",  num_classes=1000, dataset="imagenet")
    model = build_model("shape_res101", num_classes=1000, dataset="imagenet")

    # ImageNet — Modern backbones
    model = build_model("shape_convnext_tiny", num_classes=1000, dataset="imagenet")
    model = build_model("shape_effnet_b4",     num_classes=1000, dataset="imagenet")

    # Baselines (no shape stream)
    model = build_model("baseline_res18",           num_classes=10,   dataset="cifar10")
    model = build_model("baseline_convnext_tiny",   num_classes=1000, dataset="imagenet")
    model = build_model("baseline_efficientnet_b4", num_classes=1000, dataset="imagenet")

Available model names
---------------------
    See MODEL_NAMES list at the bottom of this file.

Extending
---------
To plug in a new RGB backbone: create an nn.Module that returns
(r1, r2, r3) from forward() and sets self.out_ch = [l1_ch, l2_ch, l3_ch].
Register it in build_model().
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50


# ─────────────────────────────────────────────────────────────
#  BASELINE MODELS
#  Standard off-the-shelf backbones with no shape stream.
#  Used as direct comparison points.
# ─────────────────────────────────────────────────────────────

class BaselineResNet18(nn.Module):
    """
    Standard ResNet-18 with dataset-aware stem.
    CIFAR  : 3×3 conv, stride 1, no maxpool  (preserves 32×32 spatial)
    ImageNet: 7×7 conv, stride 2, maxpool    (standard)
    """
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


class BaselineResNet50(nn.Module):
    """
    Standard ResNet-50 with dataset-aware stem.
    CIFAR  : 3×3 conv, stride 1, no maxpool
    ImageNet: 7×7 conv, stride 2, maxpool
    """
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


class BaselineResNet34(nn.Module):
    """
    Standard ResNet-34 with dataset-aware stem.
    Same channel structure as ResNet-18 (basic blocks).
    """
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


class BaselineResNet101(nn.Module):
    """
    Standard ResNet-101 with dataset-aware stem.
    Same channel structure as ResNet-50 (bottleneck blocks).
    """
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


class BaselineConvNeXt(nn.Module):
    """
    Standard ConvNeXt-Tiny or ConvNeXt-Base baseline. ImageNet-scale only.
    Uses the patchify stem (4×4 stride-4) — not suitable for 32×32 CIFAR.
    """
    def __init__(self, size: str = "tiny", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        nets = {"tiny": convnext_tiny, "base": convnext_base}
        assert size in nets, f"ConvNeXt size must be one of {list(nets.keys())}"
        self.model = nets[size](weights=None, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class BaselineEfficientNet(nn.Module):
    """
    Standard EfficientNet-B0 or EfficientNet-B4 baseline. ImageNet-scale only.
    """
    def __init__(self, size: str = "b0", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets, f"EfficientNet size must be one of {list(nets.keys())}"
        self.model = nets[size](weights=None, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────
#  SHAPE STREAM COMPONENTS
# ─────────────────────────────────────────────────────────────

class OrientationBank(nn.Module):
    """
    Fixed (non-learnable) bank of oriented edge detectors.

    Builds `out_ch` Sobel-based kernels uniformly spaced over [0, pi).
    Applied to grayscale (mean of RGB) — colour invariant by design.
    Output normalised per spatial location for scale invariance.
    Resolution-agnostic: identical behaviour on 32×32 and 224×224.
    """
    def __init__(self, out_ch: int = 16):
        super().__init__()
        gx = torch.tensor([[1, 0,-1],[2, 0,-2],[1, 0,-1]], dtype=torch.float32)
        gy = torch.tensor([[1, 2, 1],[0, 0, 0],[-1,-2,-1]], dtype=torch.float32)
        kernels = []
        for k in range(out_ch):
            theta = math.pi * k / out_ch
            kernels.append(math.cos(theta) * gx + math.sin(theta) * gy)
        # shape: (out_ch, 1, 3, 3) — register as buffer (not a parameter)
        self.register_buffer("weight", torch.stack(kernels).unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True).clamp(0, 1)
        e    = F.conv2d(gray, self.weight, padding=1).abs()
        return e / (e.mean(dim=(2, 3), keepdim=True) + 1e-6)


class ShapeDiffusion(nn.Module):
    """
    Laplacian-guided diffusion block.

    Applies a fixed discrete Laplacian (coefficient 0.12) to spread edge
    information spatially along iso-contour lines, then a depthwise-separable
    conv with a residual connection. This acts as an anisotropic diffusion
    prior: suppresses high-frequency texture noise while preserving boundaries.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lap_x = F.conv2d(x, self.lap.expand(x.size(1), 1, 3, 3),
                         padding=1, groups=x.size(1))
        y = self.pw(self.dw(x + 0.12 * lap_x))
        return F.relu(self.bn(y) + x)


class ShapeEncoder(nn.Module):
    """
    Hierarchical shape feature extractor. Identical for all datasets.

    Spatial resolutions:
        CIFAR   (32×32)  : stage1=32×32, stage2=16×16, stage3=8×8
        ImageNet(224×224): stage1=224×224, stage2=112×112, stage3=56×56

    F.interpolate in ShapeBiasNet.forward() aligns s3 to r3's spatial size
    regardless of backbone — no extra stages needed for any backbone.

    Output: always 256 channels at s3.
    """
    def __init__(self):
        super().__init__()
        self.edge = OrientationBank(out_ch=16)

        self.stage1 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            ShapeDiffusion(64),
            ShapeDiffusion(64),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            ShapeDiffusion(128),
            ShapeDiffusion(128),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            ShapeDiffusion(256),
        )

    def forward(self, x: torch.Tensor):
        s1 = self.stage1(self.edge(x))
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return s1, s2, s3


# ─────────────────────────────────────────────────────────────
#  ABLATION ONLY — not used in main models
# ─────────────────────────────────────────────────────────────

class ShapeGate(nn.Module):
    """
    [ABLATION ONLY — not used in main ShapeBiasNet]

    Multiplicative gate that modulates RGB features by shape features.
    Evaluated and found to offer no improvement over plain late fusion.
    Kept for reference.
    """
    def __init__(self, rgb_ch: int, shape_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(shape_ch, rgb_ch, kernel_size=1)
        self.bn   = nn.BatchNorm2d(rgb_ch)

    def forward(self, rgb: torch.Tensor,
                shape: torch.Tensor,
                alpha: float = 1.0) -> torch.Tensor:
        g = torch.sigmoid(self.bn(self.proj(shape)))
        return rgb * (1 - alpha) + rgb * g * alpha


# ─────────────────────────────────────────────────────────────
#  RGB BACKBONES  (used inside ShapeBiasNet)
# ─────────────────────────────────────────────────────────────

class RGBCustom(nn.Module):
    """
    Lightweight 3-stage custom RGB backbone (CIFAR-scale only).
    No pretrained weights. Used for ablations and fast experiments.
    Output channels: [64, 128, 256]
    """
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.out_ch = [64, 128, 256]

    def forward(self, x: torch.Tensor):
        r1 = self.stage1(x)
        r2 = self.stage2(r1)
        r3 = self.stage3(r2)
        return r1, r2, r3


class RGBResNet(nn.Module):
    """
    ResNet-18 / 34 / 50 / 101 with dataset-aware stem.
    Returns features from layer1, layer2, layer3 only (layer4 excluded —
    fusion at intermediate depth preserves spatial resolution).

    Output channels (self.out_ch = [l1, l2, l3]):
        ResNet-18/34  : [64,   128,  256]   basic blocks
        ResNet-50/101 : [256,  512, 1024]   bottleneck blocks
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
        assert depth in nets, f"ResNet depth must be one of {list(nets.keys())}"
        net = nets[depth]

        if "cifar" in dataset:
            net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        # ImageNet: keep original 7×7 stem + maxpool untouched

        self.stem   = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.l1, self.l2, self.l3 = net.layer1, net.layer2, net.layer3
        self.out_ch = [64, 128, 256] if depth in ("18", "34") else [256, 512, 1024]

    def forward(self, x: torch.Tensor):
        r1 = self.l1(self.stem(x))
        r2 = self.l2(r1)
        r3 = self.l3(r2)
        return r1, r2, r3


class RGBConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny or ConvNeXt-Base as RGB backbone. ImageNet-scale only.
    Returns features from stages 1, 2, 3 (stage 4 excluded).

    ConvNeXt feature layout (net.features):
        [0] stem (patchify, stride 4)
        [1] stage1   [2] downsample
        [3] stage2   [4] downsample
        [5] stage3   [6] downsample
        [7] stage4   ← excluded

    Output channels (self.out_ch = [s1, s2, s3]):
        ConvNeXt-Tiny : [96,  192, 384]
        ConvNeXt-Base : [128, 256, 512]
    """
    def __init__(self, size: str = "tiny", dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        nets = {
            "tiny": (convnext_tiny(weights=None), [96,  192, 384]),
            "base": (convnext_base(weights=None), [128, 256, 512]),
        }
        assert size in nets, f"ConvNeXt size must be one of {list(nets.keys())}"
        net, self.out_ch = nets[size]
        f = net.features
        self.stem   = f[0]
        self.stage1 = f[1]
        self.down1  = f[2]
        self.stage2 = f[3]
        self.down2  = f[4]
        self.stage3 = f[5]
        # f[6], f[7] (downsample + stage4) excluded

    def forward(self, x: torch.Tensor):
        r1 = self.stage1(self.stem(x))
        r2 = self.stage2(self.down1(r1))
        r3 = self.stage3(self.down2(r2))
        return r1, r2, r3


class RGBEfficientNet(nn.Module):
    """
    EfficientNet-B0 or EfficientNet-B4 as RGB backbone. ImageNet-scale only.
    Returns features from three depth groups (early / mid / deep).

    EfficientNet features layout (net.features, 9 blocks 0-8):
        [0]      stem conv
        [1]-[2]  early MBConv blocks  → r1
        [3]-[4]  mid   MBConv blocks  → r2
        [5]-[6]  deep  MBConv blocks  → r3
        [7]-[8]  head conv + excluded

    Output channels (self.out_ch = [s1, s2, s3]):
        EfficientNet-B0 : [24,  40, 112]
        EfficientNet-B4 : [32,  56, 160]
    """
    def __init__(self, size: str = "b0", dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {
            "b0": (efficientnet_b0(weights=None), [24,  40, 112]),
            "b4": (efficientnet_b4(weights=None), [32,  56, 160]),
        }
        assert size in nets, f"EfficientNet size must be one of {list(nets.keys())}"
        net, self.out_ch = nets[size]
        f = net.features
        # self.stage1 = nn.Sequential(*f[0:3])   # stem + early blocks
        # self.stage2 = nn.Sequential(*f[3:5])   # mid blocks
        # self.stage3 = nn.Sequential(*f[5:7])   # deep blocks
        # # f[7], f[8] excluded

        self.stage1 = nn.Sequential(*f[0:3])   # stem + early blocks
        self.stage2 = nn.Sequential(*f[3:5])   # mid blocks
        self.stage3 = nn.Sequential(*f[5:6])   # deep blocks → 112ch (B0) / 160ch (B4)
        # f[6], f[7], f[8] excluded

    def forward(self, x: torch.Tensor):
        r1 = self.stage1(x)
        r2 = self.stage2(r1)
        r3 = self.stage3(r2)
        return r1, r2, r3


# ─────────────────────────────────────────────────────────────
#  SHAPE-BIAS NET  (main model)
# ─────────────────────────────────────────────────────────────

class ShapeBiasNet(nn.Module):
    """
    Dual-stream network: shape stream + RGB backbone → late fusion.

    Architecture
    ────────────
    Input ──┬── ShapeEncoder ──────────────────────────────► s3 (256ch)
            └── RGBBackbone (any registered backbone below) ► r3 (varies)
                                   ↓
                     concat(r3, interpolate(s3 → r3 size))
                                   ↓
                     Fusion: Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU
                                   ↓
                     GlobalAvgPool → Linear(256, num_classes)

    The fusion head is backbone-agnostic — fusion_in_ch is computed
    dynamically from rgb_out_ch + 256, so any backbone plugs in cleanly.

    No gating, no alpha, no attention — pure late fusion.
    Trained on CLEAN data only. Robustness comes from the shape stream's
    inductive bias, not from data augmentation.

    Args:
        rgb_type    : "custom" | "18" | "34" | "50" | "101" |
                      "convnext_tiny" | "convnext_base" |
                      "effnet_b0" | "effnet_b4"
        num_classes : 10, 100, or 1000
        dataset     : "cifar10" | "cifar100" | "imagenet"
    """
    def __init__(self,
                 rgb_type: str    = "custom",
                 num_classes: int = 10,
                 dataset: str     = "cifar10"):
        super().__init__()

        self.shape = ShapeEncoder()

        if rgb_type == "custom":
            self.rgb   = RGBCustom()
            rgb_out_ch = 256
        elif rgb_type in ("18", "34", "50", "101"):
            self.rgb   = RGBResNet(depth=rgb_type, dataset=dataset)
            rgb_out_ch = self.rgb.out_ch[2]
        elif rgb_type.startswith("convnext"):
            size       = rgb_type.split("_")[1]        # "tiny" or "base"
            self.rgb   = RGBConvNeXt(size=size, dataset=dataset)
            rgb_out_ch = self.rgb.out_ch[2]
        elif rgb_type.startswith("effnet"):
            size       = rgb_type.split("_")[1]        # "b0" or "b4"
            self.rgb   = RGBEfficientNet(size=size, dataset=dataset)
            rgb_out_ch = self.rgb.out_ch[2]
        else:
            raise ValueError(f"Unknown rgb_type '{rgb_type}'")

        # shape stream always outputs 256ch at s3
        fusion_in_ch = rgb_out_ch + 256

        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_ch, 320, kernel_size=1),
            nn.BatchNorm2d(320), nn.ReLU(),
            nn.Conv2d(320, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False) if x.shape[2] > 64 else x
        _, _, s3 = self.shape(x_shape)
        _, _, r3 = self.rgb(x)
        # align shape spatial size to RGB — works for any backbone/resolution
        s3 = F.interpolate(s3, size=r3.shape[2:], mode="bilinear", align_corners=False)
        return self.head(self.fusion(torch.cat([r3, s3], dim=1)))


# ─────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────

MODEL_NAMES = [
    # ── Baselines (no shape stream) ──────────────────
    "baseline_res18",             # CIFAR + ImageNet
    "baseline_res50",             # CIFAR + ImageNet
    "baseline_res34",             # CIFAR + ImageNet
    "baseline_res101",            # CIFAR + ImageNet
    "baseline_convnext_tiny",     # ImageNet only
    "baseline_convnext_base",     # ImageNet only
    "baseline_efficientnet_b0",   # ImageNet only
    "baseline_efficientnet_b4",   # ImageNet only
    # ── ShapeBiasNet variants ─────────────────────────
    "shape_custom",               # CIFAR only  (lightweight custom backbone)
    "shape_res18",                # CIFAR + ImageNet
    "shape_res34",                # CIFAR + ImageNet
    "shape_res50",                # CIFAR + ImageNet
    "shape_res101",               # CIFAR + ImageNet
    "shape_convnext_tiny",        # ImageNet only
    "shape_convnext_base",        # ImageNet only
    "shape_effnet_b0",            # ImageNet only
    "shape_effnet_b4",            # ImageNet only
]


def build_model(name: str,
                num_classes: int,
                dataset: str = "cifar10") -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        name        : One of MODEL_NAMES.
        num_classes : 10 (CIFAR-10), 100 (CIFAR-100), 1000 (ImageNet).
        dataset     : "cifar10" | "cifar100" | "imagenet"

    Returns:
        nn.Module, randomly initialised.

    Raises:
        ValueError for unrecognised names.
    """
    dataset = dataset.lower()
    kw      = dict(num_classes=num_classes, dataset=dataset)

    # ── Baselines ─────────────────────────────────────────────
    if name == "baseline_res18":          return BaselineResNet18(**kw)
    if name == "baseline_res50":          return BaselineResNet50(**kw)
    if name == "baseline_res34":          return BaselineResNet34(**kw)
    if name == "baseline_res101":         return BaselineResNet101(**kw)
    if name == "baseline_convnext_tiny":  return BaselineConvNeXt("tiny", **kw)
    if name == "baseline_convnext_base":  return BaselineConvNeXt("base", **kw)
    if name == "baseline_efficientnet_b0": return BaselineEfficientNet("b0", **kw)
    if name == "baseline_efficientnet_b4": return BaselineEfficientNet("b4", **kw)

    # ── ShapeBiasNet ──────────────────────────────────────────
    if name == "shape_custom":        return ShapeBiasNet("custom",       **kw)
    if name == "shape_res18":         return ShapeBiasNet("18",           **kw)
    if name == "shape_res34":         return ShapeBiasNet("34",           **kw)
    if name == "shape_res50":         return ShapeBiasNet("50",           **kw)
    if name == "shape_res101":        return ShapeBiasNet("101",          **kw)
    if name == "shape_convnext_tiny": return ShapeBiasNet("convnext_tiny",**kw)
    if name == "shape_convnext_base": return ShapeBiasNet("convnext_base",**kw)
    if name == "shape_effnet_b0":     return ShapeBiasNet("effnet_b0",    **kw)
    if name == "shape_effnet_b4":     return ShapeBiasNet("effnet_b4",    **kw)

    raise ValueError(f"Unknown model '{name}'. Choose from: {MODEL_NAMES}")