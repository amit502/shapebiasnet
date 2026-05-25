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
        gray = x.mean(dim=1, keepdim=True)
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


# ─────────────────────────────────────────────────────────────
#  ROBUSTCONV  —  drop-in Conv2d with isotropic Laplacian diffusion
#  (ablation baseline; see PMDiffusionConv below for the full version)
# ─────────────────────────────────────────────────────────────

class RobustConv(nn.Module):
    """
    Drop-in Conv2d with one step of isotropic Laplacian (heat) diffusion.
    Kept as ablation baseline to compare against anisotropic PMDiffusionConv.

    Limitation: isotropic — smooths edges as well as noise. The network can
    partially compensate during training via linear weight adjustment.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, groups: int = 1,
                 bias: bool = False, lam: float = 0.12):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, groups=groups, bias=bias)
        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer('lap', lap.view(1, 1, 3, 3))
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lap_x = F.conv2d(x, self.lap.expand(x.size(1), 1, 3, 3),
                         padding=1, groups=x.size(1))
        return self.conv(x + self.lam * lap_x)


# ─────────────────────────────────────────────────────────────
#  PMDIFFUSIONCONV  —  Perona-Malik anisotropic diffusion Conv
# ─────────────────────────────────────────────────────────────

class PMDiffusionConv(nn.Module):
    """
    Drop-in Conv2d replacement with Perona-Malik anisotropic diffusion.

    Before every spatial convolution, applies n_steps of discrete PM diffusion:

        For each direction d ∈ {N, S, E, W}:
            ∇_d x  = x[neighbor_d] − x               (directional difference)
            c_d    = exp( −(∇_d x / k)² )             (Leclerc conductance)
        x ← x + λ · Σ_d  c_d · ∇_d x

    Conductance behaviour:
        flat region  (|∇_d x| ≪ k)  →  c_d ≈ 1  →  full diffusion  → noise removed
        edge region  (|∇_d x| ≫ k)  →  c_d ≈ 0  →  no diffusion    → edge preserved

    Critical differences from isotropic Laplacian (RobustConv):
        • Data-dependent: conductance is computed from the actual input gradient,
          not a fixed kernel — the network cannot undo it with linear weights.
        • Truly edge-preserving: boundaries carry zero diffusion flux by design.
        • Nonlinear: the exp conductance cannot be absorbed into the subsequent
          conv's weight matrix, making the bias structurally permanent.

    Hyperparameters (fixed, zero learnable parameters added):
        k       : conductance threshold.  |∇| < k → diffuse;  |∇| > k → preserve.
                  Default 0.3 for BN-normalised features (std ≈ 1).
        lam     : step size.  Must satisfy lam ≤ 0.25 for PDE stability.
        n_steps : diffusion iterations before the conv.  Default 1 — the prior
                  is applied across all layers, giving cumulative robustness
                  without the overhead of multiple steps per layer.

    Conductance: Lorentzian  c(d) = k² / (k² + d²)
        Same edge-preserving asymptotic as Leclerc exp form but computed
        with multiply-add only — no exp() calls, ~3× faster on GPU.
        Combined flux:  c(d)·d  =  k²·d / (k² + d²)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, groups: int = 1,
                 bias: bool = False,
                 n_steps: int = 1, lam: float = 0.12, k: float = 0.3):
        super().__init__()
        self.conv    = nn.Conv2d(in_channels, out_channels, kernel_size,
                                 stride, padding, groups=groups, bias=bias)
        self.n_steps = n_steps
        self.lam     = lam
        self.k2      = k * k   # store k² — only value used in forward

    def _pm_step(self, x: torch.Tensor) -> torch.Tensor:
        xp = F.pad(x, (1, 1, 1, 1), mode='reflect')
        dn = xp[:, :,  :-2, 1:-1] - x   # north neighbour − centre
        ds = xp[:, :, 2:,   1:-1] - x   # south
        de = xp[:, :, 1:-1, 2:]   - x   # east
        dw = xp[:, :, 1:-1,  :-2] - x   # west
        k2 = self.k2
        # Lorentzian flux: c(d)·d = k²·d / (k²+d²) — no exp, pure arithmetic
        return x + self.lam * (
            k2 * dn / (k2 + dn*dn) +
            k2 * ds / (k2 + ds*ds) +
            k2 * de / (k2 + de*de) +
            k2 * dw / (k2 + dw*dw)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            x = self._pm_step(x)
        return self.conv(x)


def _replace_spatial_convs(model: nn.Module, replacement_fn) -> nn.Module:
    """Walk module tree; replace every spatial Conv2d (kernel ≥ 3) via replacement_fn."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and module.kernel_size[0] >= 3:
            new_mod = replacement_fn(module)
            new_mod.conv.weight = module.weight
            if module.bias is not None:
                new_mod.conv.bias = module.bias
            setattr(model, name, new_mod)
        else:
            _replace_spatial_convs(module, replacement_fn)
    return model


def make_robust(model: nn.Module, lam: float = 0.12) -> nn.Module:
    """Replace every spatial Conv2d with RobustConv (isotropic Laplacian, ablation)."""
    def _make(m):
        return RobustConv(m.in_channels, m.out_channels,
                          m.kernel_size[0], m.stride[0], m.padding[0],
                          m.groups, m.bias is not None, lam=lam)
    return _replace_spatial_convs(model, _make)


def make_pm_robust(model: nn.Module,
                   n_steps: int = 2, lam: float = 0.12, k: float = 0.3) -> nn.Module:
    """Replace every spatial Conv2d with PMDiffusionConv (anisotropic PM diffusion)."""
    def _make(m):
        return PMDiffusionConv(m.in_channels, m.out_channels,
                               m.kernel_size[0], m.stride[0], m.padding[0],
                               m.groups, m.bias is not None,
                               n_steps=n_steps, lam=lam, k=k)
    return _replace_spatial_convs(model, _make)


# ─────────────────────────────────────────────────────────────
#  PMDIFFUSIONCONVA  —  adaptive k + shared conductance
# ─────────────────────────────────────────────────────────────

class PMDiffusionConvA(nn.Module):
    """
    Adaptive PMDiffusionConv: shared conductance + learnable per-layer k.

    Two improvements over PMDiffusionConv (pmconv_l):

    1. Shared conductance: gradient magnitude is averaged across channels,
       producing one (B, 1, H, W) edge map shared by all channels. This
       replaces C per-channel divisions with a single division — reducing
       the most expensive PM operation by ~4C×. More principled too: edges
       are structural properties of the feature field, not per-channel
       accidents.

    2. Learnable k (log-parameterized, 1 scalar per layer): each layer
       independently learns its optimal conductance threshold. Shallow layers
       tend to keep k small (fine edge preservation); deeper layers may grow k,
       allowing semantic blurring of block artifacts (pixelation) that appear
       as "edges" at the pixel level but are not meaningful boundaries in deep
       feature space.

    Overhead: ~1 scalar (log_k2) added per wrapped conv. The shared conductance
    substantially reduces the per-step FLOPs vs PMDiffusionConv.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, groups: int = 1,
                 bias: bool = False,
                 n_steps: int = 1, lam: float = 0.12, k: float = 0.3):
        super().__init__()
        self.conv    = nn.Conv2d(in_channels, out_channels, kernel_size,
                                 stride, padding, groups=groups, bias=bias)
        self.n_steps = n_steps
        self.lam     = lam
        self.log_k2  = nn.Parameter(torch.tensor(math.log(k * k)))

    def _pm_step(self, x: torch.Tensor) -> torch.Tensor:
        xp   = F.pad(x, (1, 1, 1, 1), mode='reflect')
        dn   = xp[:, :,  :-2, 1:-1] - x
        ds   = xp[:, :, 2:,   1:-1] - x
        de   = xp[:, :, 1:-1, 2:]   - x
        dw   = xp[:, :, 1:-1,  :-2] - x
        k2   = self.log_k2.exp()
        # shared edge map: mean squared directional gradient across channels
        # one division replaces C per-channel Lorentzian divisions
        mag2 = (dn.pow(2) + ds.pow(2) + de.pow(2) + dw.pow(2)).mean(dim=1, keepdim=True)
        c    = k2 / (k2 + mag2)                                   # (B, 1, H, W)
        return x + self.lam * c * (dn + ds + de + dw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            x = self._pm_step(x)
        return self.conv(x)


def make_pm_adaptive(model: nn.Module,
                     n_steps: int = 1, lam: float = 0.12, k: float = 0.3) -> nn.Module:
    """Replace every spatial Conv2d with PMDiffusionConvA (adaptive k, shared conductance)."""
    def _make(m):
        return PMDiffusionConvA(m.in_channels, m.out_channels,
                                m.kernel_size[0], m.stride[0], m.padding[0],
                                m.groups, m.bias is not None,
                                n_steps=n_steps, lam=lam, k=k)
    return _replace_spatial_convs(model, _make)


# ─────────────────────────────────────────────────────────────
#  COMPLEX MODULUS CONVOLUTION  (complex-modulus-conv branch)
#
#  Replaces every Conv2d with a complex-valued convolution followed
#  by the modulus (amplitude) nonlinearity:
#
#    Standard:  y = NonLin(BN(W * x))
#    CMConv:    y = NonLin(BN( sqrt((W_r*x)² + (W_i*x)²) ))
#
#  For real input x, the complex conv produces:
#    z_r = W_r * x   (real part)
#    z_i = W_i * x   (imaginary part)
#    |z| = sqrt(z_r² + z_i²)   (modulus — always non-negative)
#
#  The modulus is the key operation:
#    • Measures response ENERGY, independent of phase.
#    • Phase in feature maps is extremely sensitive to spatial shifts
#      and corruptions; amplitude is much more stable.
#    • Provably stable under small deformations (Mallat 2012,
#      scattering transform theory): ||F(x) - F(x+δ)|| ≤ C·||δ||
#    • Learned filters generalise fixed-wavelet scattering to arbitrary
#      deep architectures.
#
#  Universality: replaces nn.Conv2d — works on ResNet, ConvNeXt,
#  EfficientNet. Unlike FDN/SDN (BN replacement, no-op on ConvNeXt's
#  LayerNorm), CMConv applies to every architecture identically.
#
#  Cost: 2× parameters and ~2× FLOPs per conv layer.
#  Initialization: W_r ~ Kaiming, W_i ~ N(0, 0.01) so training
#  starts close to a standard conv and the imaginary part grows
#  only where useful.
# ─────────────────────────────────────────────────────────────

class CMConv2d(nn.Module):
    """Complex Modulus Conv2d — same parameter count as nn.Conv2d.

    Uses a single weight W as the real filter; rot90(W) serves as the imaginary
    (quadrature) filter. Output = sqrt((W*x)² + (rot90(W)*x)² + ε).

    For oriented filters (edges/textures), rot90 produces the perpendicular-direction
    quadrature partner — the standard complex-wavelet construction. The modulus pools
    over both orientations, giving a response stable under local deformations
    (Mallat 2012 stability bound) while using zero extra parameters vs standard Conv2d.

    Requires square kernels (kH == kW), which holds for all spatial convolutions in
    ResNet (3×3), ConvNeXt (7×7), and EfficientNet (3×3 / 5×5 depthwise).
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias: bool = False):
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self._stride   = stride   if isinstance(stride,   tuple) else (stride,   stride)
        self._padding  = padding  if isinstance(padding,  tuple) else (padding,  padding)
        self._dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self._groups   = groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, ks[0], ks[1])
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        # Stack W and rot90(W) into a single 2×C_out weight tensor so cuDNN
        # runs one GEMM instead of two — roughly the same FLOPs but one kernel
        # launch, better parallelism, ~10-20% faster than two separate conv calls.
        w_rot = torch.rot90(w, k=1, dims=[-2, -1])
        w_cat = torch.cat([w, w_rot], dim=0)           # (2*C_out, C_in, k, k)
        b_cat = (torch.cat([self.bias, self.bias], dim=0)
                 if self.bias is not None else None)
        z = F.conv2d(x, w_cat, b_cat, self._stride, self._padding,
                     self._dilation, self._groups)
        z_r, z_i = z.chunk(2, dim=1)
        return torch.sqrt(z_r.pow(2) + z_i.pow(2) + 1e-8)


def make_cmconv(model: nn.Module) -> nn.Module:
    """Replace square spatial (k>1, kH==kW) Conv2d layers with CMConv2d.
    Pointwise (1×1) and non-square convs remain standard Conv2d.
    Parameter count after replacement equals the baseline model exactly.
    """
    for name, module in model.named_children():
        k = module.kernel_size if isinstance(module, nn.Conv2d) else None
        if (isinstance(module, nn.Conv2d) and
                k[0] > 1 and k[0] == k[1]):
            cm = CMConv2d(
                module.in_channels, module.out_channels,
                kernel_size=k[0],
                stride=module.stride[0],
                padding=module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                dilation=module.dilation[0],
                groups=module.groups,
                bias=module.bias is not None,
            )
            setattr(model, name, cm)
        else:
            make_cmconv(module)
    return model


# ── CMConv backbone wrappers ───────────────────────────────────

class CMConvResNet(nn.Module):
    """ResNet with every Conv2d replaced by CMConv2d + modulus."""
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        make_cmconv(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CMConvConvNeXt(nn.Module):
    """ConvNeXt with every Conv2d replaced by CMConv2d + modulus."""
    def __init__(self, size: str = "tiny", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        builders = {"tiny": convnext_tiny, "base": convnext_base}
        assert size in builders
        net = builders[size](weights=None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        make_cmconv(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CMConvEfficientNet(nn.Module):
    """EfficientNet with every Conv2d replaced by CMConv2d + modulus."""
    def __init__(self, size: str = "b0", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_cmconv(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── CMConv half-plane (|W*x|) ─────────────────────────────────

class CMConvAbs2d(nn.Module):
    """Half-plane complex modulus — Conv2d with absolute-value output.

    Output = |W*x|. Symmetric activation: invariant to sign flip of the
    filter response. Same parameters and compute as a standard Conv2d.
    Weaker invariance than full quadrature (180° phase symmetry only),
    but zero overhead — useful as a lower bound in ablation.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(self.conv(x))


def make_cmconv_abs(model: nn.Module) -> nn.Module:
    """Replace square spatial (k>1, kH==kW) Conv2d layers with CMConvAbs2d."""
    for name, module in model.named_children():
        k = module.kernel_size if isinstance(module, nn.Conv2d) else None
        if (isinstance(module, nn.Conv2d) and
                k[0] > 1 and k[0] == k[1]):
            ca = CMConvAbs2d(
                module.in_channels, module.out_channels,
                kernel_size=k[0],
                stride=module.stride[0],
                padding=module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                dilation=module.dilation[0],
                groups=module.groups,
                bias=module.bias is not None,
            )
            setattr(model, name, ca)
        else:
            make_cmconv_abs(module)
    return model


class CMConvAbsResNet(nn.Module):
    """ResNet with spatial Conv2d replaced by CMConvAbs2d (half-plane)."""
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        make_cmconv_abs(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CMConvAbsConvNeXt(nn.Module):
    """ConvNeXt with spatial Conv2d replaced by CMConvAbs2d (half-plane)."""
    def __init__(self, size: str = "tiny", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        builders = {"tiny": convnext_tiny, "base": convnext_base}
        assert size in builders
        net = builders[size](weights=None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        make_cmconv_abs(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CMConvAbsEfficientNet(nn.Module):
    """EfficientNet with spatial Conv2d replaced by CMConvAbs2d (half-plane)."""
    def __init__(self, size: str = "b0", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_cmconv_abs(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────
#  SPATIAL VARIANCE SUPPRESSION (VarSupp)
#
#  Zero-parameter texture debiasing module inserted at stage boundaries.
#
#  Key insight: texture-sensitive channels have HIGH spatial variance
#  (the texture pattern fires across the entire feature map), while
#  structure-sensitive channels have LOW spatial variance (an "ear detector"
#  fires in one place). Reweighting channels by inverse spatial variance
#  automatically suppresses texture and amplifies structure.
#
#  Self-calibrating: corruptions increase spatial variance of texture
#  channels further → stronger suppression → more shape-based classification.
#
#  Inserted once per residual stage (3–4 times per network). Zero learned
#  parameters. Compute: one var() + one multiply per channel per stage.
# ─────────────────────────────────────────────────────────────

class VarSuppression(nn.Module):
    """Inverse spatial-variance channel attention — zero parameters.

    w_c = 1 / (1 + σ²_c / mean(σ²))   where σ²_c = var(F_c over H×W)

    Weights are renormalized so their mean is 1 (preserves feature magnitude).
    Skipped on feature maps ≤ 2×2 (too few spatial points for stable variance).
    """
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-1] <= 2:
            return x
        var = x.var(dim=[-2, -1], keepdim=True, unbiased=False)        # (B,C,1,1)
        mean_var = var.mean(dim=1, keepdim=True).clamp(min=self.eps)
        w = 1.0 / (1.0 + var / mean_var)
        w = w * (x.shape[1] / w.sum(dim=1, keepdim=True).clamp(min=self.eps))
        return x * w


class VarSuppResNet(nn.Module):
    """ResNet with VarSuppression inserted after each residual stage."""
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        net.layer1 = nn.Sequential(net.layer1, VarSuppression())
        net.layer2 = nn.Sequential(net.layer2, VarSuppression())
        net.layer3 = nn.Sequential(net.layer3, VarSuppression())
        net.layer4 = nn.Sequential(net.layer4, VarSuppression())
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VarSuppConvNeXt(nn.Module):
    """ConvNeXt with VarSuppression inserted after each stage."""
    def __init__(self, size: str = "tiny", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        builders = {"tiny": convnext_tiny, "base": convnext_base}
        assert size in builders
        net = builders[size](weights=None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        # torchvision ConvNeXt: features[1,3,5,7] are the four stages
        for i in [1, 3, 5, 7]:
            net.features[i] = nn.Sequential(net.features[i], VarSuppression())
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VarSuppEfficientNet(nn.Module):
    """EfficientNet with VarSuppression inserted after each MBConv stage."""
    def __init__(self, size: str = "b0", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        # torchvision EfficientNet: features[1..7] are the MBConv stages
        for i in range(1, 8):
            net.features[i] = nn.Sequential(net.features[i], VarSuppression())
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────
#  FREQUENCY-DECOUPLED NORMALIZATION (FDN)
#
#  Drop-in replacement for BatchNorm2d. Decomposes x into a
#  low-frequency component (local AvgPool) and high-frequency
#  residual, then applies BN to LF and IN to HF.
#
#  BN: batch-level normalisation → stable for semantic (LF) content.
#  IN: instance-level normalisation → removes corruption-induced (HF) variance.
#
#  Applied to ALL channels at ALL BN layers — unlike IBN-Net which
#  selects channels heuristically in early layers only.
#
#  Closest prior work: IBN-Net (Pan et al., ECCV 2018).
#  FDN differs: spatial-frequency split via AvgPool vs channel split.
# ─────────────────────────────────────────────────────────────

class FDN(nn.Module):
    """
    Frequency-Decoupled Normalization: drop-in BatchNorm2d replacement.

        x_lf = AvgPool2d(x, k=3)     # low-freq  (local mean)
        x_hf = x − x_lf              # high-freq (local residual)
        out  = BN(x_lf) + IN(x_hf)

    Corruptions (noise, JPEG, blur artifacts) predominantly shift
    the HF statistics. IN removes this instance-specific shift.
    BN preserves the inter-class semantic structure in LF.
    """
    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.bn  = nn.BatchNorm2d(num_features, eps=eps,
                                  momentum=momentum, affine=affine)
        self.in_ = nn.InstanceNorm2d(num_features, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] <= 4:
            return self.bn(x)
        x_lf = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_hf = x - x_lf
        return self.bn(x_lf) + self.in_(x_hf)


def make_fdn(model: nn.Module) -> nn.Module:
    """Replace every BatchNorm2d in model with FDN (in-place, recursive)."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            fdn = FDN(module.num_features, module.eps,
                      module.momentum, module.affine)
            fdn.bn.running_mean.copy_(module.running_mean)
            fdn.bn.running_var.copy_(module.running_var)
            if module.affine:
                fdn.bn.weight.data.copy_(module.weight.data)
                fdn.bn.bias.data.copy_(module.bias.data)
            setattr(model, name, fdn)
        else:
            make_fdn(module)
    return model


# ─────────────────────────────────────────────────────────────
#  SPECTRAL DECOUPLED NORMALIZATION (SDN)
#
#  Drop-in BatchNorm2d replacement. Same idea as FDN but uses an
#  exact Fourier decomposition instead of AvgPool approximation:
#
#    X     = fft2(x)
#    mask  = Gaussian(H, W, σ)      — learnable per-layer bandwidth
#    x_lf  = ifft2(X · mask).real   — low-freq  (shape / structure)
#    x_hf  = ifft2(X · (1−mask)).real — high-freq (texture / noise)
#    out   = BN(x_lf) + IN(x_hf)
#
#  Why better than FDN:
#    FDN uses AvgPool(k=3) for LF/HF split. On small feature maps
#    (CIFAR layer3 = 8×8, layer4 = 4×4) the 3×3 kernel covers most
#    of the map → x_lf ≈ mean(x), x_hf ≈ 0. FDN is a no-op there.
#    SDN's FFT decomposition is exact at any resolution — 4×4 has
#    16 genuine frequency components, all properly separated.
#
#  Learnable σ (log-parameterized, one scalar per layer):
#    Shallow layers tend to keep σ large (preserve fine detail in LF).
#    Deep layers may shrink σ (only coarsest structure in LF).
#    Init: σ = 0.25 — similar bandwidth to FDN's AvgPool(k=3).
#
#  Overhead: 2 FFT calls + Gaussian mask per BN layer. Mask is
#  recomputed each forward pass but is O(H·W) — negligible vs conv.
# ─────────────────────────────────────────────────────────────

class SDN(nn.Module):
    """
    Spectral Decoupled Normalization: drop-in BatchNorm2d replacement.

        X     = fft2(x)
        mask  = exp(−(u²+v²) / 2σ²)   Gaussian low-pass, σ learnable
        x_lf  = ifft2(X · mask).real
        x_hf  = ifft2(X · (1−mask)).real
        out   = BN(x_lf) + IN(x_hf)

    Works at any resolution including CIFAR's 4×4 feature maps where
    FDN's AvgPool degenerates. σ is log-parameterized per layer so
    each BN site learns its own LF/HF frequency boundary.
    """
    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.bn       = nn.BatchNorm2d(num_features, eps=eps,
                                       momentum=momentum, affine=affine)
        self.in_      = nn.InstanceNorm2d(num_features, eps=eps, affine=affine)
        # log σ init → σ ≈ 0.25, similar cutoff to FDN's AvgPool k=3
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.25)))

    def _mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        fh = torch.fft.fftfreq(H, device=device)   # (H,)
        fw = torch.fft.fftfreq(W, device=device)   # (W,)
        gh, gw = torch.meshgrid(fh, fw, indexing='ij')   # (H, W)
        sigma  = self.log_sigma.exp()
        return torch.exp(-(gh ** 2 + gw ** 2) / (2 * sigma ** 2))  # (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W  = x.shape[2], x.shape[3]
        X     = torch.fft.fft2(x)                        # (B, C, H, W) complex
        mask  = self._mask(H, W, x.device)               # (H, W) real
        x_lf  = torch.fft.ifft2(X * mask).real
        x_hf  = torch.fft.ifft2(X * (1.0 - mask)).real
        return self.bn(x_lf) + self.in_(x_hf)


def make_sdn(model: nn.Module) -> nn.Module:
    """Replace every BatchNorm2d in model with SDN (in-place, recursive)."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            sdn = SDN(module.num_features, module.eps,
                      module.momentum, module.affine)
            sdn.bn.running_mean.copy_(module.running_mean)
            sdn.bn.running_var.copy_(module.running_var)
            if module.affine:
                sdn.bn.weight.data.copy_(module.weight.data)
                sdn.bn.bias.data.copy_(module.bias.data)
            setattr(model, name, sdn)
        else:
            make_sdn(module)
    return model


# ─────────────────────────────────────────────────────────────
#  BACKBONE WRAPPERS  (SDN — freq-norm branch)
# ─────────────────────────────────────────────────────────────

class SDNResNet(nn.Module):
    """ResNet with all BatchNorm2d replaced by SDN (exact FFT freq split)."""
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        make_sdn(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SDNGISCResNet(nn.Module):
    """ResNet with SDN on all BN layers and Gated INSC on all skip connections."""
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        make_sdn(net)
        make_ginsc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SDNGISCConvNeXt(nn.Module):
    """ConvNeXt with Gated INSC on all skip connections. SDN is a no-op (no BN2d)."""
    def __init__(self, size: str = "tiny", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        builders = {"tiny": convnext_tiny, "base": convnext_base}
        assert size in builders
        net = builders[size](weights=None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        make_ginsc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SDNEfficientNet(nn.Module):
    """EfficientNet with all BatchNorm2d replaced by SDN."""
    def __init__(self, size: str = "b0", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_sdn(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SDNGISCEfficientNet(nn.Module):
    """EfficientNet with SDN on BN layers and Gated INSC on MBConv skips."""
    def __init__(self, size: str = "b0", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_sdn(net)
        make_ginsc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────
#  INSTANCE-NORMALIZED SKIP CONNECTIONS (INSC)
#
#  Patches every residual block to apply IN to the shortcut
#  before the residual add:
#
#    Standard:  out = F(x) + x
#    INSC:      out = F(x) + IN(x)
#
#  Skip connections are a "corruption highway" — they carry raw
#  corrupted activations directly to deep layers, bypassing all
#  learned processing inside the block. IN removes instance-level
#  texture/style variance from the skip.
#
#  Predictive-coding framing: skip = top-down prior (should be
#  style-invariant); F(x) = prediction error (discriminative update).
#  IN enforces style-invariance on the prior.
#
#  Closest prior work: ResNorm (Kim et al., DCASE 2021) uses
#  λ·x + FreqIN(x) for audio spectrograms. INSC applies spatial
#  IN to image CNN residual blocks for corruption robustness.
#
#  Supports: ResNet (BasicBlock/Bottleneck), ConvNeXt (CNBlock),
#            EfficientNet (MBConv/FusedMBConv with use_res_connect).
# ─────────────────────────────────────────────────────────────

def _patch_basicblock_insc(block) -> None:
    out_ch = block.conv2.out_channels
    block.in_skip = nn.InstanceNorm2d(out_ch, affine=True)

    def forward(x):
        out = block.conv1(x)
        out = block.bn1(out)
        out = block.relu(out)
        out = block.conv2(out)
        out = block.bn2(out)
        identity = block.downsample(x) if block.downsample is not None else x
        out = out + block.in_skip(identity)
        return block.relu(out)

    block.forward = forward


def _patch_bottleneck_insc(block) -> None:
    out_ch = block.conv3.out_channels
    block.in_skip = nn.InstanceNorm2d(out_ch, affine=True)

    def forward(x):
        out = block.conv1(x)
        out = block.bn1(out)
        out = block.relu(out)
        out = block.conv2(out)
        out = block.bn2(out)
        out = block.relu(out)
        out = block.conv3(out)
        out = block.bn3(out)
        identity = block.downsample(x) if block.downsample is not None else x
        out = out + block.in_skip(identity)
        return block.relu(out)

    block.forward = forward


def _patch_cnblock_insc(block) -> None:
    out_ch = block.layer_scale.shape[0]
    block.in_skip = nn.InstanceNorm2d(out_ch, affine=True)

    def forward(x):
        result = block.layer_scale * block.block(x)
        result = block.stochastic_depth(result)
        return result + block.in_skip(x)

    block.forward = forward


def _patch_mbconv_insc(block) -> None:
    in_ch = None
    for m in block.block.modules():
        if isinstance(m, nn.Conv2d):
            in_ch = m.in_channels
            break
    if in_ch is None:
        return
    block.in_skip = nn.InstanceNorm2d(in_ch, affine=True)

    def forward(x):
        result = block.block(x)
        if block.use_res_connect:
            result = block.stochastic_depth(result)
            result = result + block.in_skip(x)
        return result

    block.forward = forward


def make_insc(model: nn.Module) -> nn.Module:
    """
    Patch all residual/skip blocks to apply IN to shortcut before residual add.
    Handles ResNet, ConvNeXt, EfficientNet generically via class-name dispatch.
    """
    for module in model.modules():
        cls = type(module).__name__
        if cls == 'BasicBlock':
            _patch_basicblock_insc(module)
        elif cls == 'Bottleneck':
            _patch_bottleneck_insc(module)
        elif cls == 'CNBlock':
            _patch_cnblock_insc(module)
        elif cls in ('MBConv', 'FusedMBConv') and getattr(module, 'use_res_connect', False):
            _patch_mbconv_insc(module)
    return model


# ── Gated INSC ────────────────────────────────────────────────
# Learnable per-block gate: out = (1-g)·identity + g·IN(identity)
# Gate initialized to sigmoid(-3)≈0.047 so training starts as standard ResNet.
# The network learns where texture stripping is beneficial vs harmful.

class GatedINSC(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.in_skip = nn.InstanceNorm2d(num_features, affine=True)
        self.gate    = nn.Parameter(torch.full((1,), -3.0))

    def forward(self, identity: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate)
        return (1.0 - g) * identity + g * self.in_skip(identity)


def _patch_basicblock_ginsc(block) -> None:
    out_ch = block.conv2.out_channels
    block.ginsc = GatedINSC(out_ch)

    def forward(x):
        out = block.conv1(x)
        out = block.bn1(out)
        out = block.relu(out)
        out = block.conv2(out)
        out = block.bn2(out)
        identity = block.downsample(x) if block.downsample is not None else x
        out = out + block.ginsc(identity)
        return block.relu(out)

    block.forward = forward


def _patch_bottleneck_ginsc(block) -> None:
    out_ch = block.conv3.out_channels
    block.ginsc = GatedINSC(out_ch)

    def forward(x):
        out = block.conv1(x)
        out = block.bn1(out)
        out = block.relu(out)
        out = block.conv2(out)
        out = block.bn2(out)
        out = block.relu(out)
        out = block.conv3(out)
        out = block.bn3(out)
        identity = block.downsample(x) if block.downsample is not None else x
        out = out + block.ginsc(identity)
        return block.relu(out)

    block.forward = forward


def _patch_cnblock_ginsc(block) -> None:
    out_ch = block.layer_scale.shape[0]
    block.ginsc = GatedINSC(out_ch)

    def forward(x):
        result = block.layer_scale * block.block(x)
        result = block.stochastic_depth(result)
        return result + block.ginsc(x)

    block.forward = forward


def _patch_mbconv_ginsc(block) -> None:
    in_ch = None
    for m in block.block.modules():
        if isinstance(m, nn.Conv2d):
            in_ch = m.in_channels
            break
    if in_ch is None:
        return
    block.ginsc = GatedINSC(in_ch)

    def forward(x):
        result = block.block(x)
        if block.use_res_connect:
            result = block.stochastic_depth(result)
            result = result + block.ginsc(x)
        return result

    block.forward = forward


def make_ginsc(model: nn.Module) -> nn.Module:
    """
    Patch all residual/skip blocks with a learnable gated IN on the shortcut.
    Architecture-agnostic: handles ResNet, ConvNeXt, EfficientNet.
    """
    for module in model.modules():
        cls = type(module).__name__
        if cls == 'BasicBlock':
            _patch_basicblock_ginsc(module)
        elif cls == 'Bottleneck':
            _patch_bottleneck_ginsc(module)
        elif cls == 'CNBlock':
            _patch_cnblock_ginsc(module)
        elif cls in ('MBConv', 'FusedMBConv') and getattr(module, 'use_res_connect', False):
            _patch_mbconv_ginsc(module)
    return model


# ─────────────────────────────────────────────────────────────
#  BACKBONE WRAPPERS  (FDN / INSC / FDN+INSC — freq-norm branch)
# ─────────────────────────────────────────────────────────────

class FDNResNet(nn.Module):
    """ResNet with all BatchNorm2d replaced by FDN."""
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        make_fdn(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ISCResNet(nn.Module):
    """ResNet with IN applied to every residual skip before the add."""
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        make_insc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FDNISCResNet(nn.Module):
    """ResNet with FDN on all BN layers and INSC on all skip connections."""
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        make_fdn(net)
        make_insc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FDNConvNeXt(nn.Module):
    """ConvNeXt with FDN. ConvNeXt uses LayerNorm (no BN), so FDN is a no-op —
    kept for ablation completeness. Use ISCConvNeXt for the meaningful change."""
    def __init__(self, size: str = "tiny", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        nets = {"tiny": convnext_tiny, "base": convnext_base}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_fdn(net)   # no-op: ConvNeXt has no BatchNorm2d
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ISCConvNeXt(nn.Module):
    """ConvNeXt with IN applied to every CNBlock skip connection."""
    def __init__(self, size: str = "tiny", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        nets = {"tiny": convnext_tiny, "base": convnext_base}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_insc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FDNISCConvNeXt(nn.Module):
    """ConvNeXt with FDN + INSC (effectively INSC-only since FDN is no-op)."""
    def __init__(self, size: str = "tiny", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        nets = {"tiny": convnext_tiny, "base": convnext_base}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_fdn(net)
        make_insc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FDNEfficientNet(nn.Module):
    """EfficientNet with all BatchNorm2d replaced by FDN."""
    def __init__(self, size: str = "b4", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_fdn(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ISCEfficientNet(nn.Module):
    """EfficientNet with IN applied to every MBConv skip connection."""
    def __init__(self, size: str = "b4", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_insc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FDNISCEfficientNet(nn.Module):
    """EfficientNet with FDN on BN layers and INSC on MBConv skips."""
    def __init__(self, size: str = "b4", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_fdn(net)
        make_insc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GISCResNet(nn.Module):
    """ResNet with Gated INSC on all skip connections."""
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        make_ginsc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FDNGISCResNet(nn.Module):
    """ResNet with FDN on all BN layers and Gated INSC on all skip connections."""
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        make_fdn(net)
        make_ginsc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FDNGISCConvNeXt(nn.Module):
    """ConvNeXt with Gated INSC on all skip connections. FDN is a no-op (no BN2d)."""
    def __init__(self, size: str = "tiny", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        builders = {"tiny": convnext_tiny, "base": convnext_base}
        assert size in builders
        net = builders[size](weights=None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        make_ginsc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FDNGISCEfficientNet(nn.Module):
    """EfficientNet with FDN on BN layers and Gated INSC on MBConv skips."""
    def __init__(self, size: str = "b0", num_classes: int = 1000,
                 dataset: str = "imagenet"):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_fdn(net)
        make_ginsc(net)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ShapeEncoder(nn.Module):
    """
    Hierarchical shape feature extractor — fully scalable.

    Scaling axes:
      out_ch   : final-stage channel width; earlier stages scale as
                 (out_ch//4, out_ch//2, out_ch). Set by ShapeBiasNet
                 as rgb_out_ch // 4, so the shape stream is always
                 proportional to the backbone width.
      n_blocks : diffusion blocks per stage (b1, b2, b3). Deeper
                 backbones receive more blocks to match their capacity.

    Spatial resolutions (set adaptively in ShapeBiasNet.forward):
        CIFAR    (32×32 input) : 32×32 → 16×16 → 8×8
        ImageNet (56×56 input) : 56×56 → 28×28 → 14×14
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

    def forward_until_l2(self, x: torch.Tensor):
        r1 = self.stage1(x)
        r2 = self.stage2(r1)
        return r1, r2

    def run_l3(self, r2: torch.Tensor) -> torch.Tensor:
        return self.stage3(r2)

    def forward(self, x: torch.Tensor):
        r1, r2 = self.forward_until_l2(x)
        return r1, r2, self.run_l3(r2)


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

    def forward_until_l2(self, x: torch.Tensor):
        r1 = self.l1(self.stem(x))
        r2 = self.l2(r1)
        return r1, r2

    def run_l3(self, r2: torch.Tensor) -> torch.Tensor:
        return self.l3(r2)

    def forward(self, x: torch.Tensor):
        r1, r2 = self.forward_until_l2(x)
        return r1, r2, self.run_l3(r2)


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

    def forward_until_l2(self, x: torch.Tensor):
        r1 = self.stage1(self.stem(x))
        r2 = self.stage2(self.down1(r1))
        return r1, r2

    def run_l3(self, r2: torch.Tensor) -> torch.Tensor:
        return self.stage3(self.down2(r2))

    def forward(self, x: torch.Tensor):
        r1, r2 = self.forward_until_l2(x)
        return r1, r2, self.run_l3(r2)


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

    def forward_until_l2(self, x: torch.Tensor):
        r1 = self.stage1(x)
        r2 = self.stage2(r1)
        return r1, r2

    def run_l3(self, r2: torch.Tensor) -> torch.Tensor:
        return self.stage3(r2)

    def forward(self, x: torch.Tensor):
        r1, r2 = self.forward_until_l2(x)
        return r1, r2, self.run_l3(r2)


# ─────────────────────────────────────────────────────────────
#  BACKBONE WRAPPERS  (RobustConv — isotropic ablation)
# ─────────────────────────────────────────────────────────────

class RobustResNet(nn.Module):
    """
    ResNet-18/34/50/101 with all 3×3 convolutions replaced by RobustConv.

    Compared against BaselineResNet, this isolates the contribution of the
    Laplacian diffusion primitive with zero architectural overhead.
    """
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet", lam: float = 0.12):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets, f"depth must be one of {list(nets.keys())}"
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        make_robust(net, lam=lam)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RobustConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny/Base with all spatial convolutions replaced by RobustConv.
    ConvNeXt's 7×7 depthwise conv is also wrapped — Laplacian is always
    applied depthwise before the spatial aggregation, regardless of kernel size.
    """
    def __init__(self, size: str = "tiny", num_classes: int = 1000,
                 dataset: str = "imagenet", lam: float = 0.12):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        nets = {"tiny": convnext_tiny, "base": convnext_base}
        assert size in nets, f"size must be one of {list(nets.keys())}"
        net = nets[size](weights=None, num_classes=num_classes)
        make_robust(net, lam=lam)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RobustEfficientNet(nn.Module):
    """
    EfficientNet-B0/B4 with all spatial convolutions replaced by RobustConv.
    Depthwise MBConv 3×3/5×5 convolutions are wrapped (groups preserved).
    """
    def __init__(self, size: str = "b4", num_classes: int = 1000,
                 dataset: str = "imagenet", lam: float = 0.12):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets, f"size must be one of {list(nets.keys())}"
        net = nets[size](weights=None, num_classes=num_classes)
        make_robust(net, lam=lam)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────
#  BACKBONE WRAPPERS  (PMDiffusionConv — anisotropic, main model)
# ─────────────────────────────────────────────────────────────

class PMResNet(nn.Module):
    """
    ResNet-18/34/50/101 with every spatial conv replaced by PMDiffusionConv.
    Anisotropic Perona-Malik diffusion — edge-preserving, data-dependent,
    nonlinear. Zero extra parameters vs baseline.
    """
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet", n_steps: int = 1,
                 lam: float = 0.12, k: float = 0.3):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets, f"depth must be one of {list(nets.keys())}"
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        make_pm_robust(net, n_steps=n_steps, lam=lam, k=k)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PMConvNeXt(nn.Module):
    """ConvNeXt-Tiny/Base with every spatial conv replaced by PMDiffusionConv."""
    def __init__(self, size: str = "tiny", num_classes: int = 1000,
                 dataset: str = "imagenet", n_steps: int = 1,
                 lam: float = 0.12, k: float = 0.3):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        nets = {"tiny": convnext_tiny, "base": convnext_base}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_pm_robust(net, n_steps=n_steps, lam=lam, k=k)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PMEfficientNet(nn.Module):
    """EfficientNet-B0/B4 with every spatial conv replaced by PMDiffusionConv."""
    def __init__(self, size: str = "b4", num_classes: int = 1000,
                 dataset: str = "imagenet", n_steps: int = 1,
                 lam: float = 0.12, k: float = 0.3):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_pm_robust(net, n_steps=n_steps, lam=lam, k=k)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────
#  BACKBONE WRAPPERS  (PMDiffusionConvA — adaptive, this branch)
# ─────────────────────────────────────────────────────────────

class PMAdaptiveResNet(nn.Module):
    """
    ResNet-18/34/50/101 with every spatial conv replaced by PMDiffusionConvA.
    Adaptive k + shared conductance vs PMResNet (fixed k, per-channel conductance).
    """
    def __init__(self, depth: str = "50", num_classes: int = 1000,
                 dataset: str = "imagenet", n_steps: int = 1,
                 lam: float = 0.12, k: float = 0.3):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101
        nets = {"18": resnet18, "34": resnet34, "50": resnet50, "101": resnet101}
        assert depth in nets, f"depth must be one of {list(nets.keys())}"
        net = nets[depth](weights=None)
        if "cifar" in dataset:
            net.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        net.fc = nn.Linear(512 if depth in ("18", "34") else 2048, num_classes)
        make_pm_adaptive(net, n_steps=n_steps, lam=lam, k=k)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PMAdaptiveConvNeXt(nn.Module):
    """ConvNeXt-Tiny/Base with every spatial conv replaced by PMDiffusionConvA."""
    def __init__(self, size: str = "tiny", num_classes: int = 1000,
                 dataset: str = "imagenet", n_steps: int = 1,
                 lam: float = 0.12, k: float = 0.3):
        super().__init__()
        from torchvision.models import convnext_tiny, convnext_base
        nets = {"tiny": convnext_tiny, "base": convnext_base}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_pm_adaptive(net, n_steps=n_steps, lam=lam, k=k)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PMAdaptiveEfficientNet(nn.Module):
    """EfficientNet-B0/B4 with every spatial conv replaced by PMDiffusionConvA."""
    def __init__(self, size: str = "b4", num_classes: int = 1000,
                 dataset: str = "imagenet", n_steps: int = 1,
                 lam: float = 0.12, k: float = 0.3):
        super().__init__()
        from torchvision.models import efficientnet_b0, efficientnet_b4
        nets = {"b0": efficientnet_b0, "b4": efficientnet_b4}
        assert size in nets
        net = nets[size](weights=None, num_classes=num_classes)
        make_pm_adaptive(net, n_steps=n_steps, lam=lam, k=k)
        self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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

        if rgb_type == "custom":
            self.rgb   = RGBCustom()
            rgb_out_ch = 256
        elif rgb_type in ("18", "34", "50", "101"):
            self.rgb   = RGBResNet(depth=rgb_type, dataset=dataset)
            rgb_out_ch = self.rgb.out_ch[2]
        elif rgb_type.startswith("convnext"):
            size       = rgb_type.split("_")[1]
            self.rgb   = RGBConvNeXt(size=size, dataset=dataset)
            rgb_out_ch = self.rgb.out_ch[2]
        elif rgb_type.startswith("effnet"):
            size       = rgb_type.split("_")[1]
            self.rgb   = RGBEfficientNet(size=size, dataset=dataset)
            rgb_out_ch = self.rgb.out_ch[2]
        else:
            raise ValueError(f"Unknown rgb_type '{rgb_type}'")

        # ── Shape encoder: width = rgb_out_ch // 4, min 64 ──────────────
        # Depth (n_blocks) scales with backbone so deeper backbones get
        # more diffusion capacity even when channel width is equal
        # (e.g. ResNet-50 vs ResNet-101 both have 1024ch at layer3).
        _NBLOCKS = {
            "custom": (1, 1, 1),
            "18":     (1, 2, 1),
            "34":     (1, 2, 1),
            "50":     (2, 2, 1),
            "101":    (2, 2, 1),
        }
        _SHAPE_CH = {
            "custom": 256,
            "18":     256,
            "34":     256,
            "50":     256,
            "101":    256,
        }
        n_blocks     = _NBLOCKS.get(rgb_type, (2, 2, 1))
        shape_out_ch = _SHAPE_CH.get(rgb_type, max(64, rgb_out_ch // 4))
        self.shape   = ShapeEncoder(out_ch=shape_out_ch, n_blocks=n_blocks)

        # ── Fusion head: late concat r3 + s3 ─────────────────────────────
        fusion_in_ch  = rgb_out_ch + shape_out_ch
        fusion_mid_ch = max(128, fusion_in_ch // 4)
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
        _, _, s3 = self.shape(x)
        _, _, r3 = self.rgb(x)
        s3 = F.interpolate(s3, size=r3.shape[2:], mode="bilinear", align_corners=False)
        return self.head(self.fusion(torch.cat([r3, s3], dim=1)))


# ─────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────

MODEL_NAMES = [
    # ── Baselines (vanilla backbone, no modification) ──────────
    "baseline_res18",             # CIFAR + ImageNet
    "baseline_res34",             # CIFAR + ImageNet
    "baseline_res50",             # CIFAR + ImageNet
    "baseline_res101",            # CIFAR + ImageNet
    "baseline_convnext_tiny",     # ImageNet only
    "baseline_convnext_base",     # ImageNet only
    "baseline_efficientnet_b0",   # ImageNet only
    "baseline_efficientnet_b4",   # ImageNet only
    # ── PMConv-L (Lorentzian conductance, n_steps=1 — pmconv-scale branch) ──
    "pmconv_l_res18",             # CIFAR + ImageNet
    "pmconv_l_res34",             # CIFAR + ImageNet
    "pmconv_l_res50",             # CIFAR + ImageNet
    "pmconv_l_res101",            # CIFAR + ImageNet
    "pmconv_l_convnext_tiny",     # ImageNet only
    "pmconv_l_convnext_base",     # ImageNet only
    "pmconv_l_effnet_b0",         # ImageNet only
    "pmconv_l_effnet_b4",         # ImageNet only
    # ── PMConv-A (shared conductance + learnable k — pmconv-adaptive branch) ──
    "pmconv_a_res18",             # CIFAR + ImageNet
    "pmconv_a_res34",             # CIFAR + ImageNet
    "pmconv_a_res50",             # CIFAR + ImageNet
    "pmconv_a_res101",            # CIFAR + ImageNet
    "pmconv_a_convnext_tiny",     # ImageNet only
    "pmconv_a_convnext_base",     # ImageNet only
    "pmconv_a_effnet_b0",         # ImageNet only
    "pmconv_a_effnet_b4",         # ImageNet only
    # ── RobustConv (isotropic Laplacian — ablation only) ────────────
    "robustconv_res18",           # CIFAR + ImageNet
    "robustconv_res34",           # CIFAR + ImageNet
    "robustconv_res50",           # CIFAR + ImageNet
    "robustconv_res101",          # CIFAR + ImageNet
    "robustconv_convnext_tiny",   # ImageNet only
    "robustconv_convnext_base",   # ImageNet only
    "robustconv_effnet_b0",       # ImageNet only
    "robustconv_effnet_b4",       # ImageNet only
    # ── VarSupp (Spatial Variance Suppression — var-suppression branch) ──
    # Zero-parameter texture debiasing. Inserts inverse-variance channel
    # attention after each residual stage. Works on all architectures.
    "vsupp_res18",              # CIFAR + ImageNet
    "vsupp_res34",              # CIFAR + ImageNet
    "vsupp_res50",              # CIFAR + ImageNet  ★ primary benchmark
    "vsupp_res101",             # CIFAR + ImageNet
    "vsupp_convnext_tiny",      # ImageNet only
    "vsupp_convnext_base",      # ImageNet only
    "vsupp_effnet_b0",          # ImageNet only
    "vsupp_effnet_b4",          # ImageNet only
    # ── CMConv full quadrature (complex-modulus-conv branch) ─────────────
    # Single weight W; imaginary = rot90(W). ~1.5× FLOPs. Full Mallat stability.
    "cmconv_res18",              # CIFAR + ImageNet
    "cmconv_res34",              # CIFAR + ImageNet
    "cmconv_res50",              # CIFAR + ImageNet  ★ primary benchmark
    "cmconv_res101",             # CIFAR + ImageNet
    "cmconv_convnext_tiny",      # ImageNet only
    "cmconv_convnext_base",      # ImageNet only
    "cmconv_effnet_b0",          # ImageNet only
    "cmconv_effnet_b4",          # ImageNet only
    # ── CMConv half-plane (complex-modulus-conv branch) ───────────────────
    # |W*x|. Same params and compute as baseline. 180° phase symmetry only.
    "cmconv_abs_res18",          # CIFAR + ImageNet
    "cmconv_abs_res34",          # CIFAR + ImageNet
    "cmconv_abs_res50",          # CIFAR + ImageNet  ★ ablation vs full quadrature
    "cmconv_abs_res101",         # CIFAR + ImageNet
    "cmconv_abs_convnext_tiny",  # ImageNet only
    "cmconv_abs_convnext_base",  # ImageNet only
    "cmconv_abs_effnet_b0",      # ImageNet only
    "cmconv_abs_effnet_b4",      # ImageNet only
    # ── SDN (Spectral Decoupled Normalization — freq-norm branch) ────────
    # Drop-in BN replacement using exact FFT Gaussian mask split.
    # Fixes FDN's AvgPool degeneracy on small CIFAR feature maps.
    # ConvNeXt: SDN is a no-op (LayerNorm, not BN2d) — GINSC only.
    "sdn_res18",                  # CIFAR + ImageNet
    "sdn_res34",                  # CIFAR + ImageNet
    "sdn_res50",                  # CIFAR + ImageNet
    "sdn_res101",                 # CIFAR + ImageNet
    "sdn_effnet_b0",              # ImageNet only
    "sdn_effnet_b4",              # ImageNet only
    "sdn_ginsc_res18",            # CIFAR + ImageNet
    "sdn_ginsc_res34",            # CIFAR + ImageNet
    "sdn_ginsc_res50",            # CIFAR + ImageNet  ★ main CIFAR contribution
    "sdn_ginsc_res101",           # CIFAR + ImageNet
    "sdn_ginsc_convnext_tiny",    # ImageNet only (SDN no-op; GINSC only)
    "sdn_ginsc_convnext_base",    # ImageNet only (SDN no-op; GINSC only)
    "sdn_ginsc_effnet_b0",        # ImageNet only
    "sdn_ginsc_effnet_b4",        # ImageNet only  ★ main ImageNet contribution
    # ── FDN (Frequency-Decoupled Normalization — freq-norm branch) ──────
    "fdn_res18",              # CIFAR + ImageNet
    "fdn_res34",              # CIFAR + ImageNet
    "fdn_res50",              # CIFAR + ImageNet
    "fdn_res101",             # CIFAR + ImageNet
    "fdn_convnext_tiny",      # ImageNet only (FDN no-op; same as baseline)
    "fdn_convnext_base",      # ImageNet only (FDN no-op; same as baseline)
    "fdn_effnet_b0",          # ImageNet only
    "fdn_effnet_b4",          # ImageNet only
    # ── INSC (Instance-Normalized Skip Connections) ───────────────────
    "insc_res18",             # CIFAR + ImageNet
    "insc_res34",             # CIFAR + ImageNet
    "insc_res50",             # CIFAR + ImageNet
    "insc_res101",            # CIFAR + ImageNet
    "insc_convnext_tiny",     # ImageNet only
    "insc_convnext_base",     # ImageNet only
    "insc_effnet_b0",         # ImageNet only
    "insc_effnet_b4",         # ImageNet only
    # ── FDN + INSC combined ★ main contribution (freq-norm branch) ────
    "fdn_insc_res18",         # CIFAR + ImageNet
    "fdn_insc_res34",         # CIFAR + ImageNet
    "fdn_insc_res50",         # CIFAR + ImageNet
    "fdn_insc_res101",        # CIFAR + ImageNet
    "fdn_insc_convnext_tiny", # ImageNet only (INSC-only; FDN no-op)
    "fdn_insc_convnext_base", # ImageNet only (INSC-only; FDN no-op)
    "fdn_insc_effnet_b0",     # ImageNet only
    "fdn_insc_effnet_b4",     # ImageNet only
    # ── Gated INSC (learnable gate per block — architecture-agnostic) ────
    "ginsc_res18",            # CIFAR + ImageNet
    "ginsc_res34",            # CIFAR + ImageNet
    "ginsc_res50",            # CIFAR + ImageNet
    "ginsc_res101",           # CIFAR + ImageNet
    "fdn_ginsc_res18",        # CIFAR + ImageNet
    "fdn_ginsc_res34",        # CIFAR + ImageNet
    "fdn_ginsc_res50",        # CIFAR + ImageNet
    "fdn_ginsc_res101",       # CIFAR + ImageNet
    "fdn_ginsc_convnext_tiny",# ImageNet only (GINSC-only; FDN no-op)
    "fdn_ginsc_convnext_base",# ImageNet only (GINSC-only; FDN no-op)
    "fdn_ginsc_effnet_b0",    # ImageNet only
    "fdn_ginsc_effnet_b4",    # ImageNet only
    # ── ShapeBiasNet variants (dual-stream, for comparison) ────
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
    if name == "baseline_res18":           return BaselineResNet18(**kw)
    if name == "baseline_res34":           return BaselineResNet34(**kw)
    if name == "baseline_res50":           return BaselineResNet50(**kw)
    if name == "baseline_res101":          return BaselineResNet101(**kw)
    if name == "baseline_convnext_tiny":   return BaselineConvNeXt("tiny", **kw)
    if name == "baseline_convnext_base":   return BaselineConvNeXt("base", **kw)
    if name == "baseline_efficientnet_b0": return BaselineEfficientNet("b0", **kw)
    if name == "baseline_efficientnet_b4": return BaselineEfficientNet("b4", **kw)

    # ── PMConv-L (Lorentzian, n_steps=1 — pmconv-scale branch) ──
    if name == "pmconv_l_res18":         return PMResNet("18",  **kw)
    if name == "pmconv_l_res34":         return PMResNet("34",  **kw)
    if name == "pmconv_l_res50":         return PMResNet("50",  **kw)
    if name == "pmconv_l_res101":        return PMResNet("101", **kw)
    if name == "pmconv_l_convnext_tiny": return PMConvNeXt("tiny", **kw)
    if name == "pmconv_l_convnext_base": return PMConvNeXt("base", **kw)
    if name == "pmconv_l_effnet_b0":     return PMEfficientNet("b0", **kw)
    if name == "pmconv_l_effnet_b4":     return PMEfficientNet("b4", **kw)

    # ── PMConv-A (adaptive k + shared conductance — this branch) ─
    if name == "pmconv_a_res18":         return PMAdaptiveResNet("18",  **kw)
    if name == "pmconv_a_res34":         return PMAdaptiveResNet("34",  **kw)
    if name == "pmconv_a_res50":         return PMAdaptiveResNet("50",  **kw)
    if name == "pmconv_a_res101":        return PMAdaptiveResNet("101", **kw)
    if name == "pmconv_a_convnext_tiny": return PMAdaptiveConvNeXt("tiny", **kw)
    if name == "pmconv_a_convnext_base": return PMAdaptiveConvNeXt("base", **kw)
    if name == "pmconv_a_effnet_b0":     return PMAdaptiveEfficientNet("b0", **kw)
    if name == "pmconv_a_effnet_b4":     return PMAdaptiveEfficientNet("b4", **kw)
    # ── RobustConv (isotropic — ablation) ─────────────────────
    if name == "robustconv_res18":         return RobustResNet("18",  **kw)
    if name == "robustconv_res34":         return RobustResNet("34",  **kw)
    if name == "robustconv_res50":         return RobustResNet("50",  **kw)
    if name == "robustconv_res101":        return RobustResNet("101", **kw)
    if name == "robustconv_convnext_tiny": return RobustConvNeXt("tiny", **kw)
    if name == "robustconv_convnext_base": return RobustConvNeXt("base", **kw)
    if name == "robustconv_effnet_b0":     return RobustEfficientNet("b0", **kw)
    if name == "robustconv_effnet_b4":     return RobustEfficientNet("b4", **kw)

    # ── VarSupp ───────────────────────────────────────────────
    if name == "vsupp_res18":         return VarSuppResNet("18",  **kw)
    if name == "vsupp_res34":         return VarSuppResNet("34",  **kw)
    if name == "vsupp_res50":         return VarSuppResNet("50",  **kw)
    if name == "vsupp_res101":        return VarSuppResNet("101", **kw)
    if name == "vsupp_convnext_tiny": return VarSuppConvNeXt("tiny", **kw)
    if name == "vsupp_convnext_base": return VarSuppConvNeXt("base", **kw)
    if name == "vsupp_effnet_b0":     return VarSuppEfficientNet("b0", **kw)
    if name == "vsupp_effnet_b4":     return VarSuppEfficientNet("b4", **kw)
    # ── CMConv full quadrature ────────────────────────────────
    if name == "cmconv_res18":         return CMConvResNet("18",  **kw)
    if name == "cmconv_res34":         return CMConvResNet("34",  **kw)
    if name == "cmconv_res50":         return CMConvResNet("50",  **kw)
    if name == "cmconv_res101":        return CMConvResNet("101", **kw)
    if name == "cmconv_convnext_tiny": return CMConvConvNeXt("tiny", **kw)
    if name == "cmconv_convnext_base": return CMConvConvNeXt("base", **kw)
    if name == "cmconv_effnet_b0":     return CMConvEfficientNet("b0", **kw)
    if name == "cmconv_effnet_b4":     return CMConvEfficientNet("b4", **kw)
    # ── CMConv half-plane ─────────────────────────────────────
    if name == "cmconv_abs_res18":         return CMConvAbsResNet("18",  **kw)
    if name == "cmconv_abs_res34":         return CMConvAbsResNet("34",  **kw)
    if name == "cmconv_abs_res50":         return CMConvAbsResNet("50",  **kw)
    if name == "cmconv_abs_res101":        return CMConvAbsResNet("101", **kw)
    if name == "cmconv_abs_convnext_tiny": return CMConvAbsConvNeXt("tiny", **kw)
    if name == "cmconv_abs_convnext_base": return CMConvAbsConvNeXt("base", **kw)
    if name == "cmconv_abs_effnet_b0":     return CMConvAbsEfficientNet("b0", **kw)
    if name == "cmconv_abs_effnet_b4":     return CMConvAbsEfficientNet("b4", **kw)

    # ── SDN ───────────────────────────────────────────────────
    if name == "sdn_res18":               return SDNResNet("18",  **kw)
    if name == "sdn_res34":               return SDNResNet("34",  **kw)
    if name == "sdn_res50":               return SDNResNet("50",  **kw)
    if name == "sdn_res101":              return SDNResNet("101", **kw)
    if name == "sdn_effnet_b0":           return SDNEfficientNet("b0", **kw)
    if name == "sdn_effnet_b4":           return SDNEfficientNet("b4", **kw)
    if name == "sdn_ginsc_res18":         return SDNGISCResNet("18",  **kw)
    if name == "sdn_ginsc_res34":         return SDNGISCResNet("34",  **kw)
    if name == "sdn_ginsc_res50":         return SDNGISCResNet("50",  **kw)
    if name == "sdn_ginsc_res101":        return SDNGISCResNet("101", **kw)
    if name == "sdn_ginsc_convnext_tiny": return SDNGISCConvNeXt("tiny", **kw)
    if name == "sdn_ginsc_convnext_base": return SDNGISCConvNeXt("base", **kw)
    if name == "sdn_ginsc_effnet_b0":     return SDNGISCEfficientNet("b0", **kw)
    if name == "sdn_ginsc_effnet_b4":     return SDNGISCEfficientNet("b4", **kw)

    # ── FDN ───────────────────────────────────────────────────
    if name == "fdn_res18":          return FDNResNet("18",  **kw)
    if name == "fdn_res34":          return FDNResNet("34",  **kw)
    if name == "fdn_res50":          return FDNResNet("50",  **kw)
    if name == "fdn_res101":         return FDNResNet("101", **kw)
    if name == "fdn_convnext_tiny":  return FDNConvNeXt("tiny", **kw)
    if name == "fdn_convnext_base":  return FDNConvNeXt("base", **kw)
    if name == "fdn_effnet_b0":      return FDNEfficientNet("b0", **kw)
    if name == "fdn_effnet_b4":      return FDNEfficientNet("b4", **kw)

    # ── INSC ──────────────────────────────────────────────────
    if name == "insc_res18":         return ISCResNet("18",  **kw)
    if name == "insc_res34":         return ISCResNet("34",  **kw)
    if name == "insc_res50":         return ISCResNet("50",  **kw)
    if name == "insc_res101":        return ISCResNet("101", **kw)
    if name == "insc_convnext_tiny": return ISCConvNeXt("tiny", **kw)
    if name == "insc_convnext_base": return ISCConvNeXt("base", **kw)
    if name == "insc_effnet_b0":     return ISCEfficientNet("b0", **kw)
    if name == "insc_effnet_b4":     return ISCEfficientNet("b4", **kw)

    # ── FDN + INSC (main contribution) ────────────────────────
    if name == "fdn_insc_res18":          return FDNISCResNet("18",  **kw)
    if name == "fdn_insc_res34":          return FDNISCResNet("34",  **kw)
    if name == "fdn_insc_res50":          return FDNISCResNet("50",  **kw)
    if name == "fdn_insc_res101":         return FDNISCResNet("101", **kw)
    if name == "fdn_insc_convnext_tiny":  return FDNISCConvNeXt("tiny", **kw)
    if name == "fdn_insc_convnext_base":  return FDNISCConvNeXt("base", **kw)
    if name == "fdn_insc_effnet_b0":      return FDNISCEfficientNet("b0", **kw)
    if name == "fdn_insc_effnet_b4":      return FDNISCEfficientNet("b4", **kw)

    # ── Gated INSC ────────────────────────────────────────────
    if name == "ginsc_res18":             return GISCResNet("18",  **kw)
    if name == "ginsc_res34":             return GISCResNet("34",  **kw)
    if name == "ginsc_res50":             return GISCResNet("50",  **kw)
    if name == "ginsc_res101":            return GISCResNet("101", **kw)
    if name == "fdn_ginsc_res18":         return FDNGISCResNet("18",  **kw)
    if name == "fdn_ginsc_res34":         return FDNGISCResNet("34",  **kw)
    if name == "fdn_ginsc_res50":         return FDNGISCResNet("50",  **kw)
    if name == "fdn_ginsc_res101":        return FDNGISCResNet("101", **kw)
    if name == "fdn_ginsc_convnext_tiny": return FDNGISCConvNeXt("tiny", **kw)
    if name == "fdn_ginsc_convnext_base": return FDNGISCConvNeXt("base", **kw)
    if name == "fdn_ginsc_effnet_b0":     return FDNGISCEfficientNet("b0", **kw)
    if name == "fdn_ginsc_effnet_b4":     return FDNGISCEfficientNet("b4", **kw)

    # ── ShapeBiasNet ──────────────────────────────────────────
    if name == "shape_custom":        return ShapeBiasNet("custom",        **kw)
    if name == "shape_res18":         return ShapeBiasNet("18",            **kw)
    if name == "shape_res34":         return ShapeBiasNet("34",            **kw)
    if name == "shape_res50":         return ShapeBiasNet("50",            **kw)
    if name == "shape_res101":        return ShapeBiasNet("101",           **kw)
    if name == "shape_convnext_tiny": return ShapeBiasNet("convnext_tiny", **kw)
    if name == "shape_convnext_base": return ShapeBiasNet("convnext_base", **kw)
    if name == "shape_effnet_b0":     return ShapeBiasNet("effnet_b0",     **kw)
    if name == "shape_effnet_b4":     return ShapeBiasNet("effnet_b4",     **kw)

    raise ValueError(f"Unknown model '{name}'. Choose from: {MODEL_NAMES}")