import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet18, resnet50

# ============================================================
# ------------------ BASELINE MODELS -------------------------
# ============================================================

class BaselineResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


class BaselineResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet50(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)

# ============================================================
# ------------------ SHAPE COMPONENTS ------------------------
# ============================================================

class OrientationBank(nn.Module):
    def __init__(self, out_ch=16):
        super().__init__()
        kernels = []
        for k in range(out_ch):
            theta = math.pi * k / out_ch
            gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
            gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
            kernel = math.cos(theta)*gx + math.sin(theta)*gy
            kernels.append(kernel)
        weight = torch.stack(kernels).unsqueeze(1)
        self.register_buffer("weight", weight)

    def forward(self, x):
        x = x.mean(1, keepdim=True)
        e = F.conv2d(x, self.weight, padding=1)
        e = torch.abs(e)
        e = e / (e.mean(dim=(2,3), keepdim=True) + 1e-6)
        return e


class ShapeDiffusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
        self.register_buffer("lap", lap.view(1,1,3,3))

    def forward(self, x):
        lap = F.conv2d(x, self.lap.repeat(x.size(1),1,1,1),
                       padding=1, groups=x.size(1))
        y = x + 0.12 * lap
        y = self.dw(y)
        y = self.pw(y)
        return F.relu(self.bn(y) + x)


class ShapeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge = OrientationBank(16)

        self.stage1 = nn.Sequential(
            nn.Conv2d(16, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ShapeDiffusion(64),
            ShapeDiffusion(64),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ShapeDiffusion(128),
            ShapeDiffusion(128),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ShapeDiffusion(256),
        )

    def forward(self, x):
        s1 = self.stage1(self.edge(x))
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return s1, s2, s3


class ShapeGate(nn.Module):
    def __init__(self, rgb_ch, shape_ch):
        super().__init__()
        self.proj = nn.Conv2d(shape_ch, rgb_ch, 1)
        self.bn = nn.BatchNorm2d(rgb_ch)

    def forward(self, rgb, shape, alpha):
        g = torch.sigmoid(self.bn(self.proj(shape)))
        return rgb * (1 - alpha) + rgb * g * alpha

# ============================================================
# ---------------- RGB BACKBONES -----------------------------
# ============================================================

class RGBCustom(nn.Module):
    def __init__(self):
        super().__init__()
        self.r1 = nn.Sequential(nn.Conv2d(3,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU())
        self.r2 = nn.Sequential(nn.Conv2d(64,128,3,2,1), nn.BatchNorm2d(128), nn.ReLU())
        self.r3 = nn.Sequential(
            nn.Conv2d(128,256,3,2,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU()
        )

    def forward(self,x):
        r1 = self.r1(x)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        return r1,r2,r3


class RGBResNet(nn.Module):
    def __init__(self, depth="18"):
        super().__init__()
        net = resnet18(weights=None) if depth=="18" else resnet50(weights=None)
        net.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
        net.maxpool = nn.Identity()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.l1, self.l2, self.l3 = net.layer1, net.layer2, net.layer3

        # 🔹 store output channels for gates
        if depth=="18":
            self.out_ch = [64, 128, 256]
        else:
            self.out_ch = [256, 512, 1024]

    def forward(self,x):
        r1 = self.l1(self.stem(x))
        r2 = self.l2(r1)
        r3 = self.l3(r2)
        return r1,r2,r3

# ============================================================
# ---------------- SHAPE-BIAS NET ----------------------------
# ============================================================

class ShapeBiasNet(nn.Module):
    def __init__(self, rgb_type="custom", num_classes=10):
        super().__init__()
        self.shape = ShapeEncoder()
        self.rgb   = RGBCustom() if rgb_type=="custom" else RGBResNet(rgb_type)

        # 🔹 set gate channels correctly based on RGB backbone
        if rgb_type=="custom" or rgb_type=="18":
            # self.g1 = ShapeGate(64, 64)
            # self.g2 = ShapeGate(128, 128)
            # self.g3 = ShapeGate(256, 256)
            fusion_in_ch = 512
        elif rgb_type=="50":
            # self.g1 = ShapeGate(256, 64)
            # self.g2 = ShapeGate(512, 128)
            # self.g3 = ShapeGate(1024, 256)
            fusion_in_ch = 1024 + 256

        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_ch, 320,1), nn.BatchNorm2d(320), nn.ReLU(),
            nn.Conv2d(320,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU()
        )

        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(256, num_classes))
        self.alpha = 0.0

    def forward(self,x):
        s1,s2,s3 = self.shape(x)
        r1,r2,r3 = self.rgb(x)

        # r1 = self.g1(r1, F.interpolate(s1, r1.shape[2:]), self.alpha)
        # r2 = self.g2(r2, F.interpolate(s2, r2.shape[2:]), self.alpha)
        # r3 = self.g3(r3, F.interpolate(s3, r3.shape[2:]), self.alpha)

        f = self.fusion(torch.cat([r3, F.interpolate(s3, r3.shape[2:])],1))
        return self.head(f)

# ============================================================
# ----------------- MODEL REGISTRY ---------------------------
# ============================================================

def build_model(name, num_classes):
    if name=="baseline_res18": return BaselineResNet18(num_classes)
    if name=="baseline_res50": return BaselineResNet50(num_classes)
    if name=="shape_custom":   return ShapeBiasNet("custom", num_classes)
    if name=="shape_res18":    return ShapeBiasNet("18", num_classes)
    if name=="shape_res50":    return ShapeBiasNet("50", num_classes)
    raise ValueError("Unknown model name")
