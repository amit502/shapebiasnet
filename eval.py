import torch, argparse, numpy as np, torchvision, os
import torchvision.transforms as T
from models import build_model

p=argparse.ArgumentParser()
p.add_argument("--model", default="shape_custom")
p.add_argument("--dataset", default="cifar10")
p.add_argument("--ckpt", default=None)
p.add_argument("--eval_all", action="store_true")
args=p.parse_args()

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

C = ['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur',
     'motion_blur','zoom_blur','snow','frost','fog','brightness','contrast',
     'elastic_transform','pixelate','jpeg_compression']

ALL_MODELS = ["baseline_res18","baseline_res50","shape_custom","shape_res18","shape_res50"]

mean10,std10=(0.4914,0.4822,0.4465),(0.247,0.243,0.261)
mean100,std100=(0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)
mean,std = (mean10,std10) if args.dataset=="cifar10" else (mean100,std100)

# -------------------------------------------------
# CIFAR-C folder root
# -------------------------------------------------
CROOT = "./data/CIFAR-10-C" if args.dataset=="cifar10" else "./data/CIFAR-100-C"

# -------- clean ----------
tf=T.Compose([T.ToTensor(),T.Normalize(mean,std)])
DS = torchvision.datasets.CIFAR10 if args.dataset=="cifar10" else torchvision.datasets.CIFAR100
clean = torch.utils.data.DataLoader(
    DS("./data",False,download=True,transform=tf),
    256,shuffle=False,num_workers=2)

# -------- helpers ----------
def prep(x):
    x=torch.from_numpy(x).float().permute(0,3,1,2)/255.
    m=torch.tensor(mean).view(1,3,1,1); s=torch.tensor(std).view(1,3,1,1)
    return (x-m)/s

@torch.no_grad()
def clean_eval(model):
    c=t=0
    for x,y in clean:
        x,y=x.to(DEVICE),y.to(DEVICE)
        c+=(model(x).argmax(1)==y).sum().item(); t+=y.size(0)
    return 100*c/t

@torch.no_grad()
def eval_sub(model,x,y):
    c=t=0
    for i in range(0,len(x),256):
        xb=prep(x[i:i+256]).to(DEVICE)
        yb=torch.from_numpy(y[i:i+256]).to(DEVICE)
        c+=(model(xb).argmax(1)==yb).sum().item(); t+=yb.size(0)
    return 100*c/t

# -------------------------------------------------
# Evaluation runner
# -------------------------------------------------
def run(model_name, ckpt_path):

    print("\n========================================")
    print("Model:", model_name)
    print("Ckpt :", ckpt_path)
    print("========================================")

    num_classes = 10 if args.dataset=="cifar10" else 100
    model = build_model(model_name, num_classes)
    if model_name.startswith("shape"): model.alpha=1.0
    model.load_state_dict(torch.load(ckpt_path,map_location="cpu"))
    model=model.to(DEVICE).eval()

    print("\nClean accuracy:", round(clean_eval(model),2))
    print("\n--- CIFAR-C ---")

    mca=[]
    for c in C:
        x=np.load(f"{CROOT}/{c}.npy"); y=np.load(f"{CROOT}/labels.npy")
        sev=[]
        for s in range(5):
            sev.append(eval_sub(model,
                        x[s*10000:(s+1)*10000],
                        y[s*10000:(s+1)*10000]))
        print(f"{c:18s} | {[round(a,2) for a in sev]} | mean {sum(sev)/5:.2f}")
        mca.append(sum(sev)/5)

    print("\n--------------------------------")
    print("mCA:", round(sum(mca)/len(mca),2))
    print("mCE:", round(100-sum(mca)/len(mca),2))
    print("--------------------------------")

# -------------------------------------------------
# Main
# -------------------------------------------------
if args.eval_all:
    for m in ALL_MODELS:
        ckpt = f"checkpoints/{m}_{args.dataset}.pt"
        if os.path.exists(ckpt):
            run(m, ckpt)
        else:
            print(f"[SKIP] {ckpt} not found")

else:
    if args.ckpt is None:
        raise ValueError("Provide --ckpt or use --eval_all")
    run(args.model, args.ckpt)
