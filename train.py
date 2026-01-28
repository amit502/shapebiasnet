import torch, argparse, random, numpy as np, torchvision
import torchvision.transforms as T
from models import build_model

# ---------------- setup ----------------
SEED=42
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.deterministic=True
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

# ---------------- args -----------------
p = argparse.ArgumentParser()
p.add_argument("--model", default="shape_custom")
p.add_argument("--dataset", default="cifar10")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--run_all", action="store_true")
args = p.parse_args()

MODELS = ["baseline_res18","baseline_res50","shape_custom","shape_res18","shape_res50"]
if args.run_all: runs = MODELS
else: runs = [args.model]

# ---------------- data -----------------
mean10,std10=(0.4914,0.4822,0.4465),(0.247,0.243,0.261)
mean100,std100=(0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)
mean,std = (mean10,std10) if args.dataset=="cifar10" else (mean100,std100)

tf = T.Compose([T.ToTensor(),T.Normalize(mean,std)])

DS = torchvision.datasets.CIFAR10 if args.dataset=="cifar10" else torchvision.datasets.CIFAR100
trainset = DS("./data",True,download=True,transform=tf)
testset  = DS("./data",False,download=True,transform=tf)

trainloader = torch.utils.data.DataLoader(trainset,128,shuffle=True,num_workers=2,pin_memory=True)
testloader  = torch.utils.data.DataLoader(testset,256,shuffle=False,num_workers=2,pin_memory=True)

# ---------------- train fn --------------
@torch.no_grad()
def test(model):
    model.eval(); c=t=0
    for x,y in testloader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        c+=(model(x).argmax(1)==y).sum().item(); t+=y.size(0)
    return 100*c/t

# ---------------- loop ------------------
for name in runs:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED) 
    random.seed(SEED)
    print("\n==============================")
    print("Training:", name)
    print("==============================")

    model = build_model(name, 10 if args.dataset=="cifar10" else 100).to(DEVICE)
    shape = name.startswith("shape")
    if shape: model.alpha=0.0

    opt = torch.optim.SGD(model.parameters(),0.1,momentum=0.9,weight_decay=1e-4,nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best=0
    for ep in range(1,args.epochs+1):
        if shape:
            if ep<=8: model.alpha=0
            else: model.alpha=min(1,(ep-8)/16)

        model.train(); loss_sum=0
        for x,y in trainloader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss=crit(model(x),y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step(); loss_sum+=loss.item()

        sch.step()
        acc=test(model)
        print(f"{ep:03d} | α={getattr(model,'alpha',1):.2f} | loss {loss_sum/len(trainloader):.3f} | acc {acc:.2f}")

        if acc>best:
            best=acc
            torch.save(model.state_dict(), f"checkpoints/{name}_{args.dataset}.pt")

    print("Best:",best)
