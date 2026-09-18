"""
Fit the UBE encoder to a new subject.

The base encoder (DINOv2 backbone + the shared UBE layers, trained on NSD) is
frozen. The only thing we train is voxel_embed: one learned vector per voxel of
YOUR subject. That's what lets the same encoder predict a brain it has never seen.

loss = MSE - 0.1 * cosine                      (reconstruction)
     + lam * InfoNCE                           (pred_i should match obs_i more
                                                than any other obs_j in the batch)
     - went * entropy                          (small regularizer on the InfoNCE)

--lam 0 --went 0 gives the plain reconstruction encoder.
--lam 1 --went 0.1 is the discriminative one (this is what we use).

usage:
    python train_encoder.py --data data/sub-06.npz --tag sub-06_infonce
    -> checkpoints/sub-06_infonce.pth
"""
import os, sys, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so the pickled base finds models/ and dinov2/ in this folder
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from scipy.ndimage import shift as ndshift

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="npz from make_inputs_bixby.py (or your own)")
ap.add_argument("--tag", required=True)
ap.add_argument("--base", default="/scratch/gpfs/KNORMAN/ab4736/brainit-fmri/results/"
                                  "saved_models/encoder_ch128_base_nce20_ent.pth")
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--tau", type=float, default=0.1)
ap.add_argument("--went", type=float, default=0.1)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--outdir", default=f"{HERE}/checkpoints")
args = ap.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
dev = "cuda"

MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def to_tensor(imgs, augment):
    # imgs: uint8 (n,224,224,3). ImageNet normalization, same as the base was trained with.
    # augment = small random shift (+-3 px), train only.
    out = []
    for img in imgs:
        im = img / 255.0
        if augment:
            dx, dy = np.random.randint(-3, 4, size=2)
            im = ndshift(im, [dx, dy, 0], prefilter=False, order=0, mode="nearest")
        out.append(((im - MEAN) / STD).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(out).astype(np.float32)).to(dev)


def zs(a):
    a = a - a.mean(-1, keepdim=True)
    return a / (a.std(-1, keepdim=True) + 1e-8)


d = np.load(args.data)
Xtr, Ytr = d["img_train"], d["Y_train"].astype(np.float32)
N, NVOX = Ytr.shape
print(f"[{args.tag}] {args.data}: {N} train images, {NVOX} voxels, "
      f"lam={args.lam} went={args.went}", flush=True)

model = torch.load(args.base, weights_only=False)
for p in model.parameters():
    p.requires_grad = False
EMB = model.voxel_embed.shape[1]
# fresh random voxel embeddings for this subject (same init scale as the base)
ve = (0.1 / (2 * np.sqrt(EMB))) * torch.randn(NVOX, EMB)
model.voxel_embed = nn.Parameter(ve.float(), requires_grad=True)
model = model.to(dev)
opt = optim.Adam([model.voxel_embed], lr=args.lr, amsgrad=True)

Y_t = torch.from_numpy(Ytr).to(dev)
vind = torch.arange(NVOX).unsqueeze(0).to(dev)
t0 = time.time()
for ep in range(1, args.epochs + 1):
    model.train()
    perm = np.random.permutation(N)
    tot, nb = 0.0, 0
    for s in range(0, N, args.bs):
        idx = perm[s:s + args.bs]
        if len(idx) < 4:
            continue
        x = to_tensor(Xtr[idx], augment=True)
        obs = Y_t[torch.from_numpy(idx).to(dev)]
        opt.zero_grad()
        pred = model(x, vind.repeat(len(idx), 1))
        loss = F.mse_loss(pred, obs) - 0.1 * F.cosine_similarity(pred, obs).mean()
        if args.lam > 0:
            sim = (zs(pred) @ zs(obs).T) / NVOX / args.tau
            lab = torch.arange(len(idx)).to(dev)
            loss = loss + args.lam * 0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.T, lab))
            if args.went > 0:
                p_off = F.softmax(sim.clone().fill_diagonal_(-1e9), dim=1)
                H = -(p_off * torch.log(p_off + 1e-9)).sum(1).mean()
                loss = loss - args.went * H
        loss.backward()
        opt.step()
        tot += loss.item()
        nb += 1
    if ep == 1 or ep % 5 == 0:
        print(f"  epoch {ep:2d}  loss {tot / nb:.4f}  ({time.time() - t0:.0f}s)", flush=True)

os.makedirs(args.outdir, exist_ok=True)
out = f"{args.outdir}/{args.tag}.pth"
torch.save(model, out)
print(f"[{args.tag}] saved {out}", flush=True)
