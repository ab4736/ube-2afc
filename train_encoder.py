import os, sys, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from scipy.ndimage import shift as ndshift

# fits the encoder to one subject
# the pretrained base stays frozen, the only thing we train is voxel_embed, which is one
# learned vector per voxel of this person. that is why 2000 images is enough
#
# --lam 1 --went 0.1   contrastive, each prediction has to match its own beta better than
#                      the other betas in the batch. this is the one that helps pairmates
# --lam 0 --went 0     plain reconstruction, the baseline

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="input npz from make_inputs.py")
ap.add_argument("--tag", required=True, help="name for the output checkpoint")
ap.add_argument("--base", required=True, help="pretrained base weights, e.g. checkpoints/ube_base_infonce.pt")
ap.add_argument("--lam", type=float, default=1.0, help="weight on the contrastive term, 0 turns it off")
ap.add_argument("--tau", type=float, default=0.1, help="contrastive temperature")
ap.add_argument("--went", type=float, default=0.1, help="weight on the entropy term")
ap.add_argument("--epochs", type=int, default=40, help="passes over the training images")
ap.add_argument("--bs", type=int, default=32, help="batch size, also how many negatives the contrastive term sees")
ap.add_argument("--lr", type=float, default=1e-3, help="learning rate")
ap.add_argument("--seed", type=int, default=0, help="random seed")
ap.add_argument("--hub", default=os.environ.get("UBE_TORCH_HUB", ""), help="torch hub cache with dinov2, so it works offline")
ap.add_argument("--outdir", default=f"{HERE}/checkpoints", help="where to save the fitted encoder")
ap.add_argument("--warm_embed", default="", help="optional nsd_voxel_embed_*.pt to warm start from")
ap.add_argument("--warm_map", default="", help="int .npy, one entry per voxel, which nsd voxel it matches")
args = ap.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if args.hub:
    torch.hub.set_dir(args.hub)
    sys.path.insert(0, os.path.join(args.hub, "facebookresearch_dinov2_main"))

MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def to_tensor(imgs, augment):
    out = []
    for img in imgs:
        im = img / 255.0
        if augment:
            dx, dy = np.random.randint(-3, 4, size=2)     # small jitter, same as pretraining
            im = ndshift(im, [dx, dy, 0], prefilter=False, order=0, mode="nearest")
        out.append(((im - MEAN) / STD).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(out).astype(np.float32)).to(dev)


def zs(a):
    a = a - a.mean(-1, keepdim=True)
    return a / (a.std(-1, keepdim=True) + 1e-8)


d = np.load(args.data)
Xtr, Ytr = d["img_train"], d["Y_train"].astype(np.float32)
N, NVOX = Ytr.shape
print(f"[{args.tag}] {N} training images, {NVOX} voxels, lam={args.lam} went={args.went}", flush=True)

# warm start is optional. instead of random voxel embeddings, each voxel starts from the nsd
# voxel it most resembles. you have to supply that mapping yourself, see the readme
warm = None
if args.warm_embed:
    if not args.warm_map:
        sys.exit("PROBLEM: --warm_embed also needs --warm_map, which says what nsd voxel each of "
                 "your voxels should copy. see the warm start section of the readme")
    nsd = torch.load(args.warm_embed, map_location="cpu")["voxel_embed"]
    mp = np.load(args.warm_map)
    if len(mp) != NVOX:
        sys.exit(f"PROBLEM: --warm_map has {len(mp)} entries but this subject has {NVOX} voxels")
    if mp.min() < 0 or mp.max() >= nsd.shape[0]:
        sys.exit(f"PROBLEM: --warm_map values have to be between 0 and {nsd.shape[0] - 1}")
    warm = nsd[torch.from_numpy(mp.astype(np.int64))].clone()
    print(f"[{args.tag}] warm starting from {os.path.basename(args.warm_embed)}", flush=True)

if args.base.endswith(".pt"):                             # small file, we rebuild the model
    from ube.load import load_encoder
    model = load_encoder(args.base, n_voxels=NVOX, hub=args.hub, device=dev, voxel_embed=warm)
    for p in model.parameters():
        p.requires_grad = False
    model.voxel_embed.requires_grad = True                # only the voxel embeddings train
else:                                                     # the old 1.5 GB pickled checkpoint
    model = torch.load(args.base, weights_only=False, map_location="cpu")
    for p in model.parameters():
        p.requires_grad = False
    EMB = model.voxel_embed.shape[1]
    ve = warm if warm is not None else (0.1 / (2 * np.sqrt(EMB))) * torch.randn(NVOX, EMB)
    model.voxel_embed = nn.Parameter(ve.float(), requires_grad=True)
    model = model.to(dev)

# initialise the voxel embeddings from their own generator, so the result only depends on
# --seed and not on how many random numbers building the backbone happened to use. without this
# the same seed gives slightly different answers depending on how the base was loaded
if warm is None:
    EMB = model.voxel_embed.shape[1]
    g = torch.Generator().manual_seed(args.seed)
    scale = 0.1 / (2 * np.sqrt(EMB))
    model.voxel_embed.data = (scale * torch.randn(NVOX, EMB, generator=g)).to(dev)

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
            continue                                      # tiny batches make the contrastive term junk
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
                off = F.softmax(sim.clone().fill_diagonal_(-1e9), dim=1)
                loss = loss - args.went * (-(off * torch.log(off + 1e-9)).sum(1).mean())
        loss.backward()
        opt.step()
        tot += loss.item()
        nb += 1
    if ep == 1 or ep % 5 == 0:
        print(f"  epoch {ep:2d}  loss {tot / nb:.4f}  ({time.time() - t0:.0f}s)", flush=True)

os.makedirs(args.outdir, exist_ok=True)
out = f"{args.outdir}/{args.tag}.pth"
# saves the voxel embeddings and which base they belong to, a few MB instead of 1.2 GB
torch.save(dict(voxel_embed=model.voxel_embed.detach().cpu(), base=os.path.abspath(args.base),
                n_voxels=NVOX, data=os.path.abspath(args.data), tag=args.tag,
                lam=args.lam, went=args.went, epochs=args.epochs, seed=args.seed), out)
print(f"[{args.tag}] saved {out}  {os.path.getsize(out) / 1e6:.1f} MB", flush=True)
