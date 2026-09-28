import os, sys, argparse, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from scipy.ndimage import shift as ndshift

# pretrains the base encoder on nsd from scratch
#
# you do NOT need this to run a 2afc test. use the downloaded base instead. this is only for
# redoing the pretraining, for example to retrain the contrastive objective or add subjects.
# it needs the full nsd dataset and takes days on one gpu
#
# --objective recon     reconstruction only, the baseline
# --objective infonce   adds the contrastive term, this is the change that helps pairmates
#
# honest note: this reproduces the objective, not necessarily the exact hyperparameters of the
# checkpoint we ship. those were not written down. if you retrain, treat your own run as the
# reference and keep the recon and infonce runs matched to each other

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)                 # folder with the nsd files, see readme
ap.add_argument("--objective", choices=["recon", "infonce"], default="infonce")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--n_vox_sample", type=int, default=5000)    # voxels per step, nsd is too big for all
ap.add_argument("--lam", type=float, default=1.0)            # contrastive weight
ap.add_argument("--tau", type=float, default=0.1)
ap.add_argument("--went", type=float, default=0.1)           # entropy weight
ap.add_argument("--inner_ch", type=int, default=128)
ap.add_argument("--embed_dim_vox", type=int, default=256)
ap.add_argument("--dropout", type=float, default=0.25)
ap.add_argument("--r", type=int, default=7)                  # lora rank
ap.add_argument("--alpha", type=float, default=0.01)
ap.add_argument("--hub", default=os.environ.get("UBE_TORCH_HUB", ""))
ap.add_argument("--smoke", action="store_true")              # 3 steps on a tiny slice, just checks it runs
args = ap.parse_args()

torch.manual_seed(0)
np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if args.hub:
    torch.hub.set_dir(args.hub)
    sys.path.insert(0, os.path.join(args.hub, "facebookresearch_dinov2_main"))

need = ["fmri_v2.npz", "num_voxels_all_subjects.npy", "all_images_v2_224.npy"]
missing = [f for f in need if not os.path.exists(os.path.join(args.data_dir, f))]
if missing:
    sys.exit(f"PROBLEM: {args.data_dir} is missing {missing}.\n"
             f"See HUGGINGFACE_UPLOAD.md section 3 for what these files are and how to build them "
             f"from the NSD download. You do not need them to run a 2AFC test.")

nv = np.load(os.path.join(args.data_dir, "num_voxels_all_subjects.npy"))
sub_nvox = nv.sum(1).astype(int)                             # voxels per subject
offset = np.cumsum(sub_nvox) - sub_nvox                      # where each subject starts globally
NUM_VOXELS = int(sub_nvox.sum())

f = np.load(os.path.join(args.data_dir, "fmri_v2.npz"))
type_sample = f["type_sample"]
single_sub = f["single_sub"]
single_fmri = f["single_sub_fmri"]
val_ind = f["val_single_ind"]
imgs = np.load(os.path.join(args.data_dir, "all_images_v2_224.npy"), mmap_mode="r")

is_train = np.ones(single_fmri.shape[0], dtype=bool)
is_train[val_ind] = False
img_single = np.where(type_sample == 1)[0]                   # nsd ids of the single subject images
train_rows = np.where(is_train)[0]
if args.smoke:
    train_rows = train_rows[:256]                            # tiny slice so this finishes in a minute
    args.epochs = 1
print(f"nsd: {NUM_VOXELS} voxels over {len(sub_nvox)} subjects, {len(train_rows)} training rows",
      flush=True)

from models.encoder_models import Encoder, encoder_param
p = encoder_param(NUM_VOXELS)
p.inner_ch = args.inner_ch
p.embed_dim_vox = args.embed_dim_vox
p.drop_out = args.dropout
p.in_channels = 1024        # dinov2 vit-l feature width, the class default is for a bigger backbone
p.in_spatial = 257          # 16x16 patches at 224 plus the cls token
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", source="github", pretrained=True)
for q in dino.parameters():
    q.requires_grad = False                                  # backbone stays frozen, lora adapts it
model = Encoder(p, dino, num_embeds=3, select_layers=[1, 6, 12, 18, 24], r=args.r,
                alpha=args.alpha).to(dev)
train_me = [q for q in model.parameters() if q.requires_grad]
print(f"training {sum(q.numel() for q in train_me) / 1e6:.1f}M parameters", flush=True)
opt = optim.Adam(train_me, lr=args.lr, amsgrad=True)

MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def to_tensor(batch_imgs):
    out = []
    for im in batch_imgs:
        im = np.asarray(im, np.float64) / 255.0
        dx, dy = np.random.randint(-3, 4, size=2)
        im = ndshift(im, [dx, dy, 0], prefilter=False, order=0, mode="nearest")
        out.append(((im - MEAN) / STD).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(out).astype(np.float32)).to(dev)


def zs(a):
    a = a - a.mean(-1, keepdim=True)
    return a / (a.std(-1, keepdim=True) + 1e-8)


# batches are built per subject with one shared voxel sample, which is what makes the
# contrastive term meaningful. with per-item voxel samples the patterns in a batch would live
# in different voxel spaces and comparing them would be meaningless
by_sub = {s: train_rows[single_sub[train_rows] == s] for s in np.unique(single_sub[train_rows])}
t0 = time.time()
step = 0
for ep in range(1, args.epochs + 1):
    model.train()
    order = []
    for s, rows in by_sub.items():
        r = np.random.permutation(rows)
        order += [(s, r[i:i + args.bs]) for i in range(0, len(r), args.bs)]
    np.random.shuffle(order)
    tot, nb = 0.0, 0
    for s, rows in order:
        if len(rows) < 4:
            continue
        nvox_s = sub_nvox[s]
        vox = np.random.randint(0, nvox_s, args.n_vox_sample)   # same voxels for the whole batch
        gvox = torch.from_numpy((vox + offset[s]).astype(np.int64)).to(dev)
        y = torch.from_numpy(np.asarray(single_fmri[rows][:, vox], np.float32)).to(dev)
        x = to_tensor([imgs[img_single[i]] for i in rows])
        opt.zero_grad()
        pred = model(x, gvox.unsqueeze(0).repeat(len(rows), 1))
        loss = F.mse_loss(pred, y) - 0.1 * F.cosine_similarity(pred, y).mean()
        if args.objective == "infonce":
            sim = (zs(pred) @ zs(y).T) / pred.shape[1] / args.tau
            lab = torch.arange(len(rows)).to(dev)
            loss = loss + args.lam * 0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.T, lab))
            if args.went > 0:
                off = F.softmax(sim.clone().fill_diagonal_(-1e9), dim=1)
                loss = loss - args.went * (-(off * torch.log(off + 1e-9)).sum(1).mean())
        loss.backward()
        opt.step()
        tot += loss.item()
        nb += 1
        step += 1
        if step % 100 == 0:
            print(f"  ep{ep} step {step}  loss {tot / nb:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if args.smoke and step >= 3:
            break
    print(f"epoch {ep} done, loss {tot / max(nb, 1):.4f}", flush=True)
    if args.smoke:
        break

os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
shared = {k: v for k, v in model.state_dict().items()
          if not k.startswith("embed_model.") and k != "voxel_embed"}
cfg = dict(param={k: v for k, v in vars(model.param).items()}, num_embeds=model.num_embeds,
           select_layers=list(model.select_layers), r=model.r, alpha=args.alpha,
           include_reg_tokens=False, num_reg_tokens=4, backbone="dinov2_vitl14_reg",
           backbone_is_stock=True, source_checkpoint=f"trained here, objective={args.objective}",
           nsd_voxel_embed_shape=list(model.voxel_embed.shape))
torch.save(dict(config=cfg, shared=shared), args.out)
torch.save(dict(voxel_embed=model.voxel_embed.detach().cpu()),
           args.out.replace(".pt", "_voxel_embed.pt"))
print(f"saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) and the voxel embeddings "
      f"next to it", flush=True)
