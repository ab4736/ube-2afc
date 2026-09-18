"""
Build the input file for one Bixby subject (sub-06 / sub-07).

Reads {sub}_roi_vox_all_sessions.pkl, uses the pretraining sessions (01/02/03),
keeps the union_mask voxels, z-scores each voxel within each session, and
averages repeats of the same image. The 26 pair_* images (13 pairmate pairs)
are held out as the test set so the encoder never sees them.

If you have a different dataset, you don't need this script. Just write an npz
with the same keys (see README).

usage:
    python make_inputs_bixby.py --sub sub-06
    -> data/sub-06.npz
"""
import os, pickle, argparse, collections
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = "/scratch/gpfs/KNORMAN/ab4736/brainit-fmri"
STIM = "/scratch/gpfs/KNORMAN/miguelp/mindeye_offline/all_stimuli"

ap = argparse.ArgumentParser()
ap.add_argument("--sub", required=True)
ap.add_argument("--sessions", default="01,02,03")
ap.add_argument("--mask", default="union_mask")
ap.add_argument("--val_frac", type=float, default=0.10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
args = ap.parse_args()


def image_path(trial):
    if trial.startswith("notspecial_"):
        return f"{STIM}/shared1000_notspecial/{trial}.png"
    if trial.startswith("special_"):
        return f"{STIM}/special515/{trial}.jpg"
    if trial.startswith("unchosen_"):
        return f"{STIM}/unchosen_nsd_1000_images/{trial}.png"
    if trial.startswith("pair_"):
        return f"{STIM}/MST_pairs/{trial}.jpg"
    raise ValueError(f"don't know where the image for {trial} is")


def load_img(path):
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


with open(f"{PKL_DIR}/{args.sub}_roi_vox_all_sessions.pkl", "rb") as f:
    d = pickle.load(f)
mask = d[args.mask].astype(bool)

# z-score each voxel within session, keep masked voxels, pool sessions
trials, betas = [], []
for s in args.sessions.split(","):
    b = d[s]["roi"].astype(np.float32)
    b = (b - b.mean(0, keepdims=True)) / (b.std(0, keepdims=True) + 1e-8)
    betas.append(b[:, mask])
    trials += list(d[s]["trial"])
betas = np.concatenate(betas, 0)

# average repeats of each image
rows = collections.defaultdict(list)
for i, t in enumerate(trials):
    rows[t].append(i)
names = sorted(rows)
Y = np.stack([betas[rows[t]].mean(0) for t in names]).astype(np.float32)
idx = {t: i for i, t in enumerate(names)}

# test set = the pair_* images, grouped into pairs by pair number
groups = collections.defaultdict(list)
for t in names:
    if t.startswith("pair_"):
        groups[t.split("_")[1]].append(t)
test_names, pair_idx = [], []
for pid in sorted(groups, key=int):
    a, b = sorted(groups[pid])
    assert len(groups[pid]) == 2, f"pair {pid} doesn't have exactly 2 images"
    pair_idx.append((len(test_names), len(test_names) + 1))
    test_names += [a, b]

# train / val split over everything else
other = np.array([i for i, t in enumerate(names) if not t.startswith("pair_")])
perm = np.random.RandomState(args.seed).permutation(len(other))
n_val = int(len(other) * args.val_frac)
val = other[perm[:n_val]]
train = other[perm[n_val:]]

train_names = [names[i] for i in train]
val_names = [names[i] for i in val]
assert not (set(train_names + val_names) & set(test_names)), "test images leaked into training"

print(f"{args.sub}: {mask.sum()} voxels, train {len(train)}, val {len(val)}, "
      f"test {len(test_names)} images / {len(pair_idx)} pairs")
print("loading images...")
out = args.out or f"{HERE}/data/{args.sub}.npz"
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(
    out,
    Y_train=Y[train], img_train=np.stack([load_img(image_path(t)) for t in train_names]),
    Y_val=Y[val], img_val=np.stack([load_img(image_path(t)) for t in val_names]),
    Y_test=np.stack([Y[idx[t]] for t in test_names]),
    img_test=np.stack([load_img(image_path(t)) for t in test_names]),
    pair_idx=np.array(pair_idx),
    test_names=np.array(test_names),
)
print("wrote", out)
