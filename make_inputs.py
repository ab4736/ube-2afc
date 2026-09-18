"""
Turn your own data into the input file the other scripts use.

You give it 3 files (+ your images):

  1. betas     .npy or .mat, one row per trial, one column per voxel
  2. trials    .csv, one line per row of the betas, same order, with a column
               called  image  (which picture was on screen). Optional column
               session  if you want betas z-scored within each session.
  3. pairs     .csv with columns  image_a,image_b  (one line per pairmate pair)

  and the images themselves: either full paths in the image column, or just
  file names plus --images pointing at the folder they're in.

What it does: z-scores each voxel (within session if you gave one), averages
repeats of the same image, takes the pairmate images out as the test set,
keeps 10% of the rest as a sanity-check set, resizes the images to 224x224.

usage:
    python make_inputs.py --name mysub --betas betas.npy --trials trials.csv \
        --pairs pairs.csv --images /path/to/image/folder
    -> data/mysub.npz
"""
import os, sys, csv, argparse, collections
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="what to call this dataset, e.g. sub-08")
ap.add_argument("--betas", required=True)
ap.add_argument("--trials", required=True)
ap.add_argument("--pairs", required=True)
ap.add_argument("--images", default="", help="folder with the images (skip if the csv has full paths)")
ap.add_argument("--val_frac", type=float, default=0.10)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def stop(msg):
    sys.exit("PROBLEM: " + msg)


def read_csv(path):
    if not os.path.exists(path):
        stop(f"can't find {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [{k.strip().lower(): (v or "").strip() for k, v in r.items() if k} for r in rows]


# ---- betas
if not os.path.exists(args.betas):
    stop(f"can't find {args.betas}")
if args.betas.endswith(".mat"):
    import scipy.io as sio
    m = {k: v for k, v in sio.loadmat(args.betas).items() if not k.startswith("_")}
    if len(m) != 1:
        stop(f"{args.betas} has {len(m)} variables {list(m)}, save just the betas matrix in it")
    betas = np.asarray(list(m.values())[0], dtype=np.float32)
else:
    betas = np.load(args.betas).astype(np.float32)
if betas.ndim != 2:
    stop(f"betas should be a 2D table (trials x voxels) but has shape {betas.shape}")

# ---- trials
trials = read_csv(args.trials)
if not trials or "image" not in trials[0]:
    stop(f"{args.trials} needs a column called 'image'")
if len(trials) != len(betas):
    if len(trials) == betas.shape[1]:
        stop(f"betas look sideways ({betas.shape}). Rows should be trials, columns voxels. Transpose it.")
    stop(f"{args.trials} has {len(trials)} lines but betas has {len(betas)} rows. Need one line per row.")
images = [t["image"] for t in trials]
sess = [t.get("session", "") for t in trials]
bad = ~np.isfinite(betas).all(0)
if bad.any():
    print(f"note: dropping {bad.sum()} voxels that have NaN/inf somewhere")
    betas = betas[:, ~bad]

# ---- z-score each voxel within session
for s in sorted(set(sess)):
    r = np.array([i for i, x in enumerate(sess) if x == s])
    b = betas[r]
    betas[r] = (b - b.mean(0)) / (b.std(0) + 1e-8)

# ---- average repeats per image
rows = collections.defaultdict(list)
for i, im in enumerate(images):
    rows[im].append(i)
names = sorted(rows)
Y = {n: betas[rows[n]].mean(0) for n in names}

# ---- pairs = test set
pairs = read_csv(args.pairs)
if not pairs or "image_a" not in pairs[0] or "image_b" not in pairs[0]:
    stop(f"{args.pairs} needs columns 'image_a' and 'image_b'")
test_names, pair_idx = [], []
for p in pairs:
    for k in ("image_a", "image_b"):
        if p[k] not in Y:
            stop(f"pair image '{p[k]}' is never in {args.trials}, so there's no beta for it")
    ids = []
    for k in ("image_a", "image_b"):
        if p[k] not in test_names:
            test_names.append(p[k])
        ids.append(test_names.index(p[k]))
    pair_idx.append(ids)

other = [n for n in names if n not in set(test_names)]
perm = np.random.RandomState(args.seed).permutation(len(other))
n_val = int(len(other) * args.val_frac)
val_names = [other[i] for i in perm[:n_val]]
train_names = [other[i] for i in perm[n_val:]]


def load(n):
    p = n if os.path.isabs(n) or not args.images else os.path.join(args.images, n)
    if not os.path.exists(p):
        stop(f"can't find the image {p}")
    return np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.uint8)


print(f"{args.name}: {betas.shape[1]} voxels, {len(names)} unique images from {len(betas)} trials")
print(f"  train {len(train_names)}, val {len(val_names)}, test {len(test_names)} images / {len(pair_idx)} pairs")
print("loading images...")
out = f"{HERE}/data/{args.name}.npz"
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(
    out,
    Y_train=np.stack([Y[n] for n in train_names]), img_train=np.stack([load(n) for n in train_names]),
    Y_val=np.stack([Y[n] for n in val_names]), img_val=np.stack([load(n) for n in val_names]),
    Y_test=np.stack([Y[n] for n in test_names]), img_test=np.stack([load(n) for n in test_names]),
    pair_idx=np.array(pair_idx), test_names=np.array([os.path.basename(n) for n in test_names]),
)
print("wrote", out)
