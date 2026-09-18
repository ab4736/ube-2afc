"""
Checks an input npz before you spend 20 min training on it.
Prints what's wrong in plain words, or ALL GOOD.

usage:
    python check_inputs.py data/sub-06.npz
"""
import sys, hashlib
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else None
if path is None:
    sys.exit("usage: python check_inputs.py data/<name>.npz")

problems, warnings = [], []
try:
    d = np.load(path)
except Exception as e:
    sys.exit(f"PROBLEM: couldn't open {path} ({e})")

need = ["Y_train", "img_train", "Y_val", "img_val", "Y_test", "img_test", "pair_idx"]
missing = [k for k in need if k not in d.files]
if missing:
    sys.exit(f"PROBLEM: {path} is missing {missing}. It has {d.files}. See the README table.")

nvox = d["Y_train"].shape[1] if d["Y_train"].ndim == 2 else None
for part in ("train", "val", "test"):
    Y, img = d[f"Y_{part}"], d[f"img_{part}"]
    if Y.ndim != 2:
        problems.append(f"Y_{part} should be 2D (images x voxels) but has shape {Y.shape}")
        continue
    if Y.shape[1] != nvox:
        problems.append(f"Y_{part} has {Y.shape[1]} voxels but Y_train has {nvox}. Same voxels everywhere.")
    if img.ndim != 4 or img.shape[1:] != (224, 224, 3):
        problems.append(f"img_{part} should be (n, 224, 224, 3) but is {img.shape}")
    if img.dtype != np.uint8:
        problems.append(f"img_{part} should be uint8 (0-255) but is {img.dtype}")
    elif img.size and img.max() <= 1:
        problems.append(f"img_{part} max value is {img.max()}, looks like it was already scaled. Give plain 0-255 images.")
    if len(Y) != len(img):
        problems.append(f"Y_{part} has {len(Y)} rows but img_{part} has {len(img)} images. Need one image per row.")
    if not np.isfinite(Y).all():
        problems.append(f"Y_{part} has NaN or inf values ({(~np.isfinite(Y)).sum()} of them)")
    print(f"  {part:5s} {len(Y):5d} images, {Y.shape[1]} voxels")

p = d["pair_idx"]
nt = len(d["Y_test"])
if p.ndim != 2 or p.shape[1] != 2:
    problems.append(f"pair_idx should be (n_pairs, 2) but is {p.shape}")
else:
    if p.min() < 0 or p.max() >= nt:
        problems.append(f"pair_idx has numbers outside 0..{nt - 1} (there are {nt} test images)")
    if (p[:, 0] == p[:, 1]).any():
        problems.append("pair_idx pairs an image with itself")
    flat = p.ravel()
    if len(set(flat.tolist())) != len(flat):
        warnings.append("some test image is in more than one pair (fine if that's on purpose)")
    unused = set(range(nt)) - set(flat.tolist())
    if unused:
        warnings.append(f"{len(unused)} test images aren't in any pair, they'll just be ignored")
    print(f"  pairs {len(p):5d}")

# pairmate images must not be in training. compare the actual pixels.
if not problems:
    h = lambda im: hashlib.md5(im.tobytes()).hexdigest()
    seen = {h(im) for im in d["img_train"]} | {h(im) for im in d["img_val"]}
    leaked = [i for i, im in enumerate(d["img_test"]) if h(im) in seen]
    if leaked:
        problems.append(f"{len(leaked)} test (pairmate) images also appear in train/val, e.g. test row "
                        f"{leaked[0]}. Take them out of training or the 2AFC number means nothing.")

    Yt = d["Y_train"].astype(np.float64)
    m, s = np.abs(Yt.mean(0)).mean(), Yt.std(0).mean()
    if m > 0.5 or not (0.3 < s < 3):
        warnings.append(f"training betas don't look z-scored per voxel (avg |mean| {m:.2f}, avg sd {s:.2f}). "
                        f"It'll still run but z-scoring usually works better.")
    if len(Yt) < 500:
        warnings.append(f"only {len(Yt)} training images. It'll run, but fewer than ~1000 usually fits worse.")

for w in warnings:
    print("  warning:", w)
if problems:
    for pr in problems:
        print("  PROBLEM:", pr)
    sys.exit(1)
print("ALL GOOD")
