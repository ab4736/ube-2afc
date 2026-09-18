"""
Pairmate 2AFC + CPD with the encoder.

For each test trial (real beta for image A, pairmate B):
    pred_A = encoder(image A), pred_B = encoder(image B)
    pick A if corr(beta, pred_A) > corr(beta, pred_B)
corr here is Pearson over voxels = dot product of the z-scored patterns / n_vox.

CPD (same idea as the grant, but in voxel space instead of CLIP space):
    project the real beta onto the line from pred_B to pred_A.
    +1 = sitting on pred_A (correct), 0 = midpoint, -1 = sitting on pred_B.

Also prints 1-of-N retrieval, a val-set retrieval sanity check, and some
controls that should all come out at chance. Writes a per-trial csv and a json.

usage:
    python eval_2afc.py --data data/sub-06.npz --enc checkpoints/sub-06_infonce.pth
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--enc", required=True)
ap.add_argument("--out", default=None, help="output prefix (default results/<enc name>)")
args = ap.parse_args()
dev = "cuda"
MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

d = np.load(args.data)
Y, pairs, names = d["Y_test"].astype(np.float32), d["pair_idx"], d["test_names"]
model = torch.load(args.enc, weights_only=False).to(dev).eval()
NVOX = model.voxel_embed.shape[0]
assert NVOX == Y.shape[1], f"encoder has {NVOX} voxels but data has {Y.shape[1]}"


def predict(imgs):
    out = []
    vind = torch.arange(NVOX).unsqueeze(0).to(dev)
    with torch.no_grad():
        for i in range(0, len(imgs), 16):
            x = np.stack([((im / 255.0 - MEAN) / STD).transpose(2, 0, 1) for im in imgs[i:i + 16]])
            x = torch.from_numpy(x.astype(np.float32)).to(dev)
            out.append(model(x, vind.repeat(len(x), 1)).float().cpu().numpy())
    return np.concatenate(out)


def zs(a):
    a = a - a.mean(-1, keepdims=True)
    return a / (a.std(-1, keepdims=True) + 1e-8)


def corr(a, b):
    return float((zs(a) * zs(b)).mean())


def trials(beta, pred):
    # both directions of each pair: (shown, foil)
    rows = []
    for a, b in pairs:
        for shown, foil in ((a, b), (b, a)):
            r_s, r_f = corr(beta[shown], pred[shown]), corr(beta[shown], pred[foil])
            axis = pred[shown] - pred[foil]
            mid = (pred[shown] + pred[foil]) / 2
            cpd = 2 * float((beta[shown] - mid) @ axis) / (float(axis @ axis) + 1e-8)
            rows.append((shown, foil, r_s, r_f, int(r_s > r_f), cpd))
    return rows


P = predict(d["img_test"])
rows = trials(Y, P)
hit = np.array([r[4] for r in rows])
cpd = np.array([r[5] for r in rows])

# bootstrap CI, resampling pairs
rng = np.random.RandomState(0)
boot = []
for _ in range(5000):
    pick = rng.randint(0, len(pairs), len(pairs))
    boot.append(hit[np.concatenate([[2 * p, 2 * p + 1] for p in pick])].mean())
ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

# retrieval: each real beta against all test predictions
S = zs(Y) @ zs(P).T / NVOX
top1 = float((S.argmax(1) == np.arange(len(Y))).mean())

# val retrieval: is the encoder any good on normal (non-pair) held-out images?
Yv = d["Y_val"].astype(np.float32)
Sv = zs(Yv) @ zs(predict(d["img_val"])).T / NVOX
val_top1 = float((Sv.argmax(1) == np.arange(len(Yv))).mean())

# controls, these should be ~0.5
ctrl = {}
sh = np.random.RandomState(2).permutation(NVOX)
ctrl["shuffled_voxels"] = float(np.mean([r[4] for r in trials(Y, P[:, sh])]))
ctrl["mean_beta"] = float(np.mean([r[4] for r in trials(np.repeat(Y.mean(0, keepdims=True), len(Y), 0), P)]))
# wrong image: a random other scene's prediction vs the real pairmate. should be <= 0.5
rw, wrong = np.random.RandomState(3), []
for a, b in pairs:
    for shown, foil in ((a, b), (b, a)):
        j = rw.choice([k for k in range(len(Y)) if k not in (a, b)])
        wrong.append(int(corr(Y[shown], P[j]) > corr(Y[shown], P[foil])))
ctrl["wrong_image"] = float(np.mean(wrong))

print(f"\n{os.path.basename(args.enc)} on {os.path.basename(args.data)}")
print(f"  pairmate 2AFC   {hit.mean():.3f}  ({hit.sum()}/{len(hit)})  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]  chance 0.5")
print(f"  CPD             mean {cpd.mean():+.3f}  median {np.median(cpd):+.3f}  CPD>0 {np.mean(cpd > 0):.3f}")
print(f"  retrieval       top-1 {top1:.3f}  (1 of {len(Y)}, chance {1 / len(Y):.3f})")
print(f"  val retrieval   top-1 {val_top1:.3f}  (1 of {len(Yv)}, chance {1 / len(Yv):.3f})")
print(f"  controls        " + "  ".join(f"{k} {v:.2f}" for k, v in ctrl.items()))

prefix = args.out or f"{HERE}/results/{os.path.splitext(os.path.basename(args.enc))[0]}"
os.makedirs(os.path.dirname(prefix), exist_ok=True)
with open(prefix + "_trials.csv", "w") as f:
    f.write("shown,foil,r_shown,r_foil,correct,cpd\n")
    for s, fo, r_s, r_f, c, cp in rows:
        f.write(f"{names[s]},{names[fo]},{r_s:.4f},{r_f:.4f},{c},{cp:.4f}\n")
json.dump(dict(data=args.data, enc=args.enc, n_trials=len(hit), twoafc=float(hit.mean()),
               twoafc_ci=ci, cpd_mean=float(cpd.mean()), cpd_median=float(np.median(cpd)),
               cpd_pos=float(np.mean(cpd > 0)), retrieval_top1=top1, val_retrieval_top1=val_top1,
               controls=ctrl), open(prefix + ".json", "w"), indent=1)
print(f"  wrote {prefix}_trials.csv and .json")
