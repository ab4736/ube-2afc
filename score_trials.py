import os, sys, csv, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ube.online import PairmateScorer

# scores trials one at a time out of the prediction cache, the same way the online loop does
# use this after a session to get 2afc and cpd per trial, no gpu needed

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True, help="prediction cache from make_pred_cache.py")
ap.add_argument("--betas", required=True, help=".npy (n_trials, n_voxels)")
ap.add_argument("--trials", required=True, help="csv with columns shown,foil")
ap.add_argument("--out", default="", help="optional csv to write the per trial results to")
ap.add_argument("--causal_z", action="store_true", help="normalise using only past trials")
args = ap.parse_args()

s = PairmateScorer.from_cache(args.cache)
betas = np.load(args.betas)
rows = list(csv.DictReader(open(args.trials, newline="", encoding="utf-8-sig")))
rows = [{k.strip().lower(): (v or "").strip() for k, v in r.items() if k} for r in rows]
if len(rows) != len(betas):
    sys.exit(f"PROBLEM: {args.trials} has {len(rows)} lines but betas has {len(betas)} rows")

out = []
for i, r in enumerate(rows):
    b = betas[i]
    if args.causal_z:
        s.update_running_stats(b)
        b = s.causal_z(b)
        res = s.score(b, r["shown"], r["foil"], z=False)
    else:
        res = s.score(b, r["shown"], r["foil"])
    out.append(res)

acc = float(np.mean([o["correct"] for o in out]))
cpd = np.array([o["cpd"] for o in out])
print(f"{len(out)} trials")
print(f"  2AFC  {acc:.3f}  ({sum(o['correct'] for o in out)}/{len(out)})  chance 0.5")
print(f"  CPD   mean {cpd.mean():+.3f}  median {np.median(cpd):+.3f}  CPD>0 {np.mean(cpd > 0):.3f}")
if args.out:
    with open(args.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    print("  wrote", args.out)
