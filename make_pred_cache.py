import os, sys, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
from ube.cache import build_pred_cache

# predicts a brain pattern for every candidate image once and saves it
# do this before a scan, then online scoring is just dot products

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="input npz, uses img_test and test_names")
ap.add_argument("--enc", required=True, help="fitted encoder from train_encoder.py")
ap.add_argument("--out", required=True, help="where to write the prediction cache")
ap.add_argument("--hub", default=os.environ.get("UBE_TORCH_HUB", ""))
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

if args.hub:
    torch.hub.set_dir(args.hub); sys.path.insert(0, os.path.join(args.hub, "facebookresearch_dinov2_main"))

d = np.load(args.data)
imgs = d["img_test"]
names = ([str(x) for x in d["test_names"]] if "test_names" in d.files
         else [f"image_{i}" for i in range(len(imgs))])   # optional, only used as labels

from ube.load import load_subject
model = load_subject(args.enc, hub=args.hub, device=args.device, n_voxels=d["Y_test"].shape[1])

build_pred_cache(model, imgs, names, args.out, device=args.device,
                 extra=dict(pair_idx=d["pair_idx"]))
