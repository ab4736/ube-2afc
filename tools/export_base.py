import os, sys, json, argparse, hashlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# turns one of the big pickled encoders into small weight files anyone can load
# the pickle is ~1.5 GB because it carries the whole frozen dinov2 backbone plus the
# nsd voxel embeddings, neither of which a new subject needs

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)          # the pickled Encoder to convert
ap.add_argument("--out", required=True)           # where to write the small file
ap.add_argument("--hub", default=os.environ.get("UBE_TORCH_HUB", ""))
ap.add_argument("--save_voxel_embed", default="")  # optional, for warm starting
ap.add_argument("--verify", action="store_true")   # rebuild and compare predictions
ap.add_argument("--device", default="cpu")        # verify needs cuda, xformers attention is gpu only
args = ap.parse_args()

if args.hub:
    torch.hub.set_dir(args.hub)
    sys.path.insert(0, os.path.join(args.hub, "facebookresearch_dinov2_main"))

from models.encoder_models import Encoder, encoder_param

m = torch.load(args.ckpt, map_location="cpu", weights_only=False)
sd = m.state_dict()

# the backbone is stock dinov2, so check that before deciding not to ship it
fresh = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", source="github", pretrained=True)
fsd = fresh.state_dict()
same, diff = 0, []
for k, v in fsd.items():
    ck = sd.get("embed_model." + k)
    if ck is None:
        diff.append(k)
    elif torch.allclose(ck.float(), v.float(), atol=1e-6):
        same += 1
    else:
        diff.append(k)
print(f"dinov2 tensors identical to stock: {same}/{len(fsd)}  differing: {len(diff)}")
if diff[:3]:
    print("  first few that differ:", diff[:3])
backbone_is_stock = len(diff) == 0

shared = {k: v for k, v in sd.items() if not k.startswith("embed_model.") and k != "voxel_embed"}
cfg = dict(
    param={k: v for k, v in vars(m.param).items()},
    num_embeds=m.num_embeds,
    select_layers=list(m.select_layers),
    r=m.r,
    alpha=float(m.lora_scale * m.r),
    include_reg_tokens=bool(m.include_reg_tokens),
    num_reg_tokens=int(m.num_reg_tokens),
    backbone="dinov2_vitl14_reg",
    backbone_is_stock=backbone_is_stock,
    source_checkpoint=os.path.basename(args.ckpt),
    nsd_voxel_embed_shape=list(m.voxel_embed.shape),
)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save(dict(config=cfg, shared=shared), args.out)
mb = os.path.getsize(args.out) / 1e6
print(f"wrote {args.out}  {mb:.1f} MB  ({len(shared)} tensors)")
print("config:", json.dumps({k: v for k, v in cfg.items() if k != "param"}))

if args.save_voxel_embed:
    torch.save(dict(voxel_embed=m.voxel_embed.detach().cpu()), args.save_voxel_embed)
    print(f"wrote {args.save_voxel_embed}  "
          f"{os.path.getsize(args.save_voxel_embed) / 1e6:.1f} MB  (for warm starting)")

if args.verify:
    # rebuild from the small file and check it predicts the same thing as the pickle
    sys.path.insert(0, HERE)
    from ube.load import load_encoder
    nvox = m.voxel_embed.shape[0]
    dev = args.device
    rebuilt = load_encoder(args.out, n_voxels=nvox, hub=args.hub, device=dev)
    rebuilt.voxel_embed.data = m.voxel_embed.detach().clone().to(args.device)  # same voxels for a fair check
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 224, 224, generator=g)
    vind = torch.arange(256).unsqueeze(0).repeat(2, 1)  # small voxel slice keeps this quick
    m = m.to(dev); x = x.to(dev); vind = vind.to(dev)
    m.eval(); rebuilt.eval()
    with torch.no_grad():
        a = m(x, vind)
        b = rebuilt(x, vind)
    d = (a - b).abs().max().item()
    print(f"max abs difference pickle vs rebuilt: {d:.3e}")
    print("VERIFY PASS" if d < 1e-4 else "VERIFY FAIL")
