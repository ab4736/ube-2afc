import os
import numpy as np
import torch

# caches one predicted brain pattern per candidate image
# this is what makes online scoring fast, at trial time you only do a dot product instead of
# a gpu forward pass

MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def to_batch(imgs_u8, device):
    x = np.stack([((im / 255.0 - MEAN) / STD).transpose(2, 0, 1) for im in imgs_u8])
    return torch.from_numpy(x.astype(np.float32)).to(device)


def predict(model, imgs_u8, device="cuda", batch=16):
    """run the encoder on uint8 images and return (n_images, n_voxels) predictions"""
    nvox = model.voxel_embed.shape[0]
    vind = torch.arange(nvox).unsqueeze(0).to(device)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(imgs_u8), batch):
            x = to_batch(imgs_u8[i:i + batch], device)
            out.append(model(x, vind.repeat(len(x), 1)).float().cpu().numpy())
    return np.concatenate(out)


def build_pred_cache(model, imgs_u8, names, path, device="cuda", extra=None):
    """predict every candidate image once and save it, this is what online scoring reads"""
    P = predict(model, imgs_u8, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    d = dict(preds=P.astype(np.float32), names=np.array([str(n) for n in names]))
    if extra:
        d.update(extra)
    np.savez(path, **d)
    print(f"wrote {path}  {P.shape[0]} images x {P.shape[1]} voxels  "
          f"{os.path.getsize(path) / 1e6:.1f} MB")
    return P


def load_pred_cache(path):
    d = np.load(path, allow_pickle=False)
    return d["preds"].astype(np.float32), [str(s) for s in d["names"]]
