import os, sys
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# loads a small weight file (the kind tools/export_base.py writes) and rebuilds the encoder
# the dinov2 backbone is not in that file, it gets loaded separately, so the download stays small


def _dino(hub=""):
    hub = hub or os.environ.get("UBE_TORCH_HUB", "")
    if hub:
        torch.hub.set_dir(hub)                                    # keeps it working offline
        src = os.path.join(hub, "facebookresearch_dinov2_main")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg",
                          source="github", pretrained=True)


def load_encoder(weights, n_voxels, hub="", device="cuda", voxel_embed=None):
    """build the encoder for a subject with n_voxels voxels and load the pretrained weights

    weights      path to a .pt written by tools/export_base.py
    n_voxels     how many voxels this subject has
    voxel_embed  optional tensor to start the voxel embeddings from (warm start)
    """
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from models.encoder_models import Encoder, encoder_param

    ck = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = ck["config"]

    p = encoder_param(n_voxels)
    for k, v in cfg["param"].items():
        setattr(p, k, v)                                          # inner_ch etc from the checkpoint
    p.num_voxels = n_voxels                                       # this subject, not nsd

    model = Encoder(p, _dino(hub), num_embeds=cfg["num_embeds"],
                    select_layers=cfg["select_layers"], r=cfg["r"], alpha=cfg["alpha"],
                    include_reg_tokens=cfg["include_reg_tokens"],
                    num_reg_tokens=cfg["num_reg_tokens"])

    missing, unexpected = model.load_state_dict(ck["shared"], strict=False)
    # voxel_embed and the backbone are expected to be missing here, anything else is a real problem
    bad = [k for k in missing if k != "voxel_embed" and not k.startswith("embed_model.")]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing {bad}, unexpected {unexpected}")

    if voxel_embed is not None:
        if tuple(voxel_embed.shape) != tuple(model.voxel_embed.shape):
            raise ValueError(f"voxel_embed is {tuple(voxel_embed.shape)} but this subject needs "
                             f"{tuple(model.voxel_embed.shape)}")
        model.voxel_embed.data = voxel_embed.float().clone()

    return model.to(device)


def load_subject(path, hub="", device="cuda", n_voxels=None, base=None):
    """load an encoder that was fit to a subject by train_encoder.py

    handles both formats: the small file (voxel embeddings + which base to use) and the
    old 1.2 GB pickled model, so older checkpoints still work
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        return obj.to(device)                                     # old pickled whole model
    ve = obj["voxel_embed"]
    b = base or obj.get("base")
    if not b or not os.path.exists(b):
        raise FileNotFoundError(
            f"this checkpoint was trained against base '{obj.get('base')}' which is not there now. "
            f"pass the right one with --base")
    return load_encoder(b, n_voxels=n_voxels or ve.shape[0], hub=hub, device=device, voxel_embed=ve)
