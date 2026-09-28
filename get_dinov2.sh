#!/bin/bash
# run this once on a login node (compute nodes usually have no internet)
# it puts the dinov2 backbone in your torch hub cache so training can run offline
set -e
. ./config.sh
mkdir -p "$TORCH_HUB"
$PY - <<PYEOF
import torch
torch.hub.set_dir("$TORCH_HUB")
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)
print("dinov2 cached in", "$TORCH_HUB")
PYEOF
