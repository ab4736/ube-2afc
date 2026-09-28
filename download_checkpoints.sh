#!/bin/bash
# grabs the pretrained weights from hugging face into ./checkpoints
# the base files are about 10 MB each because the dinov2 backbone is not inside them,
# it gets downloaded separately by torch hub the first time you run anything
set -e
REPO=${UBE_HF_REPO:-ab4736/ube-2afc}
OUT=${1:-./checkpoints}
mkdir -p "$OUT"

files="ube_base_infonce.pt ube_base_recon.pt ube_base_original.pt"
if [ "$WITH_WARMSTART" = "1" ]; then
  files="$files nsd_voxel_embed_infonce.pt"   # 324 MB, only needed for warm starting
fi

for f in $files; do
  echo "downloading $f"
  curl -fL --progress-bar -o "$OUT/$f" "https://huggingface.co/$REPO/resolve/main/$f"
done
echo
echo "done, files are in $OUT:"
ls -la "$OUT"
echo
echo "if these are private you need to be logged in, either"
echo "  huggingface-cli login     (then rerun this)"
echo "or download them in a browser and drop them in $OUT"
