#!/bin/bash
# gets everything you need from hugging face into ./checkpoints
#
#   ./download_checkpoints.sh              the encoder weights and the dinov2 backbone
#   WITH_WARMSTART=1 ./download_checkpoints.sh    also the nsd voxel embeddings (324 MB)
#   WITH_NSD=1 ./download_checkpoints.sh          also the nsd training data (17 GB, only if
#                                                 you want to pretrain a base yourself)
set -e
REPO=${UBE_HF_REPO:-ab4736/ube-2afc}
OUT=${1:-./checkpoints}
mkdir -p "$OUT"

# the encoder weights, small. the backbone is big but you need it, and grabbing it here means
# you never depend on torch.hub reaching the internet from a compute node
files="ube_base_infonce.pt ube_base_recon.pt ube_base_original.pt dinov2_vitl14_reg4_pretrain.pth"
[ "$WITH_WARMSTART" = "1" ] && files="$files nsd_voxel_embed_infonce.pt"

for f in $files; do
  if [ -f "$OUT/$f" ]; then
    echo "already have $f"
  else
    echo "downloading $f"
    curl -fL --progress-bar -o "$OUT/$f" "https://huggingface.co/$REPO/resolve/main/$f"
  fi
done

if [ "$WITH_NSD" = "1" ]; then
  NSD=${NSD_DIR:-./nsd_data}
  mkdir -p "$NSD"
  echo
  echo "downloading the nsd training data into $NSD, this is 17 GB and will take a while"
  for f in fmri_v2.npz num_voxels_all_subjects.npy all_images_v2_224.npy; do
    if [ -f "$NSD/$f" ]; then
      echo "already have $f"
    else
      echo "downloading $f"
      curl -fL --progress-bar -o "$NSD/$f" "https://huggingface.co/$REPO/resolve/main/nsd_data/$f"
    fi
  done
fi

echo
echo "done, files are in $OUT:"
ls -la "$OUT"
echo
echo "if the repo is private you need to be logged in first:"
echo "  pip install huggingface_hub && huggingface-cli login"
echo "or download the files in a browser and drop them in $OUT"
