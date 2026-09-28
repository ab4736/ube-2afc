# fill this in once, everything else reads from here
# nothing in this repo assumes any particular cluster

# python that has torch, numpy, scipy, pillow, nibabel (see requirements.txt)
# if you made a conda env called ube, this is usually something like
#   PY=$HOME/.conda/envs/ube/bin/python
PY=python

# where you downloaded the pretrained weights to (run ./download_checkpoints.sh)
CKPT_DIR=./checkpoints

# which pretrained base to use
# ube_base_infonce.pt   the contrastive one, better at telling near identical images apart
# ube_base_recon.pt     reconstruction only, the matched baseline
BASE=$CKPT_DIR/ube_base_infonce.pt

# torch hub cache, so dinov2 does not need internet on a compute node
# download it once on a login node, then point here
TORCH_HUB=$HOME/.cache/torch/hub

# slurm settings for your cluster, leave blank if your cluster does not need them
SLURM_ACCOUNT=
SLURM_PARTITION=
SLURM_GPUS=gpu:1
SLURM_MEM=96G
SLURM_TIME=02:00:00
SLURM_CPUS=8
