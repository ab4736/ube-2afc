#!/bin/bash
#SBATCH --job-name=ube_2afc
#SBATCH --output=logs/%x_%j.log
# no account or partition here on purpose, submit.sh adds whatever your cluster needs

# checks your input file, fits the encoder to the subject, then scores 2afc and cpd
# do not run this directly, use:  ./submit.sh NAME

set -e
cd ${SLURM_SUBMIT_DIR:-.}
. ./config.sh
NAME=${1:?usage: ./submit.sh NAME}
export UBE_TORCH_HUB=$TORCH_HUB
echo "$NAME on $(hostname) at $(date)"

if [ ! -f data/$NAME.npz ]; then
  echo "PROBLEM: data/$NAME.npz does not exist. make it first with make_inputs.py, see the readme."
  exit 1
fi
if [ ! -f "$BASE" ]; then
  echo "PROBLEM: pretrained base $BASE is missing. run ./download_checkpoints.sh"
  exit 1
fi

$PY -u check_inputs.py data/$NAME.npz
$PY -u train_encoder.py --data data/$NAME.npz --tag ${NAME}_infonce --base "$BASE" --lam 1 --went 0.1
$PY -u eval_2afc.py     --data data/$NAME.npz --enc checkpoints/${NAME}_infonce.pth
$PY -u make_pred_cache.py --data data/$NAME.npz --enc checkpoints/${NAME}_infonce.pth \
    --out cache/${NAME}_preds.npz          # so the online scoring is ready to go
echo "done $(date)"
