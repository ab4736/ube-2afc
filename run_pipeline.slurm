#!/bin/bash
#SBATCH --job-name=ube_2afc
#SBATCH --account=knorman
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log

# runs everything for one dataset: check inputs -> train encoder -> 2AFC/CPD
# usage (from inside ube_2afc/):  sbatch run_pipeline.slurm NAME
#   NAME = sub-06 or sub-07 for Bixby (input file gets built automatically)
#   NAME = anything else if you already made data/NAME.npz with make_inputs.py

NAME=${1:-sub-06}
PY=/scratch/gpfs/KNORMAN/qanguyen/Brain-IT/brain-it/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $SLURM_SUBMIT_DIR
echo "$NAME  $(hostname)  $(date)"

if [ ! -f data/$NAME.npz ]; then
  if [ "$NAME" = "sub-06" ] || [ "$NAME" = "sub-07" ]; then
    $PY -u make_inputs_bixby.py --sub $NAME || exit 1
  else
    echo "PROBLEM: data/$NAME.npz doesn't exist. Make it first with make_inputs.py (see README)."
    exit 1
  fi
fi

$PY -u check_inputs.py data/$NAME.npz || { echo "stopping, fix the input file first"; exit 1; }
$PY -u train_encoder.py --data data/$NAME.npz --tag ${NAME}_infonce --lam 1 --went 0.1 || exit 1
$PY -u eval_2afc.py --data data/$NAME.npz --enc checkpoints/${NAME}_infonce.pth || exit 1

echo "done $(date)"
