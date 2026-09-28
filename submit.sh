#!/bin/bash
# sends the pipeline to your cluster's queue
# usage: ./submit.sh NAME
set -e
. ./config.sh
mkdir -p logs data cache checkpoints results
OPTS="--gres=${SLURM_GPUS} --mem=${SLURM_MEM} --time=${SLURM_TIME} --cpus-per-task=${SLURM_CPUS}"
[ -n "$SLURM_ACCOUNT" ]   && OPTS="$OPTS --account=$SLURM_ACCOUNT"
[ -n "$SLURM_PARTITION" ] && OPTS="$OPTS --partition=$SLURM_PARTITION"
echo "sbatch $OPTS run_pipeline.slurm $1"
sbatch $OPTS run_pipeline.slurm "$1"
