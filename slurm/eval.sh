#!/bin/bash
#SBATCH --job-name=rhythm_eval
#SBATCH --output=/home/joao.lima/experiments/rhythm_classifier/slurm/out/eval_%j.out
#SBATCH --error=/home/joao.lima/experiments/rhythm_classifier/slurm/out/eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s,a5000,rtx8000,rtx5000
#SBATCH --time=00:30:00
#SBATCH --mem=15G

export HOME=/home/$USER
export WANDB_MODE=offline
source ~/miniconda3/bin/activate
conda activate rtm

CONFIG="${1:?usage: sbatch slurm/eval.sh config/consolidate/<config>.json /path/to/checkpoint.pth [split]}"
CHECKPOINT="${2:?usage: sbatch slurm/eval.sh config/consolidate/<config>.json /path/to/checkpoint.pth [split]}"
SPLIT="${3:-Test}"

python -m src.eval_regression \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT"
