#!/bin/bash
#SBATCH --job-name=rhythm_hpo
#SBATCH --output=/home/joao.lima/experiments/rhythm_classifier/slurm/out/hpo_%j.out
#SBATCH --error=/home/joao.lima/experiments/rhythm_classifier/slurm/out/hpo_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s,a5000,rtx5000
#SBATCH --time=2-00:00:00
#SBATCH --mem=15G

export HOME=/home/$USER
export WANDB_MODE=offline

source ~/miniconda3/bin/activate
conda activate rtm

CONFIG="${1:?usage: sbatch slurm/hpo.sh config/consolidate/<config>.json [n_trials]}"
N_TRIALS="${2:-80}"

python -m src.hpo \
    --config "$CONFIG" \
    --n_trials "$N_TRIALS"
