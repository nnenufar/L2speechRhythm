#!/bin/bash
#SBATCH --job-name=hpo_dur_trf
#SBATCH --output=/home/joao.lima/experiments/rhythm_classifier/slurm/out/hpo_%j.out
#SBATCH --error=/home/joao.lima/experiments/rhythm_classifier/slurm/out/hpo_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s,a5000,rtx5000
#SBATCH --time=2-00:00:00
#SBATCH --mem=15G

export HOME=/home/$USER
export HF_HOME=/home/joao.lima/.cache/huggingface/hub/
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

source ~/miniconda3/bin/activate
conda activate rtm

python -m src.hpo \
    --config config/consolidate/env_fluency_lstm_bi.json \
    --n_trials 80
