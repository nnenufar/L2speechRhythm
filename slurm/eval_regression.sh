#!/bin/bash
#SBATCH --job-name=rhythm_eval
#SBATCH --output=/home/joao.lima/experiments/rhythm_classifier/slurm/out/rhythm_eval.out
#SBATCH --error=/home/joao.lima/experiments/rhythm_classifier/slurm/out/rhythm_eval.err
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s,a5000
#SBATCH --time=00:30:00
#SBATCH --mem=15G

export HOME=/home/$USER
export HF_HOME=/home/joao.lima/.cache/huggingface/hub/
export TRANSFORMERS_OFFLINE=1
source ~/miniconda3/bin/activate
conda activate rtm

CHECKPOINT="exp/rhythm_regression_speechocean/checkpoints/20260518_1307/best_model_epoch55.pth"

python -m src.eval_regression \
    --config config/rhythm_regression_speechocean.json \
    --checkpoint "$CHECKPOINT" \
    --split Test
