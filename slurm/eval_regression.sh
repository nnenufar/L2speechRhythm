#!/bin/bash
#SBATCH --job-name=rhythm_eval
#SBATCH --output=/home/joao.lima/experiments/rhythm_classifier/slurm/out/rhythm_eval.out
#SBATCH --error=/home/joao.lima/experiments/rhythm_classifier/slurm/out/rhythm_eval.err
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s,a5000,p5000,rtx8000,rtx5000
#SBATCH --time=00:30:00
#SBATCH --mem=15G

export HOME=/home/$USER
export HF_HOME=/home/joao.lima/.cache/huggingface/hub/
export TRANSFORMERS_OFFLINE=1
source ~/miniconda3/bin/activate
conda activate rtm

CHECKPOINT="exp/exp_A1c_env_fluency_expand/checkpoints/20260608_1414/best_model_epoch316.pth"

python -m src.eval_regression \
    --config config/exp_A1c_env_fluency_expand.json \
    --checkpoint "$CHECKPOINT" \
    --split Test
