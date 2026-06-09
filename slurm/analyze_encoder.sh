#!/bin/bash
#SBATCH --job-name=analysis_encoder
#SBATCH --output=/home/joao.lima/experiments/rhythm_classifier/slurm/out/analysis_encoder.out
#SBATCH --error=/home/joao.lima/experiments/rhythm_classifier/slurm/out/analysis_encoder.err
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s,a5000
#SBATCH --time=01:00:00
#SBATCH --mem=15G

export HOME=/home/$USER
export HF_HOME=/home/joao.lima/.cache/huggingface/hub/
export TRANSFORMERS_OFFLINE=1
source ~/miniconda3/bin/activate
conda activate rtm

CHECKPOINT="exp/exp_B1_env_cont_expand/checkpoints/20260608_1615/best_model.pth"

python -m src.analyze_encoder \
    --config config/exp_B1_env_cont_expand.json \
    --checkpoint "$CHECKPOINT" \
    --output_dir exp/exp_B1_env_cont_expand/eval
