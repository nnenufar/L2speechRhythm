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

CHECKPOINT="/home/joao.lima/experiments/rhythm_classifier/exp/rhythm_encoder_contrastive/checkpoints/20260527_1129/best_model.pth"

python -m src.analyze_encoder \
    --config config/rhythm_encoder_contrastive.json \
    --checkpoint "$CHECKPOINT" \
    --output_dir analysis/rhythm_encoder \
    --num_samples_a 8 \
    --num_utterances_b 6
