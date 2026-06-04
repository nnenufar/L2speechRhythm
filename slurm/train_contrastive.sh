#!/bin/bash
#SBATCH --job-name=enc_r
#SBATCH --output=/home/joao.lima/experiments/rhythm_classifier/slurm/out/enc_r.out
#SBATCH --error=/home/joao.lima/experiments/rhythm_classifier/slurm/out/enc_r.err
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --time=2-00:00:00
#SBATCH --mem=30G

export HOME=/home/$USER
export HF_HOME=/home/joao.lima/.cache/huggingface/hub/
export TRANSFORMERS_OFFLINE=1
source ~/miniconda3/bin/activate
conda activate rtm

python -m src.train_contrastive --config config/rhythm_encoder_contrastive.json
