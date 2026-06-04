#!/bin/bash
#SBATCH --job-name=dur_cont
#SBATCH --output=/home/joao.lima/experiments/rhythm_classifier/slurm/out/dur_cont.out
#SBATCH --error=/home/joao.lima/experiments/rhythm_classifier/slurm/out/dur_cont.err
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s,a5000
#SBATCH --time=2-00:00:00
#SBATCH --mem=15G

export HOME=/home/$USER
export HF_HOME=/home/joao.lima/.cache/huggingface/hub/
export TRANSFORMERS_OFFLINE=1
source ~/miniconda3/bin/activate
conda activate rtm

python -m src.train_contrastive_duration --config config/duration_contrastive_arctic.json
