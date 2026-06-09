#!/bin/bash
#SBATCH --job-name=D1
#SBATCH --output=/home/joao.lima/experiments/rhythm_classifier/slurm/out/rhythm_ft_%j.out
#SBATCH --error=/home/joao.lima/experiments/rhythm_classifier/slurm/out/rhythm_ft_%j.err
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s,a5000,rtx5000,rtx8000
#SBATCH --time=1-00:00:00
#SBATCH --mem=15G

export HOME=/home/$USER
export HF_HOME=/home/joao.lima/.cache/huggingface/hub/
export TRANSFORMERS_OFFLINE=1
source ~/miniconda3/bin/activate
conda activate rtm

python -m src.train --config config/exp_D1c_env_fluency_expand.json

