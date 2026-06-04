#!/bin/bash
#SBATCH --job-name=rtm
#SBATCH --output=/home/joao.lima/experiments/rhythm_classifier/slurm/out/rtm_l1.out
#SBATCH --error=/home/joao.lima/experiments/rhythm_classifier/slurm/out/rtm_l1.err
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
python -m src.train --config config/cnn_env_LSTM_ATTpool_noCL_noDUR.json
