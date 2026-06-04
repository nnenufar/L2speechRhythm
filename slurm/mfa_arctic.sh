#!/bin/bash
#SBATCH --job-name=mfa_arctic
#SBATCH --output=/home/joao.lima/experiments/rhythm_classifier/slurm/out/mfa_arctic.out
#SBATCH --error=/home/joao.lima/experiments/rhythm_classifier/slurm/out/mfa_arctic.err
#SBATCH --ntasks=8
#SBATCH --time=12:00:00
#SBATCH --mem=30G

export HOME=/home/$USER
source ~/miniconda3/bin/activate
conda activate rtm

echo "=== Step 1: Prepare MFA input ==="
python utils/prepare_mfa_input_arctic.py

echo "=== Step 2: Run MFA alignment ==="
mfa align data/arctic/mfa_input english_us_arpa english_us_arpa \
    data/arctic/alignments --clean

echo "=== Step 3: Parse alignments ==="
python utils/parse_alignments.py \
    --input_dir data/arctic/alignments \
    --output data/arctic/phone_alignments.json

echo "=== Step 4: Build V/C alignments ==="
python utils/build_vc_alignments.py \
    --input data/arctic/phone_alignments.json \
    --output data/arctic/vc_alignments.json

echo "=== Step 5: Build duration-only features ==="
python utils/build_vc_durations.py \
    --input data/arctic/vc_alignments.json \
    --output data/arctic/vc_durations.json

echo "=== Done ==="
