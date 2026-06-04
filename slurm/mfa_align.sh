#!/bin/bash
#SBATCH --job-name=mfa_align
#SBATCH --output=/home/joao.lima/experiments/rhythm_classifier/slurm/out/mfa_align.out
#SBATCH --error=/home/joao.lima/experiments/rhythm_classifier/slurm/out/mfa_align.err
#SBATCH --ntasks=8
#SBATCH --time=12:00:00
#SBATCH --mem=15G

export HOME=/home/$USER
source ~/miniconda3/bin/activate
conda activate rtm

echo "=== Step 1: Download MFA models ==="
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

echo "=== Step 2: Prepare MFA input ==="
python utils/prepare_mfa_input.py

echo "=== Step 3: Run MFA alignment ==="
mfa align data/speechocean/mfa_input english_us_arpa english_us_arpa \
    data/speechocean/alignments --clean

echo "=== Step 4: Parse alignments to JSON ==="
python utils/parse_alignments.py

echo "=== Done ==="
