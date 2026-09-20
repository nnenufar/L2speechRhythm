#!/bin/bash
#
# Build VC (vocalic/intervocalic) features from phone alignments.
#
# Runs in sequence: build_vc_alignments → add_phone_durs_z → build_vc_features
#
# Usage:
#     bash utils/build_vc_pipeline.sh [data_dir]
#
#   data_dir: Base directory containing phone_alignments.json.
#             Defaults to data/speechocean.
#
# Input:  <data_dir>/phone_alignments.json
# Output:
#   <data_dir>/vc_alignments.json
#   <data_dir>/phone_duration_stats.json
#   <data_dir>/vc_features.json
#

set -euo pipefail

DATA_DIR="${1:-data/speechocean}"

echo "=== Build VC features pipeline ==="
echo "  Data dir: $DATA_DIR"

PHONE_ALIGNMENTS="$DATA_DIR/phone_alignments.json"
VC_ALIGNMENTS="$DATA_DIR/vc_alignments.json"
PHONE_STATS="$DATA_DIR/phone_duration_stats.json"
VC_FEATURES="$DATA_DIR/vc_features.json"

echo "=== Step 1/3: Build V/C segments ==="
python utils/build_vc_alignments.py \
    --input "$PHONE_ALIGNMENTS" \
    --output "$VC_ALIGNMENTS"

echo "=== Step 2/3: Add z-scored phone durations ==="
python utils/add_phone_durs_z.py \
    --phone_alignments "$PHONE_ALIGNMENTS" \
    --vc_alignments "$VC_ALIGNMENTS" \
    --output "$VC_ALIGNMENTS" \
    --stats_output "$PHONE_STATS"

echo "=== Step 3/3: Build tokenised VC features ==="
python utils/build_vc_features.py \
    --input "$VC_ALIGNMENTS" \
    --output "$VC_FEATURES"

echo "=== Done ==="
echo "  vc_alignments.json     -> $VC_ALIGNMENTS"
echo "  phone_duration_stats.json -> $PHONE_STATS"
echo "  vc_features.json       -> $VC_FEATURES"
