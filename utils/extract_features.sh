#!/usr/bin/env bash
set -euo pipefail

WAVS_DIR="${1:?usage: bash utils/extract_features.sh /path/to/wavs [output_dir] [envelope_method]}"
OUTPUT_DIR="${2:-data/speechocean}"
ENVELOPE_METHOD="${3:-bark}"

if [ ! -d speech_feature_extractor ]; then
    echo "Missing speech_feature_extractor submodule. Run:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python speech_feature_extractor/src/main.py \
    -in "$WAVS_DIR" \
    -out "$OUTPUT_DIR" \
    -env "$ENVELOPE_METHOD" \
    -sr 16000

if [ -f "$OUTPUT_DIR/feats.lmdb" ]; then
    mv -f "$OUTPUT_DIR/feats.lmdb" "$OUTPUT_DIR/rtm_feats.lmdb"
fi

echo "Feature extraction complete."
echo "LMDB: $OUTPUT_DIR/rtm_feats.lmdb"
