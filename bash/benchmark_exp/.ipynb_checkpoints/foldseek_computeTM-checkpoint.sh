#!/bin/bash

# 1. Source the software
foldseek_dir=$(jq -r '.foldseek_dir' ../config/config_SMICE_benchmark.json)

# Remove any trailing slash and add bin to PATH
export PATH="${foldseek_dir%/}/bin:$PATH"

# Define paths for this specific job
CORESETS_DIR="$1"
INPUT_DIR="$2"
OUTPUT_DIR="$3"
RES_FILE="${OUTPUT_DIR}/res.csv"
TMP_DIR="${OUTPUT_DIR}/tmp"

# Check if the input directory exists before running
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: Input directory $INPUT_DIR does not exist. Skipping job $jobname." | tee -a "$LOG_FILE"
    return 1
fi

foldseek easy-search "$INPUT_DIR" \
    "$CORESETS_DIR" \
    "$RES_FILE" \
    "$TMP_DIR" \
    --format-output query,target,alntmscore \
    -v 0 \
    --threads 50

