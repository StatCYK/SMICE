
#!/bin/bash

# Exit on error and undefined variables
set -euo pipefail

# Parse arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"
LOCALCOLABFOLD_DIR="$3"

# Debug info
echo "========================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "ColabFold directory: $LOCALCOLABFOLD_DIR"
echo "========================================="

# Set up environment
export PATH="$LOCALCOLABFOLD_DIR/conda/bin:$PATH"

shift 3

# Set models (default or provided)
MODELS=("$@")
if [[ $# -eq 0 ]]; then
    MODELS=(1 2 3 4 5)
    echo "Using default models: 1,2,3,4,5"
else
    echo "Using specified models: ${MODELS[*]}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Output directory created/verified: $OUTPUT_DIR"

# Create comma-separated string for model-order
MODEL_ORDER_STRING=$(IFS=,; echo "${MODELS[*]}")
echo "Running models in order: $MODEL_ORDER_STRING"
echo "Starting ColabFold processing..."
echo "========================================="

# Run timestamp for tracking
START_TIME=$(date +%s)

# Run colabfold_batch with enhanced output
colabfold_batch \
    --num-relax 50 \
    --random-seed 2 \
    --num-seeds 1 \
    --num-recycle 3 \
    --amber \
    --use-gpu-relax \
    --max-seq 512 \
    --model-order "$MODEL_ORDER_STRING" \
    "${INPUT_DIR}" \
    "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}/colabfold.log"

EXIT_CODE=$?

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo "========================================="
echo "Processing completed"
echo "Total runtime: $((RUNTIME / 60)) minutes $((RUNTIME % 60)) seconds"

# Check exit status
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "SUCCESS: ColabFold completed successfully"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo "ERROR: ColabFold processing failed with exit code: $EXIT_CODE"
    echo "Check log file for details: ${OUTPUT_DIR}/colabfold.log"
    exit $EXIT_CODE
fi
