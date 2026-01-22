
#!/bin/bash


# Parse arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"
localcolabfold_DIR="$3"

# Set up environment
export PATH="$localcolabfold/conda/bin:$PATH"


shift 3

# Set models (default or provided)
MODELS=("$@")
[[ $# -eq 0 ]] && MODELS=(1 2 3 4 5)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# # Process each model sequentially
# for model in "${MODELS[@]}"; do
#     echo "Processing model $model..."
 # Convert array to comma-separated string for --model-order
MODEL_ORDER_STRING=$(IFS=,; echo "${MODELS[*]}")

echo "Running all models: $MODEL_ORDER_STRING"

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
    "${OUTPUT_DIR}"

# Check exit status
if [[ $? -eq 0 ]]; then
    echo "Model $model completed successfully"
else
    echo "WARNING: Model $model processing failed"
fi
    
# done
