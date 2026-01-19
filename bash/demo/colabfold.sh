
#!/bin/bash

# Load configuration
CONFIG_FILE="./config/config_SMICE_benchmark.json"


# Load required modules
module load gcc/12.2.0-fasrc01 python/3.10.12-fasrc01 cuda/12.4.1-fasrc01 cudnn

# Set up environment
export PATH="/n/kou_lab/yongkai/softwares/localcolabfold/conda/bin:$PATH"

# Parse arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2

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
