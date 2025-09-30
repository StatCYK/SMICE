#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4              # Change this to your available GPU count
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH -o ./log/enhanced/log.out
#SBATCH -e ./log/enhanced/errors.err

# Create log directory (absolute path)
mkdir -p ./log/enhanced
LOG_DIR=$(realpath ./log/enhanced)

# Clear log files if they exist
[ -f "$LOG_DIR/log.out" ] && > "$LOG_DIR/log.out"
[ -f "$LOG_DIR/errors.err" ] && > "$LOG_DIR/errors.err"

# Read config values
base_dir=$(jq -r '.base_dir' ../../config/config_SMICE_benchmark.json)
base_output_dir=$(jq -r '.base_output_dir' ../../config/config_SMICE_benchmark.json)
hhsuite_dir=$(jq -r '.hhsuite_dir' ../../config/config_SMICE_benchmark.json)

# Load modules
module load gcc/12.2.0-fasrc01
module load python/3.10.12-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01
export PATH=$CONDA_PREFIX/bin:$PATH
export PATH="${hhsuite_dir}build/bin:${hhsuite_dir}build/scripts:$PATH"
mamba activate SS_AF2

jobnames=("$@")

# Function to run a single model
run_model() {
    local jobname=$1
    local iter=$2
    local model=$3
    local gpu_id=$4

    input_dir="${base_output_dir}${jobname}/enhanced_iter${iter}_res/msa_ss/model_${model}/"
    output_dir="${base_output_dir}${jobname}/enhanced_iter${iter}_res/pdb_ss_colab/model_${model}/"
    mkdir -p "$output_dir"
    
    colab_log="${base_output_dir}${jobname}/enhanced_iter${iter}_res/log/model${model}.log"
    colab_err="${base_output_dir}${jobname}/enhanced_iter${iter}_res/log/model${model}.err"
    
    [ -f "$colab_log" ] && > "$colab_log"
    [ -f "$colab_err" ] && > "$colab_err"

    echo "Running model $model for $jobname iter $iter on GPU $gpu_id at $(date)" >> "$status_file"
    CUDA_VISIBLE_DEVICES="$gpu_id" colabfold_batch \
        --num-relax 50 \
        --random-seed 2 \
        --num-seeds 1 \
        --model-order "${model}" \
        --num-recycle 3 \
        --amber \
        --use-gpu-relax \
        --max-seq 512 \
        --overwrite-existing-results \
        "${input_dir}" \
        "${output_dir}" >> "$colab_log" 2>> "$colab_err"
}

# Get number of available GPUs from SLURM
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$SLURM_GPUS
fi
echo "Detected $NUM_GPUS available GPUs" >> "$LOG_DIR/log.out"

# Main loop
for jobname in "${jobnames[@]}"; do
    status_file="${base_output_dir}${jobname}/enhanced_iter${iter}_res/iter${iter}_status.log"
    [ -f "$status_file" ] && > "$status_file"
    echo "STARTED $(date)" > "$status_file"
    iter_log="${base_output_dir}${jobname}/enhanced_iter${iter}_res/log/enhanced.log"
    iter_err="${base_output_dir}${jobname}/enhanced_iter${iter}_res/log/enhanced.err"
    mkdir -p "${base_output_dir}${jobname}/enhanced_iter${iter}_res/log/"
    [ -f "$iter_log" ] && > "$iter_log"
    [ -f "$iter_err" ] && > "$iter_err"
    echo "Running iteration $iter for $jobname at $(date)" >> "$status_file"

    # Save current directory before cd
    original_dir=$(pwd)

    # Run Python script from base directory
    cd ../../src
    python Iter_enhance_sampling.py "$jobname" "$iter" >> "$iter_log" 2>> "$iter_err"
    cd "$original_dir"

    # Array to track running processes
    declare -A running_pids
    declare -A running_gpus

    # Initialize GPU status
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        running_gpus[$gpu]=0
    done

    # Function to find next available GPU
    get_available_gpu() {
        for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
            if [ ${running_gpus[$gpu]} -eq 0 ]; then
                echo $gpu
                return
            fi
        done
        echo -1  # No available GPU
    }

    # Run models (1-5) with auto GPU allocation
    for model in 1 2 3 4 5; do
        while true; do
            gpu_id=$(get_available_gpu)
            if [ "$gpu_id" -ne -1 ]; then
                running_gpus[$gpu_id]=1
                run_model "$jobname" "$iter" "$model" "$gpu_id" &
                pid=$!
                running_pids[$pid]=$gpu_id
                break
            else
                # Wait for any process to finish
                wait -n
                for pid in "${!running_pids[@]}"; do
                    if ! kill -0 "$pid" 2>/dev/null; then
                        gpu_id=${running_pids[$pid]}
                        running_gpus[$gpu_id]=0
                        unset running_pids[$pid]
                    fi
                done
            fi
        done
    done

    # Wait for remaining processes
    wait


    echo "COMPLETED $(date)" >> "$status_file"
done

echo "All finished"