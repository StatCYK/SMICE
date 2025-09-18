#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:5
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

# Load modules
module load gcc/12.2.0-fasrc01
module load python/3.10.12-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01
export PATH=$CONDA_PREFIX/bin:$PATH
export PATH="/n/home13/yongkai/hh-suite/build/bin:/n/home13/yongkai/hh-suite/build/scripts:$PATH"
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
        "${output_dir}" >> "$colab_log" 2>> "$colab_err" &
}

# Main loop
for jobname in "${jobnames[@]}"; do
    for iter in 1 2; do
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

        # Parallel model execution - one model per GPU (0-4)
        declare -a pids
        for model in 1 2 3 4 5; do
            gpu_id=$((model-1))  # Models 1-5 map to GPUs 0-4
            run_model "$jobname" "$iter" "$model" "$gpu_id"
            pids+=($!)
        done

        # Wait for all models to complete
        wait "${pids[@]}"

        echo "ITER${iter}_COMPLETED $(date)" >> "$status_file"
    done

    echo "COMPLETED $(date)" >> "$status_file"
done

echo "All finished"