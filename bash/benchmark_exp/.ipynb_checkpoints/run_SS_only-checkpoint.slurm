#!/bin/bash
#SBATCH --partition=test
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH -o ./log/SS_AFpred/log.out
#SBATCH -e ./log/SS_AFpred/errors.err

# Clear logs only if they exist
[ -f ./log/SS_AFpred/log.out ] && > ./log/SS_AFpred/log.out
[ -f ./log/SS_AFpred/errors.err ] && > ./log/SS_AFpred/errors.err

base_dir=$(jq -r '.base_dir' ../../config/config_SMICE_benchmark.json)
base_output_dir=$(jq -r '.base_output_dir' ../../config/config_SMICE_benchmark.json)

# Load required modules and set paths
cd ../../src
module load gcc/12.2.0-fasrc01
module load python/3.10.12-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01
export PATH=$CONDA_PREFIX/bin:$PATH
export PATH="/n/home13/yongkai/hh-suite/build/bin:/n/home13/yongkai/hh-suite/build/scripts:$PATH"
mamba activate SS_AF2

# Initialize variables
jobnames=("$@")
declare -A job_start_times
declare -A job_end_times
overall_start=$(date +%s)

# Function to calculate time difference
format_time() {
    local seconds=$1
    local hours=$((seconds/3600))
    local minutes=$(( (seconds%3600)/60 ))
    local secs=$((seconds%60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# Main processing loop
for jobname in "${jobnames[@]}"; do
    # Process each lambda value in parallel (one per GPU)
    for lamb in 0 1 2 3; do
        # Assign GPU ID (0-3) based on lambda value
        gpu_id=$lamb
        # Process neighbor counts sequentially on this GPU
        python SeqSampling.py "$jobname" "$lamb"  &
    done
    
done
