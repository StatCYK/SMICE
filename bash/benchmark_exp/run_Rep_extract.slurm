#!/bin/bash
#SBATCH --partition=test
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH -o ./log/RepExtract/log.out
#SBATCH -e ./log/RepExtract/errors.err

# Clear logs only if they exist
[ -f ./log/RepExtract/log.out ] && > ./log/RepExtract/log.out
[ -f ./log/RepExtract/errors.err ] && > ./log/RepExtract/errors.err

base_dir=$(jq -r '.base_dir' ../../config/config_SMICE_benchmark.json)
base_output_dir=$(jq -r '.base_output_dir' ../../config/config_SMICE_benchmark.json)
hhsuite_dir=$(jq -r '.hhsuite_dir' ../../config/config_SMICE_benchmark.json)

# Load required modules and set paths
cd ../../src
module load gcc/12.2.0-fasrc01
module load python/3.10.12-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01
export PATH=$CONDA_PREFIX/bin:$PATH
export PATH="${hhsuite_dir}build/bin:${hhsuite_dir}build/scripts:$PATH"
mamba activate SMICE

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
    python pool_preds.py "$jobname"
    python extract_Rep.py "$jobname"  &

    
done
