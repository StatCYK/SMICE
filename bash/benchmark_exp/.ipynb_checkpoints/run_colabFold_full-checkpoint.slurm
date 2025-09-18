#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4              # Change this to your available GPU count
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH -o ./log/full/log.out
#SBATCH -e ./log/full/errors.err


# Create log directory (absolute path)
mkdir -p ./log/full
LOG_DIR=$(realpath ./log/full)

# Clear log files if they exist
[ -f "$LOG_DIR/log.out" ] && > "$LOG_DIR/log.out"
[ -f "$LOG_DIR/errors.err" ] && > "$LOG_DIR/errors.err"

# Read config values
base_dir=$(jq -r '.base_dir' ../../config/config_SMICE_benchmark.json)
base_output_dir=$(jq -r '.base_output_dir' ../../config/config_SMICE_benchmark.json)
hhsuite_dir=$(jq -r '.hhsuite_dir' ../../config/config_SMICE_benchmark.json)

MSA_saved_basedir=$(jq -r '.MSA_saved_basedir' ../../config/config_SMICE_benchmark.json)

# Load required modules and set paths
module load gcc/12.2.0-fasrc01
module load python/3.10.12-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.1.1.17_cuda12-fasrc01
export PATH=$CONDA_PREFIX/bin:$PATH
export PATH="${hhsuite_dir}build/bin:${hhsuite_dir}build/scripts:$PATH"
mamba activate SS_AF2


# Initialize variables
jobnames=("$@")
for jobname in "${jobnames[@]}"; do
    input_dir="${MSA_saved_basedir}${jobname}/"
    output_dir="${MSA_saved_basedir}${jobname}/pdb/"
    mkdir -p "${output_dir}"
    colabfold_batch \
        --num-relax 50 \
        --random-seed 2 \
        --num-seeds 1 \
        --num-recycle 3 \
        --amber \
        --use-gpu-relax \
        --max-seq 512 \
        "${input_dir}" \
        "${output_dir}"
done
