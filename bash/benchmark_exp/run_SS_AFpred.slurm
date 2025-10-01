#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH -o ./log/SS_AFpred/log.out
#SBATCH -e ./log/SS_AFpred/errors.err

# Clear logs only if they exist
[ -f ./log/SS_AFpred/log.out ] && > ./log/SS_AFpred/log.out
[ -f ./log/SS_AFpred/errors.err ] && > ./log/SS_AFpred/errors.err

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

# Function to process a single parameter set
AF_pred_process_parameters() {
    local jobname=$1
    local n_neigbor=$2
    local lamb=$3
    local gpu_id=$4
    local detailed_log=$5
    local joblog_file=$6
    local loop_start=$(date +%s)
    echo "[$(date)] Processing ${jobname} - lambda ${lamb}, neighbors ${n_neigbor} on GPU ${gpu_id}" | tee -a "${detailed_log}"
    input_dir="${base_output_dir}${jobname}/bss_res/msa_ss_bayes_lamb${lamb}_neighbors${n_neigbor}"
    output_dir="${base_output_dir}${jobname}/bss_res/pdb_ss_bayes_colab_lamb${lamb}_neighbors${n_neigbor}"
    mkdir -p "${output_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" colabfold_batch \
        --num-relax 50 \
        --random-seed 2 \
        --num-seeds 1 \
        --num-recycle 3 \
        --amber \
        --use-gpu-relax \
        --max-seq 512 \
        --overwrite-existing-results \
        "${input_dir}" \
        "${output_dir}" >> "${detailed_log}" 2>> "${detailed_log}"
    
    local loop_end=$(date +%s)
    local loop_duration=$((loop_end - loop_start))
    echo "[$(date)] Completed ${jobname} - lambda ${lamb}, neighbors ${n_neigbor} in $(format_time $loop_duration)" | tee -a "${joblog_file}"
}

# Main processing loop
for jobname in "${jobnames[@]}"; do
    # Check if success file exists
    detailed_log="${base_output_dir}${jobname}/bss_res/detailed_log.txt"
    joblog_file="${base_output_dir}${jobname}/bss_res/joblog.txt"
    
    # Create directories if they don't exist
    mkdir -p "${base_output_dir}${jobname}/bss_res/"

    # Clear existing files
    > "$detailed_log"
    > "$joblog_file"
    
    job_start=$(date +%s)
    job_start_times["$jobname"]=$job_start
    echo "Starting processing for ${jobname} at $(date -d @$job_start)" | tee -a "${joblog_file}"
    
    # Array to hold PIDs for parallel processes
    declare -A pids
    
    # Process each lambda value in parallel (one per GPU)
    for lamb in 0 1 2 3; do
        # Assign GPU ID (0-3) based on lambda value
        gpu_id=$lamb
        # Process neighbor counts sequentially on this GPU
        (
            echo "[$(date)] Starting processing for lamb=${lamb} on GPU ${gpu_id}" | tee -a "${joblog_file}"
            python SeqSampling.py "$jobname" "$lamb" >> "$detailed_log" 2>&1
            for n_neigbor in 10 30; do
                AF_pred_process_parameters "$jobname" "$n_neigbor" "$lamb" "$gpu_id" "$detailed_log" "$joblog_file"
            done
        ) &
        pids[$lamb]=$!
    done
    
    # Wait for all lambda processes to complete
    wait ${pids[0]} ${pids[1]} ${pids[2]} ${pids[3]}
    
    job_end=$(date +%s)
    job_end_times["$jobname"]=$job_end
    job_duration=$((job_end - job_start))
    
    # Update success file
    {
        echo ""
        echo "End Time: $(date -d @$job_end)"
        echo "Total Processing Time: $(format_time $job_duration)"
        echo "Status: COMPLETED SUCCESSFULLY"
        echo ""
    } >> "${joblog_file}"
    echo "Completed processing for ${jobname} in $(format_time $job_duration)" | tee -a "${joblog_file}"
done

# Calculate overall statistics
overall_end=$(date +%s)
overall_duration=$((overall_end - overall_start))

# Final report
echo "" | tee -a "${joblog_file}"
echo "=== SUMMARY REPORT ===" | tee -a "${joblog_file}"
echo "Overall Start Time: $(date -d @$overall_start)" | tee -a "${joblog_file}"
for jobname in "${!job_start_times[@]}"; do
    duration=$((job_end_times["$jobname"] - job_start_times["$jobname"]))
    echo "- ${jobname}: $(format_time $duration)" | tee -a "${joblog_file}"
done
echo "Overall End Time: $(date -d @$overall_end)" | tee -a "${joblog_file}"
echo "Total Elapsed Time: $(format_time $overall_duration)" | tee -a "${joblog_file}"
echo "All jobs completed successfully!" | tee -a "${joblog_file}"
