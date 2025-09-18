#!/bin/bash

# Define the list of jobnames
jobnames=($(jq -r '.jobnames[]' ../../config/config_SMICE_benchmark.json))


# Maximum number of batch jobs to run in parallel (each handles 1 sequence)
MAX_JOBS=4
CHUNK_SIZE=22

# Split jobnames into chunks and submit
for ((i=0; i<${#jobnames[@]}; i+=CHUNK_SIZE)); do
    # Extract chunk of jobnames
    chunk=("${jobnames[@]:i:CHUNK_SIZE}")
    jobname="${chunk[0]}"  # Since CHUNK_SIZE=1
    
    # Submit the first job and capture its job ID
    jobid1=$(sbatch -J "$jobname" run_SS_only.slurm "${chunk[@]}" | awk '{print $4}')
    echo "Submitted job '$jobname' with job ID $jobid1"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "All jobs have been submitted"
