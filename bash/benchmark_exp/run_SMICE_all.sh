#!/bin/bash

# Define the list of jobnames
jobnames=($(jq -r '.jobnames[]' ../../config/config_SMICE_benchmark.json))


CHUNK_SIZE=1

# Split jobnames into chunks and submit
for ((i=0; i<${#jobnames[@]}; i+=CHUNK_SIZE)); do
    # Extract chunk of jobnames
    chunk=("${jobnames[@]:i:CHUNK_SIZE}")
    jobname="${chunk[0]}"  # Since CHUNK_SIZE=1
    # Submit the first job and capture its job ID
    jobid1=$(sbatch -J "$jobname" run_SS_AFpred.slurm "${chunk[@]}" | awk '{print $4}')
    echo "Submitted job '$jobname' with job ID $jobid1"
    
    # Submit the second job with dependency on the first
    jobid2=$(sbatch --dependency=afterok:$jobid1 -J "${jobname}_enhancedSamp" run_enhancedSamp.slurm "${chunk[@]}" | awk '{print $4}')
    echo "Submitted dependent job '${jobname}_enhancedSamp' after job ID $jobid1"
    
    # Submit the second job with dependency on the first
    sbatch --dependency=afterok:$jobid2 -J "${jobname}_RepExtract" run_Rep_extract.slurm "${chunk[@]}"
    echo "Submitted dependent job '${jobname}_RepExtract' after job ID $jobid2"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "All jobs have been submitted"
