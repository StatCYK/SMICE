# SMICE

## Set environments and install packages

cuda version >=12

cudnn version >=9

GCC >= 12

#### python
conda env create -f SMICE.yml



#### colabFold

## Dataset
Upload and unzip MSA_cov75_all.zip to the "dataset" directory

unzip pdbs_92.zip

## run toy example


## Run Experiment on 92 foldswitching proteins

### Step1. Run Sequential Sampling
cd code

conda activate SMICE

python BSS_parallel.py

### Step2. Run AF2 prediction on MSA subsets from sequential sampling
cd code/bash

chmod +x run_colabFold_bss.sh

./run_colabFold_bss.sh

### Step3. Run enhanced sampling

### Calculate TMscores and Visualization
cd code

python calculate_TMscores_BSS.py
