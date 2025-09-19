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




## Run Experiment on 92 foldswitching proteins
### Setup configuration ###
Change the paths in the ./config/config_SMICE_benchmark.json

### Run all ###
cd bash/benchmark_exp
chmod +x *.sh
./run_SMICE_all.sh

## Analysis ##


### Calculate TMscores and Visualization
cd code

