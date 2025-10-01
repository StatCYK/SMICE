# SMICE

## Set environments and install packages

cuda version >=12

cudnn version >=9

GCC >= 12

#### python
`conda env create -f SMICE.yml`

#### HHsuite
check this link [https://github.com/soedinglab/hh-suite](https://github.com/soedinglab/hh-suite) for installation

#### colabFold
check this link [https://github.com/sokrypton/ColabFold](https://github.com/sokrypton/ColabFold) for installation

#### Foldseek
check this link [https://github.com/steineggerlab/foldseek](https://github.com/steineggerlab/foldseek) for installation

#### AFcluster and Random Sampling
check this link [https://github.com/HWaymentSteele/AF_Cluster](https://github.com/HWaymentSteele/AF_Cluster) for installation and implementation


## Dataset
Upload and unzip MSA_cov75_all.zip to the "dataset" directory

unzip pdbs_92.zip

## Run Experiment on benchmark foldswitching proteins
`cd bash/benchmark_exp`

`chmod +x *.sh`
### Setup configuration ###
Change the paths in the `./config/config_SMICE_benchmark.json`

Check the `./config/README.md` for the details of the configuration file
### Run demo example ###
`./run_SMICE_exmp.sh`

### Run all benchmark fold-switching proteins ###
`./run_SMICE_all.sh`

## Validation ##

### Calculate TMscores and plot the scattering plots
`cd experiment/validation`

`conda activate SMICE`

`python all_calculate_TMscores.py`

### compare against AF-Cluster and Random Sampling

Compare the TMscores of the top predictions

`python compare_TopPred.py` 

Compare the overall prediction accuracy of the prediction set

`python compare_OverallPred.py` 

## Analysis ##

Analyze the confidence metric against the TMscores
`cd experiment/analysis`

`python confidence_metric_analysis.py`
