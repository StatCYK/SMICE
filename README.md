# SMICE

This repository contains the code to produce the results in the paper "Uncovering distinct protein conformations using coevolutionary information and AlphaFold" by Yongkai Chen, Samuel W.K. Wong, and S. C. Kou.

## Installation

### Basic requirements (Code was run and tested on LINUX machines.)

* cuda version >=12
* cudnn version >=9
* GCC >= 12
* **python-3.10.12**
 
### Create Python environment
Via mamba (recommended): `mamba env create -f SMICE.yml`

or via conda: `conda env create -f SMICE.yml`

 - This takes 10-15 mins to build

### Install external packages

* HHsuite: refer to [https://github.com/soedinglab/hh-suite](https://github.com/soedinglab/hh-suite) for installation

* colabFold: refer to [https://github.com/sokrypton/ColabFold](https://github.com/sokrypton/ColabFold) for installing localColabFold

* Foldseek: refer to [https://github.com/steineggerlab/foldseek](https://github.com/steineggerlab/foldseek) for installation

* AFcluster and Random Sampling *(optional, only to reproduce results from these methods)*: refer to [https://github.com/HWaymentSteele/AF_Cluster](https://github.com/HWaymentSteele/AF_Cluster) for installation and implementation

## Dataset
- Obtain and unzip the MSA files [MSA_cov75_all.zip](https://drive.google.com/file/d/1sTRjkz6UXTvQKDi33I8Xx3jcCd0O8a1S/view?usp=drive_link) to the base directory

- Unzip the pdb files of the true conformations `pdbs_92.zip`

- Unzip `PDB_annotations.txt.zip`

## Run SMICE on benchmark fold-switching proteins
The instructions in this section assume access to a Slurm-based high-performance computing cluster. For instructions on how to run an example on a local workstation or laptop, please skip to the "Run Locally" section at the end of this README.

`cd bash/benchmark_exp`

`chmod +x *.sh`

See `bash/benchmark_exp/README.md` for instructions on updating all `*.slurm` files

### Setup configuration ###

See `config/README.md` for details of setting up the configuration file

Set the paths in `config/config_SMICE_benchmark.json` accordingly


### Run one demo example ###
`./run_SMICE_exmp.sh`

### Run on all benchmark proteins ###
`./run_SMICE_all.sh`

## Results analysis ##

### Calculate TMscores and create scatter plots
`cd experiment/validation`

`conda activate SMICE`

`python all_calculate_TMscores.py`

### Compare with AF-Cluster and Random Sampling
We use the TMscore results of AF-Cluster and Random Sampling as saved in `AFclust_random_res.zip`.

`unzip AFclust_random_res.zip`

Compare the TMscores of the top predictions:

`python compare_TopPred.py` 

Compare the overall prediction accuracy of the prediction sets:

`python compare_OverallPred.py` 

### Assess confidence metrics

Analyze relationships between confidence metrics and TMscores:

`cd experiment/analysis`

`python confidence_metric_analysis.py`


## The outputs
All generated files are organized as follows:

*   **Job Outputs** are saved in a subfolder (named by the PDB ID) within the directory specified by `base_output_dir` in `config/config_SMICE_benchmark.json`. This includes:
    *   Sampled MSA subsets and predicted PDB files for different steps of SMICE
    *   `RepStructure.zip` (contains the extracted representative structures)
    *   `Clustering_Res` (contains the detailed clustering results)
    *   `outputs_SMICE.json.zip` (the file paths of the predicted PDB files, the corresponding MSA file paths, and the corresponding confidence scores)

*   **Results & Analysis** are saved in the directory specified by `base_result_dir` in `config/config_SMICE_benchmark.json`. This includes summary figures for:
    *   TM-score scatter plots
    *   Comparison results
    *   Confidence metric analysis
 
## Run Locally
The [SMICE code](https://drive.google.com/drive/folders/1i9BZG2pvLqs_Bz1EXdcldZiKEtiAted-?usp=sharing) for running locally is provided, which can be downloaded and unzipped on a local workstation or laptop.

* Follow the same "Installation" and "Dataset" instructions as above.

* Set the paths in `config/config_SMICE_benchmark.json` to match your local folder setup.

* Activate the `SMICE` environment, and set your `CONDA_PREFIX` accordingly in line 33 of `run_SMICE_exmp_local.sh` in the `bash/benchmark_exp` directory.

* Colabfold will use the GPU if available. Otherwise, CPU-based colabFold will run ~10 times slower on a system without an Nvidia GPU/CUDA driver, so we reduce the MSA sampling size of SMICE for demonstration purposes. Note that this could take over 20 hours on a desktop computer, but will run significantly faster if a recent GPU is available. This demonstration version for an example protein can be executed by running the following script from the `bash/benchmark_exp` directory:

    `./run_SMICE_exmp_local_toy.sh`

* To instead run the full version of SMICE for this example, run the following script from the `bash/benchmark_exp` directory:

    `./run_SMICE_exmp_local.sh`
