# SMICE

This repository contains the code to produce the results in the paper "Uncovering distinct protein conformations using coevolutionary information and AlphaFold" by Yongkai Chen, Samuel W.K. Wong, and S. C. Kou.

We provide two ways to run the code on Linux environments:
* Jupyter notebook, for a quick start that demonstrates SMICE on individual fold-switching proteins
* Slurm cluster setup, recommended for running SMICE on the full benchmark set of fold-switching proteins

For either approach, please begin by following the steps in the **Installation** and **Dataset** sections below.

## Installation

### Basic requirements

Before proceeding, make sure your system meets the following requirements:

- CUDA ≥ 12
- cuDNN ≥ 9
- GCC ≥ 12

Begin by cloning this repository:

```bash
git clone https://github.com/StatCYK/SMICE
cd SMICE
```

### Creating the Python environment

We recommend using **mamba** for faster setup:

```bash
mamba env create -f SMICE.yml
```

Alternatively, **conda** can also be used:

```bash
conda env create -f SMICE.yml
```

*Note*: Environment creation may take ~10–15 minutes. The preferred Python version for SMICE is 3.10.13 and will be installed automatically in the environment.

After the environment is created, activate it:

```bash
mamba activate SMICE
# or
conda activate SMICE
```

### Install external packages

These tools are required to run the experiments. HHsuite and Foldseek can be installed simply using the commands below (replace `conda` with `mamba` below if you created your environment using `mamba`).

- **HHsuite**

  ```bash
  conda install -c conda-forge -c bioconda hhsuite
  ```

  or refer to [https://github.com/soedinglab/hh-suite](https://github.com/soedinglab/hh-suite) for official instructions

- **Foldseek**

  ```bash
  conda install -c conda-forge -c bioconda foldseek
  ```
  
  or refer to [https://github.com/steineggerlab/foldseek](https://github.com/steineggerlab/foldseek) for official instructions


- **LocalColabFold**

  Please follow the installation instructions in the official repository: [https://github.com/YoshitakaMo/localcolabfold](https://github.com/YoshitakaMo/localcolabfold) 


## Dataset

This section prepares the dataset of benchmark fold-switching proteins.

1. Download the ZIP file of MSAs [MSA_cov75_all.zip](https://drive.google.com/file/d/1sTRjkz6UXTvQKDi33I8Xx3jcCd0O8a1S/view?usp=drive_link) and place it into the base repo directory. Then run `unzip MSA_cov75_all.zip` to create a `MSA_cov75_all/` folder inside `SMICE`.

1. Run `unzip pdbs_92.zip` to extract the PDB files of the known conformations.

1. Run `unzip PDB_annotations.txt.zip` to extract the PDB annotations file.


Once these **Installation** and **Dataset** steps are complete, proceed to either the **Jupyter notebook** or **Slurm** section below.

## Quick Start: Running SMICE on Jupyter Notebook

Proceed to open the demo notebook [demo.ipynb](https://github.com/StatCYK/SMICE/blob/master/demo.ipynb) after the **Installation** and **Dataset** steps above have been completed. Ensure that the Python kernel for the SMICE environment created during the **Installation** steps is selected in Jupyter. The notebook uses KaiB (PDB ID: 5jytA) as the default example, which takes about an hour to run on an A100 GPU.

## Full Benchmark: Running SMICE on a Slurm Cluster

The instructions in this section assume access to a Slurm-based high-performance computing (HPC) cluster. Ensure the **Installation** and **Dataset** steps above have been completed.

### Setup and configuration

* Begin by preparing the shell scripts for execution:
  ```bash
  cd bash/benchmark_exp
  chmod +x *.sh
  ```

* See `bash/benchmark_exp/README.md` for instructions on how to prepare all of the `*.slurm` files.

* See `config/README.md` for details on how to set up the configuration file. Then set the paths in `config/config_SMICE_benchmark.json` as appropriate for your system.

### Run SMICE on the full set of benchmark fold-switching proteins

After preparing the shell scripts, Slurm scripts, and configuration files above, execute `./run_SMICE_all.sh` in the `bash/benchmark_exp` directory to submit the jobs to the Slurm cluster.

### Calculate TMscores and create scatterplots

After the jobs have completed, run the following from the base repo directory to calculate TMscores and create scatterplots of TMscores relative to each protein's two known conformations.

```bash
cd experiment/validation
conda activate SMICE
python all_calculate_TMscores.py
```

### Compare with AF-Cluster and Random Sampling
The TMscore results of *AF-Cluster* and *Random Sampling* are saved in `AFclust_random_res.zip`. Run 
```bash
unzip AFclust_random_res.zip
```
in the base repo directory to extract them. Alternatively, to reproduce the results from these methods, refer to [https://github.com/HWaymentSteele/AF_Cluster](https://github.com/HWaymentSteele/AF_Cluster) for installation and implementation.

Continuing in the `experiment/validation` directory, to compare the TMscores of the top predictions, run:

```bash
python compare_TopPred.py
```

Then, to compare the overall prediction accuracy of the ensemble of predictions, run:

```bash
python compare_OverallPred.py
```

### Assess confidence metrics

To analyze the relationships between confidence metrics and TMscores, navigate to the `experiment/analysis` directory and run the corresponding Python script:

```bash
cd experiment/analysis
python confidence_metric_analysis.py
```


### SMICE outputs
The files generated by SMICE are organized as follows:

-   **Job Outputs** are saved in a subfolder (named by the PDB ID) within the directory specified by `base_output_dir` in `config/config_SMICE_benchmark.json`. This includes:
    *   Sampled MSA subsets and PDB structure predictions from the sampling step of SMICE (including both sequential and enhanced sampling)
    *   `RepStructure.zip`, which contains the extracted representative structures
    *   `Clustering_Res`, which contains the detailed clustering results
    *   `outputs_SMICE.json.zip`, which contains the file paths of the structure predictions, the corresponding MSA file paths, and the corresponding confidence scores

-   **Results and Analyses** are saved in the directory specified by `base_result_dir` in `config/config_SMICE_benchmark.json`. This includes summary figures for:
    *   TMscore scatterplots
    *   Method comparison results
    *   Confidence metric analysis
