The config_SMICE_benchmark.json storing a dictionary as described below

| key               |  Default   |         Description                                                                                                                  |
|---------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------|
| base_dir                  |  `./SMICE/` |          The directory of the SMICE folder                     |
| base_output_dir    |  `./SMICE/outputs_benchmark/` |      The directory of storing the sampled MSA and predicted pdb files               |
| base_result_dir | `./SMICE/results/` | The directory of storing the created figures and other summary files                              |
| meta_path         | `./SMICE/metadata/92PDBs.csv` |     The directory of storing the meta information of the fold-switching proteins                 |
| true_pdb_path     | `./SMICE/pdbs_92/` |  The directory of storing the pdb files of the fold-switching proteins' conformations |
| pdb_seq_file     | `./SMICE/PDB_annotations.txt` |  The directory of storing the sequence information of the fold-switching proteins |
| MSA_saved_basedir | `./SMICE/MSA_cov75_all/`   | The directory of storing the MSA of the fold-switching proteins                                    |
| base_TMscores_output_dir   | `./SMICE/output_TMscores/`   | The directory of storing the TMscore calculation results of the fold-switching proteins    |
| base_dir_AFclust_random   | `./SMICE/AFclust_random_res/`   | The directory of storing the results of AF-Cluster and random sampling |
| hhsuite_dir          | `./hh-suite`   |  The directory of installed hh-suite                                                     |
| foldseek_dir        | `./foldseek/` | The directory of installed foldseek     |
| jobnames        | `["1ceeB",...]` | The list of the job IDs for the fold-switching proteins    |


