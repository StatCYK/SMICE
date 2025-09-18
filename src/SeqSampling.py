import os, time, gc
import re, tempfile
import numpy as np
import random
import string
import pandas as pd
from util_SMICE import save_plot_cluster_msas,run_BSS
import sys
import json

jobname = sys.argv[1] 
lamb= int(sys.argv[2])  # lamb for bayesian sequential sampling
with open('../config/config_SMICE_benchmark.json', 'r') as f:
    config = json.load(f)

base_dir = config["base_dir"]
base_output_dir = config["base_output_dir"]
base_result_dir = config["base_result_dir"]
pdb_seq_file = config["pdb_seq_file"]
MSA_saved_basedir = config["MSA_saved_basedir"]
cov = 75
MSA_saved_dir = f"{MSA_saved_basedir}{jobname}"

save_subMSA_dir = f"{base_output_dir}{jobname}/bss_res/"
if os.path.exists(os.path.join(MSA_saved_dir, "msa/msa.npy")):
    msa = np.load(os.path.join(MSA_saved_dir, "msa/msa.npy"))
    if len(msa)<20:
        print("Warning!! Number of MSA is %d"%len(msa))
    else:
        save_plot_cluster_msas([msa],save_subMSA_dir+"/msa.jpg", sort_by_dist=True)
        run_BSS(msa, os.path.join(MSA_saved_dir, "msa.a3m"), jobname, save_subMSA_dir,[lamb])
else:
    print("MSA of %s not found"%jobname)


