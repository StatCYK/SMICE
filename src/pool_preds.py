import glob
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import random
import string
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.getcwd())  
from util_SMICE import *
import sys
import pickle
import traceback
import plotly.graph_objects as go
import zipfile
import json

with open('../config/config_SMICE_benchmark.json', 'r') as f:
    config = json.load(f)


jobname = sys.argv[1]
metadata_92 = pd.read_csv(config["meta_path"])
base_output_dir = config["base_output_dir"]
base_result_dir = config["base_result_dir"]

true_pdb_path = config["true_pdb_path"]

def process_BSS_jobname(jobname):
    try:
        meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
        outputs=[]
        lamb_list = []
        for lamb in [0,1,2,3]:
            for n_neighbors in [10, 30]:
                pdb_path = base_output_dir+jobname+"/bss_res/pdb_ss_bayes_colab_lamb%d_neighbors%d/"%(lamb,n_neighbors)
                msa_path = base_output_dir+jobname+"/bss_res/msa_ss_bayes_lamb%d_neighbors%d/"%(lamb,n_neighbors)
                msa_file_pattern = f"ss*.a3m"
                msa_files = glob.glob(os.path.join(msa_path, msa_file_pattern))
                num_msa = len(msa_files)
                for model in range(1,6):
                    for ss in range(num_msa):
                        lamb_list.append(lamb)
                        o = {}
                        pattern = f"ss_{ss:02d}*_relaxed*model_{model:01d}*.pdb"
                        pdb_files = glob.glob(os.path.join(pdb_path, pattern))
                        score_file_pattern = f"ss_{ss:02d}*_model_{model:01d}*.json"
                        if len(pdb_files)>0:
                            score_file = glob.glob(os.path.join(pdb_path, score_file_pattern))[0]
                            pdb_file = pdb_files[0]
                            with open(score_file,"r") as f:
                                plddt_scores = pd.read_json(f)
                                avg_pae = np.mean(np.mean(np.array(plddt_scores["pae"])))
                                max_pae = plddt_scores["max_pae"].iloc[0]
                                ptm = plddt_scores["ptm"].iloc[0]
                                avg_plddt = np.mean(plddt_scores["plddt"])/100
                            o.update({'msa_path': f"{msa_path}ss_{ss:02d}.a3m"})
                            o.update({'pdb_path': pdb_file})
                            o.update({'score_path': score_file})
                            o.update({'model': model})
                            o.update({'avg_plddt': avg_plddt})
                            o.update({'avg_pae': avg_pae })
                            o.update({'max_pae': max_pae})
                            o.update({'ptm': ptm})
                            o.update
                            outputs.append(o)  
        outputs = pd.DataFrame.from_records(outputs)
        outputs.to_json(base_output_dir+jobname+f"/bss_res/outputs_bss.json.zip")
    except Exception as e:
        error_msg = f"Error processing {jobname}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
def process_enhanced_jobname(jobname):
    try:
        n_coreset = 5
        if jobname in list(metadata_92['jobnames']):
            meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
            outputs=[]
            for iter in [1,2]:
                save_dir = base_output_dir+f"/{jobname}/enhanced_iter{iter}_res"
                ### with coevol two way
                for model in range(1,6):
                    pdb_path = save_dir+"/pdb_ss_colab/model_%d/"%model
                    pdb_files = {}
                    for ii in range(n_coreset):
                        for jj in range(n_coreset):
                            for set_size in ["020","100"]:
                                o = {}
                                pattern = f"ss_MRF_{ii}_2_MRF_{jj}_size_{set_size}_relaxed*model_{model}*.pdb"
                                score_file_pattern = f"ss_MRF_{ii}_2_MRF_{jj}_size_{set_size}_*model_{model}*.json"
                                pdb_files = glob.glob(os.path.join(pdb_path, pattern))
                                if len(pdb_files)>0:
                                    score_file = glob.glob(os.path.join(pdb_path, score_file_pattern))[0]
                                    pdb_file = pdb_files[0]
                                    with open(score_file,"r") as f:
                                        plddt_scores = pd.read_json(f)
                                    avg_plddt = np.mean(plddt_scores["plddt"])/100
                                    o.update({'msa_path': f"{save_dir}/msa_ss/model_{model}/ss_MRF_{ii}_2_MRF_{jj}_size_{set_size}.a3m"})
                                    o.update({'pdb_path': pdb_file})
                                    o.update({'score_path': score_file})
                                    o.update({'model': model})
                                    o.update({'avg_plddt': avg_plddt})
                                    o.update({'avg_pae': np.mean(np.mean(np.array(plddt_scores["pae"])))})
                                    o.update({'max_pae': plddt_scores["max_pae"].iloc[0]})
                                    o.update({'ptm': plddt_scores["ptm"].iloc[0]})
                                    o.update
                                    outputs.append(o)
            # Combine dataframes
            outputs_bss = pd.read_json(f"{base_output_dir}{jobname}/bss_res/outputs_bss.json.zip")
            outputs = pd.DataFrame.from_records(outputs)
            outputs.to_json(base_output_dir+jobname+f"/outputs_enhanced.json.zip")
            outputs_bss['source'] = "SMICE_SeqSamp"
            outputs['source'] = "SMICE_enhanced"
            # Combine dataframes
            combined_data = pd.concat([outputs_bss,outputs], ignore_index=True)
            combined_data.to_json(base_output_dir+jobname+f"/outputs_SMICE.json.zip")
    except Exception as e:
        error_msg = f"Error processing {jobname}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        #raise  # Re-raise the exception after logging


process_BSS_jobname(jobname)
process_enhanced_jobname(jobname)

