import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
from util_SMICE import *
import sys
import matplotlib.pyplot as plt
import pickle
import re
import glob
import subprocess
import random

np.random.seed(123)
random.seed(123)

jobname = sys.argv[1]
iter = int(sys.argv[2])

import sys
import json

with open('../config/config_SMICE_benchmark.json', 'r') as f:
    config = json.load(f)

metadata_92 = pd.read_csv(config["meta_path"])
base_dir = config["base_dir"]
base_output_dir = config["base_output_dir"]
base_result_dir = config["base_result_dir"]
pdb_seq_file = config["pdb_seq_file"]
MSA_saved_basedir = config["MSA_saved_basedir"]
cov = 75
MSA_saved_dir = f"{MSA_saved_basedir}{jobname}"
job_base_dir = f"{base_output_dir}{jobname}"
if iter == 1:
    saved_dir = os.path.join(job_base_dir, "bss_res")
    save_dir = os.path.join(job_base_dir, "enhanced_iter1_res")
else:
    saved_dir = os.path.join(job_base_dir, f"enhanced_iter{iter-1}_res")
    save_dir = os.path.join(job_base_dir, f"enhanced_iter{iter}_res")

meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
sequence = meta_info['sequences']
fsr_seq = meta_info["Sequence of fold-switching region"]#sequence#
aa_order = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20, '-': 20}
IDs,seqs = load_fasta(os.path.join(MSA_saved_dir, "msa.a3m"))
seqs =[remove_insertions(seq).upper() for seq in seqs]
msa = np.array([[aa_order[aa] for aa in seq] for seq in seqs])
msa_df = mk_msa_df(msa)

tf.reset_default_graph()
mrf_one = GREMLIN(msa_df,opt_iter=100)

N = len(IDs)
if N > 300:
    samp_sizes = [20]#[20,100]
else:
    samp_sizes = [20]

pdb_files = {}
save_msa_ss_dirs = {}
run_coevol = True
run_marginal = False
for model in range(1,6):
    save_msa_ss_dir = save_dir+"/msa_ss/model_%d"%model
    save_msa_fig_dir = save_dir+"/res_fig/model_%d"%model
    os.makedirs(save_msa_ss_dir, exist_ok=True)
    os.makedirs(save_msa_fig_dir, exist_ok=True)
    msa_end = "_relaxed_"
    if iter ==1:
        pattern = f"**/*_relaxed*model_{model}*.pdb"
    else:
        pattern = f"**/**/*_relaxed*model_{model}*.pdb"
    pdb_files_model =glob.glob(os.path.join(saved_dir, pattern))
    save_msa_ss_dirs_model= [re.sub("_colab","",re.sub("pdb", "msa" ,file[:file.index(msa_end)]+".a3m")) for file in pdb_files_model]
    save_msa_ss_dirs["model%d"%model] = save_msa_ss_dirs_model
    pdb_files["model%d"%model] = pdb_files_model
    ### extract the cluster centers of the pdbs by contact map #####
    contacts = [get_contacts(pdb_file) for pdb_file in pdb_files_model]
    mdl = PCA(n_components=0.9, random_state=42)
    embedding = mdl.fit_transform(np.array(contacts))
    n_coreset = 5
    _,msa_dirs_indx = coreset_sampling2(embedding,n_coreset)
    msa_dirs_selected = [save_msa_ss_dirs_model[idx] for idx in msa_dirs_indx]
    MRF_lliks_coevol = []
    MRF_lliks_marginal = []
    for msa_dir in msa_dirs_selected:
        IDs_sub,seqs_sub = load_fasta(msa_dir)
        seqs_sub =[remove_insertions(seq).upper() for seq in seqs_sub]
        msa_sub = np.array([[aa_order[aa] for aa in seq] for seq in seqs_sub])
        msa_df = mk_msa_df(msa_sub)
        tf.reset_default_graph()
        mrf_coevol = GREMLIN(msa_df,opt_iter=100)
        tf.reset_default_graph()
        mrf_marginal = GREMLIN(msa_df,opt_iter=100,lamb_w = 0)
        MRF_lliks_coevol.append(GREMLIN_llik(msa, mrf_coevol))
        MRF_lliks_marginal.append(GREMLIN_llik(msa, mrf_marginal))
    MRF_lliks_coevol = np.array(MRF_lliks_coevol)
    MRF_lliks_marginal = np.array(MRF_lliks_marginal)
    ### with coevol
    if run_coevol:
        for i in range(n_coreset):
            for j in range(n_coreset):
                if i!=j:
                    ranked_seq = np.argsort(MRF_lliks_coevol[i,:]-MRF_lliks_coevol[j,:])
                    for s in samp_sizes:
                        msa_ss_seqs = [seqs[0]]
                        msa_ss_seqs.extend([seqs[idx] for idx in ranked_seq[0:s]])
                        IDs_ss = [IDs[0]]
                        IDs_ss.extend([IDs[idx] for idx in ranked_seq[0:s]])
                        write_fasta(IDs_ss, msa_ss_seqs, outfile=save_msa_ss_dir+'/ss_MRF_%d_2_MRF_%d_size_%03d'%(i,j,s) +'.a3m') 
    ### without coevol
    if run_marginal:
        print("saving MSA(marginal)")
        for i in range(n_coreset):
            for j in range(n_coreset):
                if i!=j:
                    ranked_seq = np.argsort(MRF_lliks_marginal[i,:]-MRF_lliks_marginal[j,:])
                    for s in samp_sizes:
                        msa_ss_seqs = [seqs[0]]
                        msa_ss_seqs.extend([seqs[idx] for idx in ranked_seq[0:s]])
                        IDs_ss = [IDs[0]]
                        IDs_ss.extend([IDs[idx] for idx in ranked_seq[0:s]])
                        write_fasta(IDs_ss, msa_ss_seqs, outfile=save_msa_ss_dir+'/ss_MRF_%d_2_MRF_%d_size_%03d_marginal'%(i,j,s) +'.a3m') 
