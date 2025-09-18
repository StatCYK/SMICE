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
os.chdir("../../src/")
sys.path.insert(0, os.getcwd())  
from util_SMICE import *
import sys
import pickle
import traceback
import plotly.graph_objects as go
import zipfile
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Set larger font sizes for all plots
plt.rcParams.update({
    'font.size': 24,              # Base font size
    'axes.titlesize': 24,        # Axis titles
    'axes.labelsize': 24,        # Axis labels
    'xtick.labelsize': 24,       # X-axis tick labels
    'ytick.labelsize': 24,       # Y-axis tick labels
    'legend.fontsize': 24,       # Legend
    'figure.titlesize': 20       # Figure title
})

with open('../config/config_SMICE_benchmark.json', 'r') as f:
    config = json.load(f)

ALPHA_VALUES = np.arange(0,2.1,0.1)
alpha_choice = 10  # "adaptive"

metadata_92 = pd.read_csv(config["meta_path"])
base_output_dir = config["base_output_dir"]
base_result_dir = config["base_result_dir"]
jobnames = config["jobnames"]
true_pdb_path = config["true_pdb_path"]

#base_output_dir = "/n/kou_lab/yongkai/SS_AF2/results_cov75_ss_bayes_updated/"

def process_jobname_and_plot(jobname):
    # Load SMICE data
    outputs_bss = pd.read_json(f"{base_output_dir}{jobname}/bss_res/outputs_bss_alpha_{alpha_choice}.json.zip")
    outputs_enhanced = pd.read_json(f"{base_output_dir}{jobname}/outputs_enhanced_alpha_{alpha_choice}.json.zip")
    outputs_SMICE = pd.concat([outputs_bss, outputs_enhanced], ignore_index=True) 
    #contacts_SMICE = np.array([get_contacts(pdb_file) for pdb_file in outputs_SMICE["pdb_path"]])
    #np.save(f"{base_output_dir}{jobname}/contacts.npy",contacts_SMICE)
    contacts_SMICE = np.load(f"{base_output_dir}{jobname}/contacts.npy")
    outputs_SMICE['max_TMscore'] = outputs_SMICE.apply(lambda x: max(x['TMscore1'], x['TMscore2']), axis=1)
    TMscore_diff_SMICE = np.sign(outputs_SMICE['TMscore1'] - outputs_SMICE['TMscore2'])*outputs_SMICE['max_TMscore']
    meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
    sequence = meta_info['sequences']
    ID1 = meta_info['Fold1']
    ID2 = meta_info['Fold2']
    ID1_dir = "/n/kou_lab/yongkai/SS_AF2/pdbs_92/"+ID1[0:4]+"_"+ID1[4]+".pdb"
    ID2_dir = "/n/kou_lab/yongkai/SS_AF2/pdbs_92/"+ID2[0:4]+"_"+ID2[4]+".pdb"
    Seq1 = meta_info['Seq1']
    Seq2 = meta_info['Seq2']
    Seq1 = ''.join([three_to_one.get(name, 'X') for name in [res.name for res in md.load_pdb(ID1_dir).topology.residues]])
    Seq2 = ''.join([three_to_one.get(name, 'X') for name in [res.name for res in md.load_pdb(ID2_dir).topology.residues]])
    seq_len1 = int(np.sqrt(len(get_contacts(ID1_dir))))
    seq_len2 = int(np.sqrt(len(get_contacts(ID2_dir))))
    Seq1 = Seq1[0:seq_len1]
    Seq2 = Seq2[0:seq_len2]
    contact1 = get_contacts(ID1_dir).reshape((len(Seq1),len(Seq1)))
    contact2 = get_contacts(ID2_dir).reshape((len(Seq2),len(Seq2)))
    TMscores_fsr = np.load(f"/n/home13/yongkai/Yongkai/Alphafold2/SS_AF2/metadata/{jobname}/TMscores_fsr.npy")# the tmscores(fsr) between two folds
    if alpha_choice == "adaptive":
        alpha_indice = np.argmin(TMscores_fsr)
    else:
        alpha_indice = alpha_choice

    TMscore12 = TMscores_fsr[alpha_indice]
    # Create aligner and perform alignment
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    L = int(np.sqrt(contacts_SMICE.shape[1]))
    aligned_contact1 = align_contact_maps(aligner.align(sequence, Seq1)[0], contact1, L)
    aligned_contact2 = align_contact_maps(aligner.align(sequence, Seq2)[0], contact2, L)
    aligned_contact1 = aligned_contact1.reshape((-1,))
    aligned_contact2 = aligned_contact2.reshape((-1,))
    contact12 = np.array([aligned_contact1,aligned_contact2])# contact map of two true folds
    # Load comparison data
    contacts_uniform = np.load("/n/kou_lab/yongkai/Alphafold2/AFclust_random_res/"+jobname+f"_contacts_random.npy")
    contacts_clust = np.load("/n/kou_lab/yongkai/Alphafold2/AFclust_random_res/"+jobname+f"_contacts_clust.npy")

    TMscores_uniform = np.load("/n/kou_lab/yongkai/Alphafold2/AFclust_random_res/"+jobname+f"_TMscores_random_complete_alpha_{alpha_choice}.npy")
    TMscores_uniform = TMscores_uniform.reshape((2, len(TMscores_uniform)//2))
    TMscores_clust = np.load("/n/kou_lab/yongkai/Alphafold2/AFclust_random_res/"+jobname+f"_TMscores_clust_alpha_{alpha_choice}.npy")
    TMscores_clust = TMscores_clust.reshape((2, len(TMscores_clust)//2))
    TMscore_diff_uniform = np.sign(TMscores_uniform[0,:] - TMscores_uniform[1,:])*np.max(TMscores_uniform,0)
    TMscore_diff_clust = np.sign(TMscores_clust[0,:] - TMscores_clust[1,:])*np.max(TMscores_clust,0)

    # Combine all contacts for PCA fitting
    all_contacts = np.concatenate([
        contacts_SMICE, 
        contacts_uniform, 
        contacts_clust,
        contact12
    ])
    all_contacts = all_contacts.reshape((all_contacts.shape[0],L,L))
    all_contacts = all_contacts[:,150:300,150:300]
    all_contacts = all_contacts.reshape((all_contacts.shape[0],150**2))
    
    # Split back into the different methods
    n_SMICE = len(contacts_SMICE)
    n_SMICE_SS = len(outputs_bss)
    n_uniform = len(contacts_uniform)
    n_clust = len(contacts_clust)
    # Fit PCA on all contacts
    mdl = PCA(n_components=2, random_state=42)
    mdl.fit(all_contacts[:n_SMICE,:])
    embedding_all = mdl.transform(all_contacts[:n_SMICE,:])
    print([np.max(outputs_SMICE['TMscore1']),np.max(outputs_SMICE['TMscore2'])])
    embedding_SMICE = embedding_all[:n_SMICE]
    embedding_uniform = embedding_all[n_SMICE:n_SMICE+n_uniform]
    embedding_clust = embedding_all[n_SMICE+n_uniform:n_SMICE+n_uniform+n_clust]
    embedding_Fold12 = embedding_all[n_SMICE+n_uniform+n_clust:]
    # Determine axis limits that will cover all data points
    x_min = embedding_all[:,0].min()
    x_max = embedding_all[:,0].max()
    y_min = embedding_all[:,1].min()
    y_max = embedding_all[:,1].max()
    # Add some padding to the limits
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_lim = (x_min - x_padding, x_max + x_padding)
    y_lim = (y_min - y_padding, y_max + y_padding)
    # Determine symmetric vmin/vmax for consistent coloring
    max_diff = max(
        np.abs(TMscore_diff_SMICE).max(),
        np.abs(TMscore_diff_uniform).max(),
        np.abs(TMscore_diff_clust).max()
    )
    vmin, vmax = -max_diff, max_diff

    # Create the figure with larger size
    fig = plt.figure(figsize=(16, 24))  # Increased from (24, 12)
    fig.suptitle(f'PCA of predicted Contact Maps Colored by TMscore Difference', fontsize=20)  # Increased from 16

    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    cbar_ax = fig.add_subplot(gs[:, 2])

    # Plot SMICE (not enhanced)
    sc1 = ax1.scatter(
        embedding_SMICE[0:n_SMICE_SS, 0], embedding_SMICE[0:n_SMICE_SS, 1], 
        c=TMscore_diff_SMICE[0:n_SMICE_SS], cmap='RdYlBu', s=300, vmin=vmin, vmax=vmax
    )
    ax1.set_title('SMICE(not enhanced)', fontsize=16)  # Explicitly set
    ax1.set_xlabel('PC1', fontsize=24)  # Explicitly set
    ax1.set_ylabel('PC2', fontsize=24)  # Explicitly set
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)

    # Plot SMICE
    sc2 = ax2.scatter(
        embedding_SMICE[:, 0], embedding_SMICE[:, 1], 
        c=TMscore_diff_SMICE, cmap='RdYlBu', s=300, vmin=vmin, vmax=vmax
    )
    ax2.set_title('SMICE', fontsize=16)
    ax2.set_xlabel('PC1', fontsize=24)
    ax2.set_ylabel('PC2', fontsize=24)
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)

    # Plot Clust
    sc3 = ax3.scatter(
        embedding_clust[:, 0], embedding_clust[:, 1], 
        c=TMscore_diff_clust, cmap='RdYlBu', s=300, vmin=vmin, vmax=vmax
    )
    ax3.set_title('AF-Cluster', fontsize=16)
    ax3.set_xlabel('PC1', fontsize=24)
    ax3.set_ylabel('PC2', fontsize=24)
    ax3.set_xlim(x_lim)
    ax3.set_ylim(y_lim)

    # Plot Uniform
    sc4 = ax4.scatter(
        embedding_uniform[:, 0], embedding_uniform[:, 1], 
        c=TMscore_diff_uniform, cmap='RdYlBu', s=300, vmin=vmin, vmax=vmax
    )
    ax4.set_title('Random Sampling', fontsize=16)
    ax4.set_xlabel('PC1', fontsize=24)
    ax4.set_ylabel('PC2', fontsize=24)
    ax4.set_xlim(x_lim)
    ax4.set_ylim(y_lim)
    for ax in [ax1,ax2]:
        ax.scatter(
        embedding_Fold12[:, 0], embedding_Fold12[:, 1], 
        c=[1,-1], cmap='RdYlBu', vmin=vmin, vmax=vmax, marker='*', s=1000,edgecolors='black',linewidths=3
        )

    # Add single colorbar for all plots
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='signed Max-TMscore')
    cbar.ax.tick_params(labelsize=24)  # Set colorbar tick label size
    cbar.set_label('TMscore Difference', size=24)  # Set colorbar label size
    plt.tight_layout()
    # Save the figure
    os.makedirs(f"{base_result_dir}comparison/", exist_ok=True)
    output_plot_path = f"{base_result_dir}comparison/pca_plot_{jobname}.png"
    plt.savefig(output_plot_path, dpi=600, bbox_inches='tight')
    plt.close()

def main():
    jobnames = ["2c1uC"]#["3jv6A"]  # ["1xntA","4wsgC"]
    try:
        num_processes = multiprocessing.cpu_count() - 1  # Leave one core free
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(process_jobname_and_plot, jobnames)
            print("finish all")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
    finally:
        # Clean up
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()