import multiprocessing
from functools import partial
import numpy as np
import matplotlib.pylab as plt
from scipy import stats
from scipy.spatial.distance import pdist,squareform
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from util_SMICE import *
import sys
import matplotlib.pyplot as plt
import pickle
import re
import os
import kmedoids
import glob
import kmedoids
from sklearn.metrics import silhouette_score
from umap import UMAP
import subprocess
import random
from scipy.spatial import ConvexHull
from sklearn.metrics.pairwise import euclidean_distances
import json
np.random.seed(123)
random.seed(123)
import sys
module_dir = "./"
sys.path.append(module_dir)
with open('../config/config_SMICE_benchmark.json', 'r') as f:
    config = json.load(f)

metadata_92 = pd.read_csv(config["meta_path"])
base_output_dir = config["base_output_dir"]
base_result_dir = config["base_result_dir"]
true_pdb_path = config["true_pdb_path"]
MSA_saved_basedir = config["MSA_saved_basedir"]
jobnames = config["jobnames"]
ALPHA_VALUES = np.arange(0,2.1,0.1)
alpha_choice = 10

def no_diag_sum(arr):
    mask = ~np.eye(arr.shape[0], dtype=bool)
    # Extract non-diagonal elements
    non_diag_elements = arr[mask]
    return(np.sum(non_diag_elements))

def no_diag_ratio(arr,fsr_resi):
    seq_len = arr.shape[0]
    fsr_len = fsr_resi[1]-fsr_resi[0]
    mask = np.ones(seq_len, dtype=bool)
    mask[fsr_resi[0]:fsr_resi[1]] = False
    arr_no_fsr_sum = no_diag_sum(arr[mask, :][:, mask])
    arr_whole_sum = no_diag_sum(arr)
    fsr_var_ratio2whole = (arr_whole_sum-arr_no_fsr_sum)/(2*seq_len*fsr_len-fsr_len**2-fsr_len)/(arr_whole_sum/(seq_len**2-seq_len))
    return fsr_var_ratio2whole

def contact_diff_ratio(contact1,contact2,fsr_resi):
    arr = np.abs(contact1-contact2)
    if np.sum(arr)==0:
        return None
    seq_len = arr.shape[0]
    fsr_len = fsr_resi[1]-fsr_resi[0]
    mask = np.ones(seq_len, dtype=bool)
    mask[fsr_resi[0]:fsr_resi[1]] = False
    arr_no_fsr_sum = no_diag_sum(arr[mask, :][:, mask])
    arr_whole_sum = no_diag_sum(arr)
    fsr_diff_ratio2whole = (arr_whole_sum-arr_no_fsr_sum)/(2*seq_len*fsr_len-fsr_len**2-fsr_len)/(arr_whole_sum/(seq_len**2-seq_len))
    return fsr_diff_ratio2whole


def fsr_identify(jobname):
    try:
        MSA_saved_dir = f"{MSA_saved_basedir}{jobname}"
        meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
        sequence = meta_info['sequences']
        fsr_seq = meta_info["Sequence of fold-switching region"]#sequence#
        contacts = []
        outputs_SMICE = pd.read_json(f"{base_output_dir}{jobname}/outputs_SMICE.json.zip")
        filtered_data = outputs_SMICE[outputs_SMICE['avg_plddt']>0.5]
        #contacts = [get_contacts(pdb_file) for pdb_file in filtered_data['pdb_path']]
        #np.save(f"{base_output_dir}{jobname}/contacts.npy",contacts)
        contacts = np.load(f"{base_output_dir}{jobname}/contacts.npy")
        contacts_variance = np.var(np.array(contacts), axis=0)
        seq_len = int(np.sqrt(len(contacts_variance)))
        contacts_variance = contacts_variance.reshape((seq_len,seq_len))
        np.save(f"{base_output_dir}{jobname}/contacts_bss_variance.npy",contacts_variance)
        contacts_variance = np.load(f"{base_output_dir}{jobname}/contacts_bss_variance.npy")
        seq_len = contacts_variance.shape[0]
        if np.sum(contacts_variance)==0:
            return None
        
        meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
        sequence = meta_info['sequences']
        ID1 = meta_info['Fold1']
        ID2 = meta_info['Fold2']
        ID1_dir = true_pdb_path+ID1[0:4]+"_"+ID1[4]+".pdb"
        ID2_dir = true_pdb_path+ID2[0:4]+"_"+ID2[4]+".pdb"
        Seq1 = meta_info['Seq1']
        Seq2 = meta_info['Seq2']
        fsr_seq = meta_info["Sequence of fold-switching region"]
        if alpha_choice == "adaptive":
            alpha_indice = np.argmin(TMscores_fsr)
        else:
            alpha_indice = alpha_choice
        fsr_seq_extend = extend_fsr_seq(fsr_seq,sequence,int(len(fsr_seq)*ALPHA_VALUES[alpha_indice]))        
        fsr_resi = align_fsr(sequence, fsr_seq_extend)
        fsr_resi_ori = align_fsr(sequence, fsr_seq)
        seq_len = contacts_variance.shape[0]
        
        # Determine initial block size (max of seq_len-30 and 30)
        initial_block_size = max(seq_len - 40, 40)
        initial_block_size = min(initial_block_size, seq_len)  # Ensure it doesn't exceed sequence length
        # Function to calculate average for a block excluding its own rows/columns
        def get_excluded_avg_std(start, end):
            # Create mask for residues not in the block
            mask = np.ones(seq_len, dtype=bool)
            mask[start:end] = False
            # Get submatrix excluding block's rows and columns
            submatrix = contacts_variance[mask, :][:, mask]
            return np.mean(submatrix),np.std(submatrix.reshape(-1,1))
        
        # Find initial block position that minimizes the average of excluded residues
        min_avg = float('inf')
        initial_start = 0
        
        for i in range(seq_len - initial_block_size + 1):
            current_avg,_ = get_excluded_avg_std(i, i + initial_block_size)
            if current_avg < min_avg:
                min_avg = current_avg
                initial_start = i
        
        initial_end = initial_start + initial_block_size
        initial_avg,initial_std = get_excluded_avg_std(initial_start, initial_end)
        
        # Now iteratively remove residues from either end to find no conserved core
        current_start, current_end = initial_start, initial_end
        while current_end - current_start > 50:  # Need at least 50 residue
            # Calculate averages if we remove left or right residue
            avg_remove_left,_ = get_excluded_avg_std(current_start + 1, current_end)
            avg_remove_right,_ = get_excluded_avg_std(current_start, current_end - 1)
            cur_no_conserved_size = current_end-current_start
            cur_conserved_size = seq_len-cur_no_conserved_size
            # Determine which removal gives better (lower) average
            if avg_remove_left < avg_remove_right:
                # Prefer to remove left if averages are equal
                sum_including = avg_remove_left*(1+cur_conserved_size)**2-current_avg*cur_conserved_size**2
                if sum_including/(2*cur_conserved_size) <= 4*initial_avg:
                    current_start += 1
                    current_avg = avg_remove_left
                else:
                    break
            else:
                sum_including = avg_remove_right*(1+cur_conserved_size)**2-current_avg*cur_conserved_size**2
                if sum_including/(2*cur_conserved_size) <= 4*initial_avg:
                    current_end -= 1
                    current_avg = avg_remove_right
                else:
                    break
        
        not_conserved_start, not_conserved_end = current_start, current_end
        not_conserved_residues = list(range(not_conserved_start + 1, not_conserved_end + 1))  # 1-based indexing
        np.save(f"{base_output_dir}{jobname}/pred_fsr.npy",np.array([not_conserved_start,not_conserved_end]))
        # Visualization
        plt.figure(figsize=(10, 11))
        plt.imshow(contacts_variance, origin="lower", extent=(0, seq_len, 0, seq_len), cmap='RdBu')
        # Highlight the no conserved core (red)
        plt.gca().add_patch(plt.Rectangle((not_conserved_start, not_conserved_start), 
                           not_conserved_end - not_conserved_start, not_conserved_end - not_conserved_start,
                           fill=False, edgecolor='yellow', lw=4))
        plt.xlabel("Residue Index")
        plt.ylabel("Residue Index")
        plt.colorbar(label="contact distance variance")
        plt.savefig(f"{base_output_dir}{jobname}/contacts_variance.png")
        plt.close()
        return {
            'jobname': jobname,
            'not_conserved_residues': not_conserved_residues,
            'fsr_pred_resi':[not_conserved_start, not_conserved_end],
            'fsr_resi':fsr_resi
        }
        
    except Exception as e:
        print(f"Error processing {jobname}: {str(e)}")
        return None

def main():
    with multiprocessing.Pool() as pool:
        results = pool.map(partial(fsr_identify), jobnames)
    
if __name__ == "__main__":
    main()