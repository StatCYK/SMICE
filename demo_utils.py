import os, time, gc
import re, tempfile
from IPython.display import HTML
import random
import sys

import matplotlib.pyplot as plt
import string
import numpy as np
import pandas as pd
import pickle
from colabdesign.af.contrib import predict
from multiprocessing import Pool, cpu_count
import functools
import json
np.random.seed(123)
random.seed(123)
from get_MSA import process_jobname as run_get_MSA
from colabdesign import mk_af_model, clear_mem
from colabdesign.shared.protein import _np_rmsd
import shutil  # Added for file operations
import zipfile

from io import BytesIO
from Bio import PDB
from Bio.PDB import PDBIO

def pool_BSS_preds(jobname,base_output_dir,lambs = [0,1,2,3], model_list=[1,2,3,4,5]):
    try:
        outputs=[]
        lamb_list = []
        for lamb in lambs:
            for n_neighbors in [10, 30]:
                pdb_path = base_output_dir+jobname+"/bss_res/pdb_ss_bayes_colab_lamb%d_neighbors%d/"%(lamb,n_neighbors)
                msa_path = base_output_dir+jobname+"/bss_res/msa_ss_bayes_lamb%d_neighbors%d/"%(lamb,n_neighbors)
                msa_file_pattern = f"ss*.a3m"
                msa_files = glob.glob(os.path.join(msa_path, msa_file_pattern))
                num_msa = len(msa_files)
                for model in model_list:
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
        
def pool_enhanced_preds(jobname,base_output_dir,n_iters,model_list,n_coreset = 5):
    try:
        outputs=[]
        for iter in np.arange(1,1+n_iters):
            save_dir = base_output_dir+f"/{jobname}/enhanced_iter{iter}_res"
            ### with coevol two way
            for model in model_list:
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

def fsr_identify(jobname,base_output_dir):
    try:
        outputs_SMICE = pd.read_json(f"{base_output_dir}{jobname}/outputs_SMICE.json.zip")
        filtered_data = outputs_SMICE[outputs_SMICE['avg_plddt']>0.5]
        contacts = np.array([get_contacts(pdb_file) for pdb_file in filtered_data['pdb_path']])
        np.save(f"{base_output_dir}{jobname}/contacts.npy",contacts)
        contacts = np.load(f"{base_output_dir}{jobname}/contacts.npy")
        contacts_variance = np.var(np.array(contacts), axis=0)
        seq_len = int(np.sqrt(len(contacts_variance)))
        contacts_variance = contacts_variance.reshape((seq_len,seq_len))
        np.save(f"{base_output_dir}{jobname}/contacts_bss_variance.npy",contacts_variance)
        contacts_variance = np.load(f"{base_output_dir}{jobname}/contacts_bss_variance.npy")
        seq_len = contacts_variance.shape[0]
        if np.sum(contacts_variance)==0:
            return None
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
            'fsr_pred_resi':[not_conserved_start, not_conserved_end]
        }
        
    except Exception as e:
        print(f"Error processing {jobname}: {str(e)}")
        return None
    
    
def extract_rep_strucs(jobname,filtered_data,outputs_full,base_output_dir,start_res, end_res,PCA_visualization=True,TMscore_visualize=True):

    filtered_files = list(filtered_data['pdb_path'])
    filtered_plddt = list(filtered_data['avg_plddt'])
    TMscore_threshold = 0.85
    cluster_size_threshold = 3
    cluster_res = []
    clusters_files = []
    clusters_files.append(outputs_full.nlargest(1, 'avg_plddt')['pdb_path'].iloc[0])
    select_idx = 0
    ## create tmp folder to store files_to_cluster 
    files_to_cluster_Dir = f"{base_output_dir}{jobname}/all_preds/"
    ## delete the whole dir if it exists
    if os.path.exists(files_to_cluster_Dir):
        shutil.rmtree(files_to_cluster_Dir)
    os.makedirs(files_to_cluster_Dir, exist_ok=True)
    files_to_cluster = [f"{i}.pdb" for i in range(len(filtered_files))]
    for source, target in zip(filtered_files, files_to_cluster):
        extract_substructure_biopython(source, os.path.join(files_to_cluster_Dir, target), start_res, end_res)
    num_clusters = 0
    while len(files_to_cluster) > cluster_size_threshold:
        cluster_file = clusters_files[select_idx]
        ## create tmp folder to store cluster file 
        cluster_Dir = f"{base_output_dir}{jobname}/cluster/"
        ## delete the cluster dir if it exists
        if os.path.exists(cluster_Dir):
            shutil.rmtree(cluster_Dir)
        os.makedirs(cluster_Dir, exist_ok=True)
        cluster_copied_file = cluster_Dir + cluster_file.replace("//", "/").replace("/", "_")
        extract_substructure_biopython(cluster_file, cluster_copied_file,start_res,  end_res)
        cluster_res_tmp = []
        ## cluster predictions close to current cluster
        Res_Dir = f"{base_output_dir}{jobname}/clustering_TMscoreCompute_tmp/"
        if os.path.exists(f"{Res_Dir}"):
            shutil.rmtree(f"{Res_Dir}")
        os.makedirs(Res_Dir, exist_ok=True)
        os.system(f'./bash/benchmark_exp/foldseek_computeTM.sh {cluster_Dir} {files_to_cluster_Dir} {Res_Dir}')
        Res = pd.read_csv(f"{Res_Dir}res.csv", sep='\t',header = None).sort_values(by=0)
        TM_scores_compare = np.array(Res[2])
        next_cluster_file = filtered_files[int(list(Res[0])[np.argmin(TM_scores_compare)] )]
        clustered_files_idx = np.where(TM_scores_compare > TMscore_threshold)[0] # 
        if len(clustered_files_idx)>0:
            if len(clustered_files_idx)>cluster_size_threshold:
                num_clusters+=1
            clustered_files = [files_to_cluster[idx] for idx in clustered_files_idx]
            clustered_files_idx_orig = np.array([ int(list(Res[0])[idx] ) for idx in clustered_files_idx ])
            ### replace the cluster file with the file with highest plddt
            clusters_files[select_idx] = filtered_files[clustered_files_idx_orig[np.argmax(np.array(filtered_plddt)[clustered_files_idx_orig])]]
            ### delete all clustered files from the files_to_cluster_Dir
            for file,idx_orig in zip(clustered_files,clustered_files_idx_orig):
                os.remove(os.path.join(files_to_cluster_Dir, file))
                o = {"file": filtered_files[idx_orig], "cluster_file": clusters_files[select_idx]}
                files_to_cluster.remove(file)
                cluster_res_tmp.append(o)
            cluster_res.extend(cluster_res_tmp)
        if len(files_to_cluster)>cluster_size_threshold:
            select_idx +=1
            clusters_files.append(next_cluster_file)

    cluster_res_df = pd.DataFrame(cluster_res)
    value_counts = cluster_res_df['cluster_file'].value_counts()
    low_freq_elements = value_counts[value_counts < cluster_size_threshold].index.tolist()
    cluster_res_filtered = cluster_res_df[~cluster_res_df['cluster_file'].isin(low_freq_elements)]
    cluster_files_filtered = list(set(cluster_res_filtered['cluster_file']))
    cluster_dir = f"{base_output_dir}{jobname}/Clustering_Res"
    os.makedirs(cluster_dir, exist_ok=True)
    cluster_res_filtered.to_csv(f"{cluster_dir}/res_cluster.tsv", sep='\t')
    cluster_indices = np.array([filtered_files.index(file) for file in cluster_files_filtered ])
    print(f"finish cluster selection for {jobname}")
    cluster_rows = filtered_data.iloc[cluster_indices]
    cluster_info = {
        'jobname': jobname,
        'pdb_path': cluster_rows['pdb_path'].tolist(),
        'avg_plddt': cluster_rows['avg_plddt'].tolist(),
        'score_path':cluster_rows['score_path'].tolist(),
        'n_cluster':len(cluster_files_filtered)
    }
    cluster_file = f"{cluster_dir}/cluster.json.zip"
    with open(cluster_file, 'w') as f:
        json.dump(cluster_info, f)
    # Save PDB files for this cluster
    member_counts = cluster_res_filtered['cluster_file'].value_counts()
    cluster_sizes = [member_counts[cluster_files_filtered[i]] for i in range(len(cluster_files_filtered))]
    cluster_files_sorted = [cluster_files_filtered[i] for i in np.argsort(cluster_sizes)[::-1]]
    cluster_sizes_sorted = np.sort(cluster_sizes)[::-1]
    save_cluster_centers_pdbs(jobname, cluster_files_sorted, cluster_sizes_sorted)

    ### visualization of selected cluster
    if PCA_visualization:
        # Visualization 
        mdl = PCA(n_components=2, random_state=42)
        contacts_SMICE_filtered = np.array([get_contacts(pdb_file) for pdb_file in filtered_data["pdb_path"]])
        contacts_SMICE_cluster = np.array([get_contacts(pdb_file) for pdb_file in cluster_rows["pdb_path"]])
        embedding = mdl.fit_transform(contacts_SMICE_filtered )
        plt.figure(figsize=(7, 6))

        sc = plt.scatter(embedding[:, 0], embedding[:, 1], 
                        c="blue", 
                        alpha=0.6)
        plt.scatter(embedding[cluster_indices, 0], embedding[cluster_indices, 1], 
                   marker='*', s=200, c='black', 
                   edgecolors='white', linewidths=0.5,
                   label='Rep. Structures')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend()
        plot_dir = f"{cluster_dir}/plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = f"{plot_dir}/pca_cluster.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        

def extract_substructure_biopython(pdb_file, output_file, start_res, end_res):
    """
    Extract residues from start_res to end_res (inclusive) from a PDB file
    Args:
        pdb_file: Input PDB file path
        output_file: Output PDB file path
        start_res: Starting residue number
        end_res: Ending residue number
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    # Create a new structure for the substructure
    io = PDBIO()
    # Filter residues in the specified range
    class ResidueSelect(PDB.Select):
        def accept_residue(self, residue):
            res_id = residue.get_id()
            # Check if residue is in the specified chain and range
            if (res_id[0] == ' ' and  # Skip hetero/water residues
                start_res <= res_id[1] <= end_res):
                return True
            return False
    # Save the filtered structure
    io.set_structure(structure)
    io.save(output_file, ResidueSelect())

def save_cluster_centers_pdbs(jobname, cluster_files, cluster_sizes):
    """
    Save PDB files of cluster centers directly to a ZIP file.
    Args:
        jobname (str): Name of the job
        cluster_indices (list): Indices of cluster centers
        filtered_data (DataFrame): DataFrame containing all structures
    Returns:
        str: Path to the created ZIP file
    """
    # Determine ZIP file name
    zip_filename = f"{base_output_dir}{jobname}/RepStructure.zip"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(zip_filename), exist_ok=True)
    # Create ZIP file and add PDB files
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i in range(len(cluster_files)):
                src_path = cluster_files[i]
                cluster_size = cluster_sizes[i]
                internal_filename = f"cluster_center_{i+1}_size_{cluster_size}.pdb"
                # Add file to ZIP
                zipf.write(src_path, arcname=internal_filename)
        print(f"Successfully created ZIP file: {zip_filename}")
        return zip_filename  
    except Exception as e:
        print(f"Failed to create ZIP file {zip_filename}. Reason: {e}")
        return None
