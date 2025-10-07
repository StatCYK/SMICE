import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn.decomposition import PCA
from util_SMICE import *
import sys
import matplotlib.pyplot as plt
import pickle
import re
import os
import glob
import subprocess
import random
from sklearn.metrics.pairwise import euclidean_distances
import json
from fsr_identify import *
import shutil  # Added for file operations
import zipfile
from io import BytesIO
from Bio import PDB
from Bio.PDB import PDBIO

np.random.seed(123)
random.seed(123)

jobname = sys.argv[1] 
with open('../config/config_SMICE_benchmark.json', 'r') as f:
    config = json.load(f)


base_output_dir = config["base_output_dir"]
base_result_dir = config["base_result_dir"]
MSA_saved_basedir = config["MSA_saved_basedir"]
cov = 75
MSA_saved_dir = f"{MSA_saved_basedir}{jobname}"
PCA_visualization = True


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
    zip_filename = f"{base_output_dir}{jobname}/cluster_centers_greedy.zip"
    
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


MSA_saved_dir = f"{MSA_saved_basedir}{jobname}/"

IDs,seqs = load_fasta(os.path.join(MSA_saved_dir, "msa.a3m"))
outputs_SMICE = pd.read_json(f"{base_output_dir}{jobname}/outputs_SMICE.json.zip")
#
filtered_data = outputs_SMICE[outputs_SMICE['avg_plddt']>0.5]
### identify the variable region
fsr_identify_res = fsr_identify(jobname)
fsr_len = fsr_identify_res['fsr_resi'][1] - fsr_identify_res['fsr_resi'][0]
start_res,end_res = extend_interval_symmetric(fsr_identify_res['fsr_resi'][0], fsr_identify_res['fsr_resi'][1], 1, len(seqs[0]))
filtered_files = list(filtered_data['pdb_path'])
filtered_plddt = list(filtered_data['avg_plddt'])
### greedy algorithm
## obtain the first cluster as the pred by full MSA
outputs_full = []
for model in range(1,6):
    o = {}
    pattern = f"*_relaxed*model_{model:01d}*.pdb"
    pdb_files = glob.glob(os.path.join(f"{MSA_saved_dir}pdb", pattern))
    score_file_pattern = f"*_model_{model:01d}*.json"
    score_file = glob.glob(os.path.join(f"{MSA_saved_dir}pdb", score_file_pattern))[0]
    with open(score_file,"r") as f:
        plddt_scores = pd.read_json(f)
    avg_plddt = np.mean(plddt_scores["plddt"])/100
    o.update({'msa_path': f"{MSA_saved_dir}msa.a3m"})
    o.update({'pdb_path': pdb_files[0]})
    o.update({'score_path': score_file})
    o.update({'model': model})
    o.update({'avg_plddt': avg_plddt})
    o.update({'avg_pae': np.mean(np.mean(np.mean(np.array(plddt_scores["pae"]))))})
    o.update({'max_pae': plddt_scores["max_pae"].iloc[0]})
    o.update({'ptm': plddt_scores["ptm"].iloc[0]})
    o.update
    outputs_full.append(o)
outputs_full = pd.DataFrame.from_records(outputs_full)

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
    os.system(f'../bash/benchmark_exp/foldseek_computeTM.sh {cluster_Dir} {files_to_cluster_Dir} {Res_Dir}')
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
cluster_dir = f"{base_output_dir}{jobname}/cluster_greedy_fast"
os.makedirs(cluster_dir, exist_ok=True)
cluster_res_filtered.to_csv(f"{cluster_dir}/res_cluster.tsv", sep='\t')
cluster_indices = np.array([filtered_files.index(file) for file in cluster_files_filtered ])
print(f"finish cluster selection for {jobname}")
cluster_rows = filtered_data.iloc[cluster_indices]
output_file = f"{cluster_dir}/cluster.json.zip"
cluster_rows.to_json(output_file, index=False)
cluster_info = {
    'jobname': jobname,
    'pdb_path': cluster_rows['pdb_path'].tolist(),
    'avg_plddt': cluster_rows['avg_plddt'].tolist(),
    'score_path':cluster_rows['score_path'].tolist(),
    'n_cluster':len(cluster_files_filtered)
}
cluster_file = f"{cluster_dir}/cluster.json"
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
    mdl = PCA(random_state=42)
    ## create tmp folder for foldseek
    shutil.rmtree(files_to_cluster_Dir)
    os.makedirs(files_to_cluster_Dir, exist_ok=True)
    all_files_copy = [f"{i}.pdb" for i in range(len(filtered_files))]
    for source, target in zip(filtered_files, all_files_copy):
        extract_substructure_biopython(source, os.path.join(files_to_cluster_Dir, target),start_res,  end_res)
    if os.path.exists(cluster_Dir):
            shutil.rmtree(cluster_Dir)
    os.makedirs(cluster_Dir, exist_ok=True)
    cluster_files_copy = [f"{i}.pdb" for i in range(len(cluster_indices))]
    for source, target in zip(list(cluster_info['pdb_path']), cluster_files_copy):
        extract_substructure_biopython(source, os.path.join(cluster_Dir, target),start_res,  end_res)
    if os.path.exists(f"{Res_Dir}"):
        shutil.rmtree(f"{Res_Dir}")
    os.makedirs(Res_Dir, exist_ok=True)
    os.system(f'../bash/benchmark_exp/foldseek_computeTM.sh {cluster_Dir} {files_to_cluster_Dir} {Res_Dir}')
    TMscores_cluster2all_res = pd.read_csv(f"{Res_Dir}res.csv", sep='\t', header=None,
        names=['all_files','cluster_files','TMscore'])
    TMscores2cluster_df = TMscores_cluster2all_res.pivot(index='all_files', columns='cluster_files', values='TMscore').sort_index()
    TMscores2cluster = np.array(TMscores2cluster_df)
    TMscores2cluster = np.where(np.isnan(TMscores2cluster), np.nanmean(TMscores2cluster, axis=1, keepdims=True), TMscores2cluster)
    embedding = mdl.fit_transform(TMscores2cluster)
    # Visualization 
    outputs_SMICE['max_TMscore'] = outputs_SMICE.apply(lambda x: max(x['TMscore1'], x['TMscore2']), axis=1)
    mdl = PCA(n_components=2, random_state=42)
    contacts_SMICE_filtered = np.array([get_contacts(pdb_file) for pdb_file in filtered_data["pdb_path"]])
    contacts_SMICE_cluster = np.array([get_contacts(pdb_file) for pdb_file in cluster_rows["pdb_path"]])
    embedding = mdl.transform(contacts_SMICE_filtered )
    plt.figure(figsize=(7, 6))
    TM_score_diff = np.sign(filtered_data['TMscore1']-filtered_data['TMscore2'])*filtered_data['max_TMscore']
    v_abs_max = np.max(np.max(TM_score_diff))
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], 
                    c=TM_score_diff, 
                    cmap='RdYlBu',
                    vmin=-v_abs_max, vmax=v_abs_max,
                    alpha=0.6)
    plt.scatter(embedding[cluster_indices, 0], embedding[cluster_indices, 1], 
               marker='*', s=200, c='black', 
               edgecolors='white', linewidths=0.5,
               label='Rep. Structures')
    cbar = plt.colorbar(sc)
    cbar.set_label('signed Max-TMscore')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plot_dir = f"{cluster_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_file = f"{plot_dir}/pca_cluster.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
