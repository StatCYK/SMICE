import os, time, gc
import re, tempfile
from IPython.display import HTML
import numpy as np
from colabdesign import mk_af_model, clear_mem
from colabdesign.af.contrib import predict
from colabdesign.shared.protein import _np_rmsd
import random
from api import run_mmseqs2
import matplotlib.pyplot as plt
import string
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import sys

from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import silhouette_score
from multiprocessing import Pool, cpu_count
import functools

# Define the jobnames list
jobnames = ['1ceeB', '1g2cF', '1h38D', '1k0nA', '1miqB', '1mnmC', '1nqdA', '1qomB', '1repC', '1rkpA', '1uxmK', '1wyyB', '1x0gA', '1xjtA', '1xntA', '2a73B', '2axzA', '2c1uC', '2ce7C', '2gedB', '2hdmA', '2k0qA', '2kb8A', '2kxoA', '2lqwA', '2lv1A', '2n0aD', '2naoF', '2nntA', '2nxqB', '2ougC', '2p3vA', '3ejhA', '3ewsB', '3gmhL', '3hdeA', '3ifaA', '3j7wB', '3j97M', '3j9cA', '3jv6A', '3kuyA', '3m1bF', '3njqA', '3o44A', '3qy2A', '3t1pA', '3tp2A', '3uyiA', '3zwgN', '4a5wB', '4aanA', '4ae0A', '4b3oB', '4dxtA', '4fu4C', '4gqcC', '4hddA', '4j3oF', '4jphB', '4nc9C', '4o0pA', '4phqA', '4pyiA', '4qdsA', '4qhfA', '4rmbA', '4rr2D', '4rwnA', '4twaA', '4uv2D', '4wsgC', '4y0mJ', '4zrbC', '4zt0C', '5aoeB', '5b3zA', '5c1vA', '5ec5P', '5ejbC', '5f3kA', '5fhcJ', '5fluE', '5hmgA', '5i2mA', '5ineA', '5jytA', '5jzhA', '5keqF', '5l35D', '6z4uA', '7ahlE']

metadata_92 = pd.read_csv("~/Yongkai/Alphafold2/SS_AF2/metadata/92PDBs.csv")

def find_displacements_by_indice(pdb_file1, pdb_file2, indices):
    # Load the PDB files as trajectories
    traj1 = (md.load(pdb_file1)).atom_slice(indices)
    traj2 = md.load(pdb_file2)
    # Find indices of matching atoms
    indices1, indices2 = find_matching_atoms(traj1, traj2)
    # Superpose traj2 onto traj1 using the matching atoms
    traj2.superpose(traj1, atom_indices=indices2, ref_atom_indices=indices1)
    # Calculate distances for the matching atoms in the first frame
    distances = 10 * np.linalg.norm(traj1.xyz[0, indices1] - traj2.xyz[0, indices2], axis=1)
    return np.mean(distances)


def download_pdb_chain(pdb_id, chain_id, file_path):
    """
    Download a specific chain from a PDB file.

    Args:
    pdb_id (str): The ID of the PDB entry.
    chain_id (str): The specific chain ID to download.
    file_path (str): Path where the PDB file should be saved.

    Returns:
    None
    """
    url = f"http://files.rcsb.org/view/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.split('\n')
        with open(file_path, 'w') as file:
            for line in lines:
                # Check if the line corresponds to the desired chain
                if line.startswith(('ATOM', 'HETATM')) and line[21] == chain_id:
                    file.write(line + '\n')
        #print(f"Success to download PDB file for {pdb_id}")
    else:
        print(f"Failed to download PDB file for {pdb_id}. HTTP Status: {response.status_code}")

def fill_gaps(msa):
    '''fill gap with random imputation'''
    tmp = (msa == 21).astype("int")
    seq_prob_all = seq_prob(msa)
    seq_prob_nongap = [seq_prob_all[i,0:22-1]/sum(seq_prob_all[i,0:22-1]) for i in range(seq_prob_all.shape[0])]
    random_impute = [np.random.choice(np.arange(22-1), size=msa.shape[0], p=seq_prob_nongap_i) for seq_prob_nongap_i in seq_prob_nongap]
    random_impute = (np.array(random_impute).T)
    msa_fill = msa*(1-tmp)+random_impute*tmp
    return msa_fill

def get_pdb_withpath(pdb_code, path):
  os.system(f"wget -qnc https://files.rcsb.org/download/{pdb_code}.pdb -P {path}")
  return f"{path}/{pdb_code}.pdb"

def get_pdb_and_chain_withpath(pdb_code="",chain="A", path = "./"):
  get_pdb_withpath(pdb_code, path)
  from colabdesign.shared.protein import pdb_to_string, renum_pdb_str
  string = pdb_to_string('{0}/{1}.pdb'.format(path,pdb_code),chains=chain,models=1)
  with open('{0}/{1}_{2}.pdb'.format(path,pdb_code,chain),'w') as f:
    f.write(string)


def get_sequence_by_pdb_id(file_path, pdb_id):
    """
    Reads a FASTA-like file and retrieves the sequence for a given PDB ID.
    :param file_path: Path to the input text file.
    :param pdb_id: The PDB ID to search for.
    :return: The sequence as a string or None if not found.
    """
    sequence = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.startswith(">") and pdb_id in line:
            # The sequence is on the next line
            sequence = lines[i + 1].strip()
            break
    return sequence

def run_hhalign(query_sequence, target_sequence, query_a3m=None, target_a3m=None):
  with tempfile.NamedTemporaryFile() as tmp_query, \
  tempfile.NamedTemporaryFile() as tmp_target, \
  tempfile.NamedTemporaryFile() as tmp_alignment:
    if query_a3m is None:
      tmp_query.write(f">Q\n{query_sequence}\n".encode())
      tmp_query.flush()
      query_a3m = tmp_query.name
    if target_a3m is None:
      tmp_target.write(f">T\n{target_sequence}\n".encode())
      tmp_target.flush()
      target_a3m = tmp_target.name
    os.system(f"hhalign -hide_cons -i {query_a3m} -t {target_a3m} -o {tmp_alignment.name}")
    X, start_indices = predict.parse_hhalign_output(tmp_alignment.name)
  return X, start_indices

def run_hhfilter(input, output, id=90, qid=10):
  os.system(f"hhfilter -id {id} -qid {qid} -i {input} -o {output}")


def process_jobname(jobname, cov=75):
    try:
        meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
        sequence = meta_info['sequences']
        ID1 = meta_info['Fold1']
        ID2 = meta_info['Fold2']
        Seq1 = meta_info['Seq1']
        Seq2 = meta_info['Seq2']
        if Seq1 !=Seq2:
            if jobname == ID1:
                IDs = [ID2]
            else:
                IDs = [ID1]
        for jobname in IDs:
            copies = 1
            pdb_seq_file = "/n/kou_lab/yongkai/pdb_annotations.txt"
            sequence = get_sequence_by_pdb_id(pdb_seq_file,jobname[0:4]+"_"+jobname[4])
            # MSA options
            msa_method = "mmseqs2"
            pair_mode = "unpaired_paired"
            id = 90
            qid = 0
            do_not_filter = False
            # Templates options
            template_mode = "none"
            use_templates = template_mode in ["mmseqs2","custom"]
            pdb = ""
            chain = ""
            flexible = False
            propagate_to_copies = True
            rm_interchain = False
            rm_sidechain = rm_sequence = flexible
            # filter options
            sequence = str(sequence).replace("/",":")
            sequence = re.sub("[^A-Z:/]", "", sequence.upper())
            sequence = re.sub(":+",":",sequence)
            sequence = re.sub("^[:/]+","",sequence)
            sequence = re.sub("[:/]+$","",sequence)
            # process sequence
            sequences = sequence.split(":")
            u_sequences = predict.get_unique_sequences(sequences)
            if len(sequences) > len(u_sequences):
                print("WARNING: use copies to define homooligomers")
            u_lengths = [len(s) for s in u_sequences]
            sub_seq = "".join(u_sequences)
            seq = sub_seq * copies
            input_opts = {"sequence":u_sequences,
                          "copies":copies,
                          "msa_method":msa_method,
                          "pair_mode":pair_mode,
                          "do_not_filter":do_not_filter,
                          "cov":cov,
                          "id":id,
                          "template_mode":template_mode,
                          "propagate_to_copies":propagate_to_copies}
            save_dir = f"/n/kou_lab/yongkai/SS_AF2/MSA_cov{cov}_all/{jobname}"
            #if not os.path.exists(save_dir+"/msa/msa.npy"):
            msa, deletion_matrix = predict.get_msa(u_sequences, save_dir,
                mode=pair_mode,
                cov=cov, id=id, qid=qid, max_msa=4096,
                do_not_filter=do_not_filter,
                mmseqs2_fn=run_mmseqs2,
                hhfilter_fn=run_hhfilter)
            os.makedirs(save_dir+"/msa/", exist_ok=True)
            np.save(save_dir+"/msa/msa.npy",msa)
            np.save(save_dir+"/msa/del_mat.npy",deletion_matrix)
            if len(msa)>20:
              return jobname  
    except Exception as e:
        print(f"Error processing {jobname}: {str(e)}")

validate_jobs = []
for jobname in jobnames:
    validate_jobs.append(process_jobname(jobname, cov=75))
    
filtered_list = validate_jobs[:]
while None in filtered_list:
    filtered_list.remove(None)