import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import matplotlib.pylab as plt
from scipy import stats
from scipy.spatial.distance import pdist,squareform
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os, time, gc
import re, tempfile
from pathlib import Path
import glob
from IPython.display import HTML
import pickle
import random
import requests
# from api import run_mmseqs2
import matplotlib.pyplot as plt
import string
import mdtraj as md
import pandas as pd
import copy
from scipy.spatial.distance import squareform, pdist, cdist
from typing import List, Tuple, Optional, Dict, NamedTuple, Union, Callable
import Bio
import Bio.PDB
import Bio.SeqRecord
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio import SeqIO, PDB, AlignIO
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import PDBParser
# import biotite.structure as bs
from biotite.structure.io.pdb import PDBFile
from biotite.database import rcsb
from biotite.structure.io.pdbx import get_structure
from biotite.structure import filter_amino_acids, distance, AtomArray
from biotite.structure.residues import get_residues
from biotite.sequence import ProteinSequence
from sklearn.metrics import pairwise_distances
from tmtools import tm_align
from tmtools.io import get_residue_data  # can't have get_structure here too !!!
from tmtools.io import get_structure as tmtool_get_structure  # can't have get_structure here too !!!
from Bio.PDB.MMCIFParser import MMCIFParser
from umap import UMAP
from kmedoids import KMedoids
import numpy as np
import matplotlib.pyplot as plt
from string import ascii_uppercase, ascii_lowercase
import hashlib
import random
from collections import OrderedDict, Counter
from sklearn.neighbors import NearestNeighbors
import Bio
from Bio import Align
from Bio.Align import substitution_matrices
from sklearn.decomposition import PCA

################
alphabet = "ARNDCQEGHILKMFPSTWYVX"
states = len(alphabet)
a2n = {}
for a,n in zip(alphabet,range(states)):
  a2n[a] = n
################

three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)



def align_sequences(seq1, seq2, matrix_name="BLOSUM62"):
    """Align seq2 to seq1 using pairwise alignment"""
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load(matrix_name)
    aligner.mode = 'global'  # or 'local' depending on your needs
    alignments = aligner.align(seq1, seq2)
    return alignments[0]  # Get the best alignment

def align_contact_maps(alignment, contact_map2, seq1_len):
    """
    Transfer contact map from seq2 to seq1 coordinates
    
    Parameters:
    - alignment: Bio.Align.PairwiseAlignment object
    - contact_map2: 2D numpy array for seq2's contacts (shape [len(seq2), len(seq2)])
    - seq1_len: length of the first sequence
    
    Returns:
    - aligned_contact_map: contact map in seq1's coordinates (shape [seq1_len, seq1_len])
    """
    # Initialize empty contact map
    aligned_contact_map = np.zeros((seq1_len, seq1_len))
    
    # Get the alignment coordinates (now properly handling numpy arrays)
    target_indices, query_indices = alignment.aligned
    
    # Create mapping from seq2 positions to seq1 positions
    seq2_to_seq1 = {}
    
    # Iterate through each aligned block
    for block_idx in range(target_indices.shape[0]):
        # Get start/end positions for this block
        t_start, t_end = target_indices[block_idx, 0], target_indices[block_idx, 1]
        q_start, q_end = query_indices[block_idx, 0], query_indices[block_idx, 1]
        
        # Calculate length of this aligned block
        block_len = t_end - t_start
        
        # Create position mapping for this block
        for offset in range(block_len):
            seq2_pos = q_start + offset
            seq1_pos = t_start + offset
            seq2_to_seq1[seq2_pos] = seq1_pos
    
    # Transfer contacts
    for (i,j), value in np.ndenumerate(contact_map2):
        if value > 0:  # if there's a contact
            if i in seq2_to_seq1 and j in seq2_to_seq1:
                aligned_contact_map[seq2_to_seq1[i], seq2_to_seq1[j]] = value
    
    return aligned_contact_map

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def normalize(x):
  x = stats.boxcox(x - np.amin(x) + 1.0)[0]
  x_mean = np.mean(x)
  x_std = np.std(x)
  return((x-x_mean)/x_std)

def coreset_sampling(points, k):
    coreset_idx = [np.argmax(points[:,0])]
    coreset = [points[coreset_idx[0]]]
    for _ in range(k-1):
        dists = pairwise_distances(points, coreset).min(axis=1)
        idx = np.argmax(dists)
        new_point = points[idx]
        coreset_idx.append(idx)
        coreset.append(new_point)
    return np.array(coreset),np.array(coreset_idx)

def coreset_sampling2(points, k):
    kmedoids = KMedoids(n_clusters=k, metric='euclidean')
    kmedoids.fit(np.array(points))
    coreset_idx = kmedoids.medoid_indices_
    cluster_labels = kmedoids.labels_
    strucs_clustered_indices = [[node for node, cluster in enumerate(cluster_labels) if cluster == c] for c in range(k)]
    coreset = [points[idx] for idx in coreset_idx]
    return strucs_clustered_indices,np.array(coreset_idx)

from sklearn.metrics import silhouette_score
from kneed import KneeLocator

def determine_optimal_k(points, k_range=np.arange(10,51,5)):
    """
    Determine optimal number of clusters using silhouette score and elbow method
    """
    silhouette_scores = []
    distortions = []
    for n_clusters in list(k_range):
        kmedoids = KMedoids(n_clusters=int(n_clusters), metric='euclidean', random_state=42)
        labels = kmedoids.fit_predict(points)
        
        # Silhouette score
        if n_clusters > 1:  # Silhouette requires at least 2 clusters
            silhouette_scores.append(silhouette_score(points, labels))
        else:
            silhouette_scores.append(-1)
            
        # Inertia (distortion)
        distortions.append(kmedoids.inertia_)
    
    # Find optimal k using silhouette (max score)
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    
    # Find optimal k using elbow method
    kneedle = KneeLocator(k_range, distortions, curve='convex', direction='decreasing')
    optimal_k_elbow = kneedle.elbow
    
    # Combine both methods (you can customize this logic)
    optimal_k = max(optimal_k_silhouette, optimal_k_elbow)
    
    return optimal_k

def coreset_sampling_adaptive(points, k=None, k_range=np.arange(10,51,5)):
    """
    Adaptive coreset sampling that automatically determines optimal K
    
    Parameters:
    - points: Input data points
    - k: Optional fixed number of clusters (if None, determined automatically)
    - k_range: Range of k values to test for automatic determination
    
    Returns:
    - strucs_clustered_indices: List of lists containing indices for each cluster
    - coreset_idx: Array of medoid indices
    - optimal_k: The determined number of clusters
    """
    if k is None:
        optimal_k = int(determine_optimal_k(points, k_range))
    else:
        optimal_k = k
    
    kmedoids = KMedoids(n_clusters=optimal_k, metric='euclidean')
    kmedoids.fit(np.array(points))
    
    coreset_idx = kmedoids.medoid_indices_
    cluster_labels = kmedoids.labels_
    
    strucs_clustered_indices = [
        [node for node, cluster in enumerate(cluster_labels) if cluster == c] 
        for c in range(optimal_k)
    ]
    
    return strucs_clustered_indices, np.array(coreset_idx), optimal_k

def extend_fsr_seq(fsr_seq,sequence,extend_length = None):
    if extend_length == None:
        extend_length = len(fsr_seq)
    fsr_seq_start = sequence.index(fsr_seq[0:8])
    fsr_seq_end = fsr_seq_start+len(fsr_seq)
    fsr_seq_start_extend = max(0,fsr_seq_start-extend_length//2)
    fsr_seq_end_extend = min(len(sequence),fsr_seq_end+extend_length//2 - min(0, fsr_seq_start-extend_length//2 )  )
    fsr_seq_extend = sequence[fsr_seq_start_extend:fsr_seq_end_extend  ]
    return(fsr_seq_extend)

def load_fasta(fil):
    seqs, IDs =[], []
    with open(fil) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = ''.join([x for x in record.seq])
                IDs.append(record.id)
                seqs.append(seq)
    return IDs, seqs

def extend_interval_symmetric(X1, X2, U1, U2):
    """
    Extend interval [X1, X2] symmetrically by twice its length,
    ensuring the result stays within [U1, U2].
    
    Args:
        X1, X2: Current interval bounds
        U1, U2: Overall bounds
    
    Returns:
        tuple: (new_X1, new_X2) of the extended interval
    """
    current_length = X2 - X1
    target_length = 3 * current_length  # original + 2x extension = 3x
    
    # Calculate center of current interval
    center = (X1 + X2) / 2
    
    # Calculate ideal extended bounds
    ideal_X1 = center - target_length / 2
    ideal_X2 = center + target_length / 2
    
    # Apply bounds constraints
    new_X1 = max(U1, min(ideal_X1, U2))
    new_X2 = max(U1, min(ideal_X2, U2))
    
    # If ideal extension exceeds bounds, shift to fit
    if ideal_X1 < U1:
        new_X2 = min(U2, new_X2 + (U1 - ideal_X1))
        new_X1 = U1
    elif ideal_X2 > U2:
        new_X1 = max(U1, new_X1 - (ideal_X2 - U2))
        new_X2 = U2
    
    return new_X1, new_X2


def write_fasta(names, seqs, outfile='tmp.fasta'):
        with open(outfile,'w') as f:
                for nm, seq in list(zip(names, seqs)):
                        f.write(">%s\n%s\n" % (nm, seq))

def encode_seqs(seqs, max_len=108, alphabet=None):
    if alphabet is None:
        alphabet = "ACDEFGHIKLMNPQRSTVWY-"
    arr = np.zeros([len(seqs), max_len, len(alphabet)])
    for j, seq in enumerate(seqs):
        for i,char in enumerate(seq):
            for k, res in enumerate(alphabet):
                if char==res:
                    arr[j,i,k]+=1
    return arr.reshape([len(seqs), max_len*len(alphabet)])

def encode_seqs2(seqs, max_len=108, alphabet=None):
    if alphabet is None:
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
    arr = np.zeros([len(seqs), max_len])
    for j, seq in enumerate(seqs):
        for i,char in enumerate(seq):
            if char in alphabet:
                arr[j,i]= alphabet.index(char)
            elif char =='-':
                arr[j,i]= 21
            else:
                arr[j,i]=len(alphabet)
    return arr


def find_matching_atoms(traj1, traj2):
    # Build sets of (residue index, atom name) tuples for each trajectory
    set1 = {(atom.residue.index, atom.name) for atom in traj1.topology.atoms if atom.name in ['N', 'CA', 'C', 'O']}
    set2 = {(atom.residue.index, atom.name) for atom in traj2.topology.atoms if atom.name in ['N', 'CA', 'C', 'O']}
    # Find the intersection of these sets to identify common atoms
    common_atoms = set1.intersection(set2)
    # Convert these sets into indices for trajectory analysis
    indices1 = [atom.index for atom in traj1.topology.atoms if (atom.residue.index, atom.name) in common_atoms]
    indices2 = [atom.index for atom in traj2.topology.atoms if (atom.residue.index, atom.name) in common_atoms]
    return indices1, indices2

def find_large_displacements(pdb_file1, pdb_file2, threshold=1.0):
    # Load the PDB files as trajectories
    traj1 = md.load(pdb_file1)
    traj2 = md.load(pdb_file2)
    # Find indices of matching atoms
    indices1, indices2 = find_matching_atoms(traj1, traj2)
    # Superpose traj2 onto traj1 using the matching atoms
    traj2.superpose(traj1, atom_indices=indices2, ref_atom_indices=indices1)
    # Calculate distances for the matching atoms in the first frame
    distances = 10 * np.linalg.norm(traj1.xyz[0, indices1] - traj2.xyz[0, indices2], axis=1)
    # Find atoms with distances greater than the threshold
    atoms_above_threshold = np.where(distances > threshold)[0]
    # Extract atom details from the topology
    selected_atoms = [traj1.topology.atom(indices1[i]) for i in atoms_above_threshold]
    return selected_atoms, distances[atoms_above_threshold],indices1, indices2

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

def one_hot(x,cat=None):
  if cat is None: cat = np.max(x)+1
  oh = np.concatenate((np.eye(cat),np.zeros([1,cat])))
  return oh[x]

def sweep_dbscan(msa, deletion_matrix, verbose=True, min_eps=3, max_eps=20, eps_step=0.5,min_samples=5):
  '''Input: MSA with shape N, L (N=num sequences, L=length) where chars are integers
  Ouptut: list of MSA clusters. each is shape [M,L] where M is variable, is the length of the new clustered MSA.
  Each cluster MSA has query sequence at start
  '''

  N, L = msa.shape
  ohe_msa = one_hot(msa).reshape(N,-1)
  eps_test_vals=np.arange(min_eps, max_eps+eps_step, eps_step)
  smaller_split = np.random.choice(range(len(msa)), int(len(msa)/4))
  test_split = ohe_msa[smaller_split]
  n_clusters=[]
  if verbose:
    print('eps\tnum clusters\tn not clustered')
  for eps in eps_test_vals:
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(test_split)
    n_clust = len(set(clustering.labels_))
    n_not_clustered = len(clustering.labels_[np.where(clustering.labels_==-1)])
    if verbose:
       print('%.2f\t%d\t%d' % (eps, n_clust, n_not_clustered))
    n_clusters.append(n_clust)
    if eps>10 and n_clust==1:
        break

  eps_max = eps_test_vals[np.argmax(n_clusters)]
  if verbose: print("eps max:", eps_max)
  clustering = DBSCAN(eps=eps_max, min_samples=min_samples).fit(ohe_msa)
  clusters = [x for x in list(set(clustering.labels_)) if x>=0]
  n_not_clustered = len(clustering.labels_[np.where(clustering.labels_==-1)])
  if verbose:
    print('%d clusters' % len(clusters))
    print('%d seqs not clustered' % n_not_clustered)

  clustered_msas=[]
  clustered_dtxs=[]
  query_seq=msa[0]
  query_dtx = deletion_matrix[0]

  for c in clusters:
    cluster_msa = msa[np.where(clustering.labels_==c)]
    cluster_dtx = deletion_matrix[np.where(clustering.labels_==c)]
    cluster_msa = np.concatenate([[query_seq], cluster_msa])
    cluster_dtx = np.concatenate([[query_dtx], cluster_dtx])

    clustered_msas.append(cluster_msa)
    clustered_dtxs.append(cluster_dtx)

  return clustered_msas, clustered_dtxs


def make_msa_arr(msa, query_seq):
    # get dists between each member of msa and query seq
    dist_arr = [1-len(np.where(x- query_seq!=0)[0])/len(query_seq) for x in msa]

    sorted_msa = [msa[x] for x in np.argsort(dist_arr)]
    sorted_dist_arr = [dist_arr[x] for x in np.argsort(dist_arr)]

    plotting_arr = np.zeros([len(msa), len(query_seq)])*np.nan

    ctr=0
    for dist, msa_seq in list(zip(sorted_dist_arr, sorted_msa)):
        for j, char in enumerate(msa_seq):
          if char!=21:
            plotting_arr[ctr,j] = dist
        ctr+=1

    return plotting_arr

def plot_cluster_msas(list_of_cluster_msas, pad_size=20, figsize=(8,8),dpi=100,sort_by_dist=False):
    query_seq=list_of_cluster_msas[0][0]
    arrs = [make_msa_arr(msa[1:], query_seq) for msa in list(reversed(list_of_cluster_msas))] # exclude 0 since that's the query seq
    mean_cluster_dists = [np.nanmean(x) for x in arrs]

    if sort_by_dist:
        arrs = [arrs[x] for x in np.argsort(mean_cluster_dists)]

    pad = np.zeros([pad_size, len(query_seq)])*np.nan
    padded_arrs = [np.vstack([x, pad]) for x in arrs]

    plotting_arr = np.vstack(padded_arrs)

    yaxis_size,  xaxis_size= plotting_arr.shape

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(
        plotting_arr,
    interpolation="nearest",
    aspect="auto",
    cmap="rainbow_r",
    vmin=0,
    vmax=1,
    origin="lower",
    extent=(0,xaxis_size, 0, yaxis_size),)

    for x in np.cumsum([len(x) for x in padded_arrs]):
        plt.axhline(x-0.5*pad_size,color='k',linewidth=0.5)

    plt.xlim(0, xaxis_size)
    plt.ylim(-0.5*pad_size, yaxis_size)
    plt.colorbar(label="Sequence identity to query",aspect=2, fraction=0.05)
    plt.xlabel("Positions")
    plt.ylabel("Sequences")
    
def get_msa_query_dist(msa, query_seq):
    # get dists between each member of msa and query seq
    return [1-len(np.where(x- query_seq!=0)[0])/len(query_seq) for x in msa]

def seq_entropy(x):
    """
    x is assumed to be an (nsignals, nsamples) array containing integers between
    0 and n_unique_vals
    """
    x = np.transpose(np.atleast_2d(x))
    nrows, ncols = x.shape
    nbins = x.max() + 1

    # count the number of occurrences for each unique integer between 0 and x.max()
    # in each row of x
    counts = np.vstack([np.bincount(row, minlength=nbins) for row in x])
    #print(counts.shape)

    # divide by number of columns to get the probability of each unique value
    p = counts / float(ncols)
    #print(p)

    # compute Shannon entropy in bits
    return -np.sum(np.where(p == 0, 0, p * np.log2(p)), axis=1)

def seq_prob(x):
    """
    x is assumed to be an (nsignals, nsamples) array containing integers between
    0 and n_unique_vals
    """
    x = np.transpose(np.atleast_2d(x))
    nrows, ncols = x.shape
    nbins = x.max() + 1

    p = np.zeros([nrows, 22])
    for row in range(nrows):
      for col in range(ncols):
        p[row,x[row,col]] += 1
      p[row] = p[row]/ncols
    return p


def seq_entropy_sd_vec(p):
#     """
#     x is assumed to be an (nsignals, nsamples) array containing integers between
#     0 and n_unique_vals
#     """
    entropy_vec = -np.where(p == 0, 0, p * np.log2(p))
    sd_inv_vec = np.where((p == 0)+(p ==1), 0, (1/p/(1-p)) )
    sd_inv_vec = np.where(sd_inv_vec<0.05, 0.05, sd_inv_vec)
    return entropy_vec,sd_inv_vec

def seq_entropy_exp_sd(p):
    entropy_vec = -np.where(p == 0, 0, p * np.log2(p))
    delta_p_exp = np.sum(2*p*(1-p), 1)
    delta_p_sd = np.sqrt(np.sum(4*p*((1-p)**2), 1)-delta_p_exp**2)
    delta_p_sd_inv = np.where(delta_p_sd<0.05, 1/0.05, 1/delta_p_sd)
    return entropy_vec,delta_p_exp,delta_p_sd_inv




def select_coreset(distance_matrix, k):
    """
    Select a coreset using the k-Center Greedy Algorithm.
    Args:
        distance_matrix (np.ndarray): A symmetric distance matrix of shape (n, n),
                                      where n is the number of data points.
        k (int): The number of points to select for the coreset.
    Returns:
        coreset_indices (list): Indices of the selected coreset points.
    """
    n = distance_matrix.shape[0]  # Number of data points
    coreset_indices = []  # List to store the indices of the coreset points
    # Step 1: Select the first point randomly
    first_point = np.random.choice(n)
    coreset_indices.append(first_point)
    # Step 2: Iteratively select the farthest point from the current coreset
    for _ in range(1, k):
        # Compute the minimum distance from each point to the current coreset
        min_distances = np.min(distance_matrix[:, coreset_indices], axis=1)
        # Select the point with the maximum minimum distance
        next_point = np.argmax(min_distances)
        coreset_indices.append(next_point)
    return coreset_indices

def seq2mat(seq):
    mat = np.zeros([len(seq),22])
    for i in range(len(seq)):
        mat[i,seq[i]] = 1
    return mat

#@title Setup code to compare to experimental
#@markdown

def get_pdb(pdb_code=""):
  if pdb_code is None or pdb_code == "":
    upload_dict = files.upload()
    pdb_string = upload_dict[list(upload_dict.keys())[0]]
    with open("tmp.pdb","wb") as out: out.write(pdb_string)
    return "tmp.pdb"
  elif os.path.isfile(pdb_code):
    return pdb_code
  elif len(pdb_code) == 4:
    os.makedirs("tmp",exist_ok=True)
    os.system(f"wget -qnc https://files.rcsb.org/download/{pdb_code}.pdb -P tmp/")
    return f"tmp/{pdb_code}.pdb"
  else:
    os.makedirs("tmp",exist_ok=True)
    os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v4.pdb -P tmp/")
    return f"tmp/AF-{pdb_code}-F1-model_v4.pdb"

def get_pdb_and_chain(pdb_code="",chain="A",len=None,offset=None):
  get_pdb(pdb_code)
  from colabdesign.shared.protein import pdb_to_string, renum_pdb_str
  string = pdb_to_string('tmp/{0}.pdb'.format(pdb_code),chains=chain,models=1)
  if offset is not None:
    string = renum_pdb_str(string,[len],offset=offset)
  else:
    string = renum_pdb_str(string,[len])

  with open('tmp/{0}{1}.pdb'.format(pdb_code,chain),'w') as f:
    f.write(string)

def lik(prob,shuf_msa):
    #prob: len(seq)*22
    #print(prob.shape)
    lik_list = []
    for seq in shuf_msa:
        lik_vec = np.sum(seq2mat(seq)*prob ,1)
        llik =  np.sum(np.where(lik_vec == 0, -1e6, np.log(lik_vec) ))
        lik_list.append(llik)
    return np.array(lik_list)


if not os.path.exists('TMscore_cpp'):
  os.system('wget https://zhanggroup.org/TM-score/TMscore_cpp.gz')
  os.system('gunzip TMscore_cpp.gz')
  os.system('chmod +x TMscore_cpp')

"""
Compute TM Scores between two PDBs and parse outputs
pdb_pred -- The path to the predicted PDB
pdb_native -- The path to the native PDB
test_len -- run asserts that the input and output should have the same length
"""
import subprocess

# tm_re = re.compile(r'TM-score[\s]*=[\s]*(\d.\d+)')
# ref_len_re = re.compile(r'Length=[\s]*(\d+)[\s]*\(by which all scores are normalized\)')
# common_re = re.compile(r'Number of residues in common=[\s]*(\d+)')
# super_re = re.compile(r'\(":" denotes the residue pairs of distance < 5\.0 Angstrom\)\\n([A-Z\-]+)\\n[" ", :]+\\n([A-Z\-]+)\\n')


def compute_tmscore(pdb_pred, pdb_native, test_len=False):
  cmd = (['./TMscore_cpp', pdb_pred, pdb_native])
  tmscore_output = str(subprocess.check_output(cmd))
  try:
    tm_out = float(tm_re.search(tmscore_output).group(1))
    reflen = int(ref_len_re.search(tmscore_output).group(1))
    common = int(common_re.search(tmscore_output).group(1))

    seq1 = super_re.search(tmscore_output).group(1)
    seq2 = super_re.search(tmscore_output).group(1)
  except Exception as e:
    print("Failed on: " + " ".join(cmd))
    raise e

  if test_len:
    assert reflen == common, cmd
    assert seq1 == seq2, cmd
    assert len(seq1) == reflen, cmd

  return tm_out

def compute_tmscore_(row, native_pdb):
  return compute_tmscore(row['pdb_path'],native_pdb)

def get_contacts(pdb_file, i=0, f=-1):
    obj = md.load_pdb(pdb_file)
    distances, pairs = md.compute_contacts(obj, scheme='ca')
    arr = md.geometry.squareform(distances, pairs)[0]
    arr = arr[i:f, i:f]
    arr = arr.flatten()
    return arr

def save_plot_cluster_msas(list_of_cluster_msas, save_filepath,pad_size=20, figsize=(8,8),dpi=100,sort_by_dist=False):
    query_seq=list_of_cluster_msas[0][0]
    arrs = [make_msa_arr(msa[1:], query_seq) for msa in list(reversed(list_of_cluster_msas))] # exclude 0 since that's the query seq
    mean_cluster_dists = [np.nanmean(x) for x in arrs]

    if sort_by_dist:
        arrs = [arrs[x] for x in np.argsort(mean_cluster_dists)]

    pad = np.zeros([pad_size, len(query_seq)])*np.nan
    padded_arrs = [np.vstack([x, pad]) for x in arrs]

    plotting_arr = np.vstack(padded_arrs)

    yaxis_size,  xaxis_size= plotting_arr.shape

    
    plt.imshow(
        plotting_arr,
    interpolation="nearest",
    aspect="auto",
    cmap="rainbow_r",
    vmin=0,
    vmax=1,
    origin="lower",
    extent=(0,xaxis_size, 0, yaxis_size),)

    for x in np.cumsum([len(x) for x in padded_arrs]):
        plt.axhline(x-0.5*pad_size,color='k',linewidth=0.5)

    plt.xlim(0, xaxis_size)
    plt.ylim(-0.5*pad_size, yaxis_size)
    plt.colorbar(label="Sequence identity to query",aspect=2, fraction=0.05)
    plt.xlabel("Positions")
    plt.ylabel("Sequences")
    plt.savefig(save_filepath, bbox_inches='tight')
    plt.cla()

aa_long_short = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                 'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
                 'ASX': 'B', 'XLE': 'J', 'PYL': 'O', 'SEC': 'U', 'UNK': 'X', 'GLX': 'Z'}



aa_short_long = {y: x for x, y in aa_long_short.items()}


# Compute tmscores of two structures, interface to tmscore module
# Input:
# pdb_file1, pdb_file2 - names of two input pdb files (without chain name)
# chain1, chain2 - names of two chains
# Output:
#
def compute_tmscore_withIDs(pdb_file1, pdb_file2, chain1=None, chain2=None):
    #print("Compute tmscore: File 1:", pdb_file1, ", File 2:", pdb_file2, " ; Chains:", chain1, chain2)
    # Fetch or read structures
    if len(pdb_file1) == 4:  # PDB ID
        s1 = get_structure(rcsb.fetch(pdb_file1, "cif"), model=1)
    else:
        s1 = PDBFile.read(pdb_file1).get_structure(model=1)
    # Find all CA atoms
    ca_mask1 = (s1.atom_name == 'CA')
    # Find all residues that have a CA atom
    residues_with_ca = s1.res_id[ca_mask1]
    unique_residues_with_ca = np.unique(residues_with_ca)
    all_residues_mask = np.isin(s1.res_id, unique_residues_with_ca)
    # Filter the atom array to only include residues with a CA atom
    filtered_s1 = s1[all_residues_mask]

    if len(pdb_file2) == 4:  # PDB ID
        s2 = get_structure(rcsb.fetch(pdb_file2, "cif"), model=1)
    else:
        s2 = PDBFile.read(pdb_file2).get_structure(model=1)

    ca_mask2 = (s2.atom_name == 'CA')
    residues_with_ca = s2.res_id[ca_mask2]
    unique_residues_with_ca = np.unique(residues_with_ca)
    all_residues_mask = np.isin(s2.res_id, unique_residues_with_ca)
    # Filter the atom array to only include residues with a CA atom
    filtered_s2 = s2[all_residues_mask]

    # Process chains and sequences
    pdb_dists1, pdb_contacts1, pdb_seq1, pdb_good_res_inds1, coords1 = \
        read_seq_coord_contacts_from_pdb(filtered_s1, chain=chain1)
    pdb_dists2, pdb_contacts2, pdb_seq2, pdb_good_res_inds2, coords2 = \
        read_seq_coord_contacts_from_pdb(filtered_s2, chain=chain2)
    # Perform alignment
    res = tm_align(coords1, coords2, pdb_seq1, pdb_seq2)
    #print("Normalized TM-score (chain1):", round(res.tm_norm_chain1, 3))
    return res.tm_norm_chain1

def compute_local_tmscore(pdb_file1, pdb_file2, chain1=None, chain2=None, start_res=None, end_res=None):
    """
    Compute the local TM-score for a specific region (residues start_res to end_res) of pdb_seq2.
    Args:
        pdb_file1 (str): Path to the first PDB file or PDB ID.
        pdb_file2 (str): Path to the second PDB file or PDB ID.
        chain1 (str, optional): Chain ID for the first structure.
        chain2 (str, optional): Chain ID for the second structure.
        start_res (int): Start residue index for the local region.
        end_res (int): End residue index for the local region.
    Returns:
        float: Local TM-score for the specified region.
    """
    # Fetch or read structures
    if len(pdb_file1) == 4:  # PDB ID
        s1 = get_structure(rcsb.fetch(pdb_file1, "cif"), model=1)
    else:
        s1 = PDBFile.read(pdb_file1).get_structure(model=1)
    # Find all CA atoms
    ca_mask1 = (s1.atom_name == 'CA')
    # Find all residues that have a CA atom
    residues_with_ca = s1.res_id[ca_mask1]
    unique_residues_with_ca = np.unique(residues_with_ca)
    all_residues_mask = np.isin(s1.res_id, unique_residues_with_ca)
    # Filter the atom array to only include residues with a CA atom
    filtered_s1 = s1[all_residues_mask]

    if len(pdb_file2) == 4:  # PDB ID
        s2 = get_structure(rcsb.fetch(pdb_file2, "cif"), model=1)
    else:
        s2 = PDBFile.read(pdb_file2).get_structure(model=1)

    ca_mask2 = (s2.atom_name == 'CA')
    residues_with_ca = s2.res_id[ca_mask2]
    unique_residues_with_ca = np.unique(residues_with_ca)
    all_residues_mask = np.isin(s2.res_id, unique_residues_with_ca)
    # Filter the atom array to only include residues with a CA atom
    filtered_s2 = s2[all_residues_mask]

    # Process chains and sequences
    pdb_dists1, pdb_contacts1, pdb_seq1, pdb_good_res_inds1, coords1 = \
        read_seq_coord_contacts_from_pdb(filtered_s1, chain=chain1)
    pdb_dists2, pdb_contacts2, pdb_seq2, pdb_good_res_inds2, coords2 = \
        read_seq_coord_contacts_from_pdb(filtered_s2, chain=chain2)

    if  start_res ==None:
      start_res = 1
    if end_res == None:
      end_res = len(pdb_seq2)
    # Perform sequence alignment
    alignments = pairwise2.align.globalxx(pdb_seq1, pdb_seq2)  # Global alignment
    best_alignment = alignments[0]  # Take the best alignment
    aligned_seq1, aligned_seq2 = best_alignment[0], best_alignment[1]

    # Find the corresponding region in pdb_seq1
    def map_residues(aligned_seq1, aligned_seq2, start_res, end_res):
        """
        Map residues from pdb_seq2 to pdb_seq1 using the alignment.
        """
        # Convert 1-based residue indices to 0-based indices
        start_res_idx = start_res - 1
        end_res_idx = end_res - 1

        # Find the corresponding positions in the aligned sequences
        pos_in_seq2 = 0
        pos_in_seq1 = 0
        mapped_start, mapped_end = None, None

        for a1, a2 in zip(aligned_seq1, aligned_seq2):
            if a2 != '-':  # Not a gap in seq2
                if pos_in_seq2 == start_res_idx:
                    mapped_start = pos_in_seq1
                if pos_in_seq2 == end_res_idx:
                    mapped_end = pos_in_seq1
                    break
                pos_in_seq2 += 1
            if a1 != '-':  # Not a gap in seq1
                pos_in_seq1 += 1

        return mapped_start, mapped_end

    # Map residues 81-91 in pdb_seq2 to pdb_seq1
    mapped_start, mapped_end = map_residues(aligned_seq1, aligned_seq2, start_res, end_res)

    if mapped_start is None or mapped_end is None:
        raise ValueError("Could not map the specified region between the sequences.")

    # Extract coordinates for the local regions
    def extract_local_coords(coords, seq, start_res, end_res):
        """
        Extract coordinates for a specific region of the sequence.
        """
        local_coords = []
        for i, res in enumerate(seq):
            if start_res <= i + 1 <= end_res:  # Residue indices are 1-based
                local_coords.append(coords[i])
        return np.array(local_coords)

    # Extract local coordinates for pdb_seq2
    local_coords2 = extract_local_coords(coords2, pdb_seq2, start_res, end_res)
# Extract corresponding region in pdb_seq1
    local_coords1 = extract_local_coords(coords1, pdb_seq1, mapped_start + 1, mapped_end + 1)

    # Perform alignment for the local region
    res = tm_align(local_coords1, local_coords2, pdb_seq1[mapped_start:mapped_end+1], pdb_seq2[start_res-1:end_res])

    # Return the local TM-score
    return res.tm_norm_chain1

def get_coords(pdbfile,fs_range):
    """
    parameters:
    pdbfile - path to pdbfile
    fs_range - range of residues at the fold-switching region, given as string - "112-162"
    returns:
    numpy array of coords
    string of seqs in 1-letter-code
    """

    seq = ""
    struct = pdbParser.get_structure('x',str(pdbfile))
    coords = []
    seq_dict = {}

    # for residues within a certain range, using numpy to save the coords
    # and save the sequence as a dict and then sorted list of tuples
    # return the coords and the seq

    # convert str to residue range for the fs region
    (start,stop) = fs_range.split("-")
    res_range = range(int(start),int(stop)+1)

    for atom in struct.get_atoms():
            residue = atom.get_parent() # from atom we can get the parent residue
            res_id = residue.get_id()[1]
            resname = residue.get_resname()
            if res_id in res_range and atom.get_name()=="CA":
                    x,y,z = atom.get_coord()
                    coords.append([x,y,z])
                    if res_id not in seq_dict:
                            seq_dict[res_id]=aa_long_short[resname]

    # convert to np array
    coords_np = np.array(coords)
    # sort the seq_dict by keys a.k.a res_ids
    sorted_data = sorted(seq_dict.items())
    for i in sorted_data:
            seq+=i[1]

    return  coords_np,seq


def get_tmscore(coords1,seq1,modelpath,res_range):
    """
    parameters:
    coords1, seq1 - the numpy array of PDB coords and its seqs
    predfilepath - path for predicted files
    res_range - fs range in predicted models

    returns:
    tmscore list

    """
    #modelpath = Path(model)
    coords2, seq2 = get_coords(modelpath,res_range)
    res = tm_align(coords1, coords2, seq1, seq2)
    tmscore = round(res.tm_norm_chain1,2) # wrt to model
    return tmscore

def compute_fsr_tmscore2(pdb_file1, pdb_file2, res_range, chain1=None, chain2=None):
    """
    Compute the local TM-score for a specific region (residues start_res to end_res) of pdb_seq2.
    Args:
        pdb_file1 (str): Path to the first PDB file or PDB ID.
        pdb_file2 (str): Path to the second PDB file or PDB ID.
        chain1 (str, optional): Chain ID for the first structure.
        chain2 (str, optional): Chain ID for the second structure.
        start_res (int): Start residue index for the local region.
        end_res (int): End residue index for the local region.
    Returns:
        float: Local TM-score for the specified region.
    """
    # Fetch or read structures
    if len(pdb_file1) == 4:  # PDB ID
        s1 = get_structure(rcsb.fetch(pdb_file1, "cif"), model=1)
    else:
        s1 = PDBFile.read(pdb_file1).get_structure(model=1)
        
    coords1,seq1 = get_coords(pdb_file1,res_range)
    tmscore = get_tmscore(coords1,seq1,pdb_file2,res_range)
    return tmscore

def read_seq_coord_contacts_from_pdb(
        structure: AtomArray,
        distance_threshold: float = 8.0,
        chain: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, str, np.ndarray, np.ndarray]:
    """
    Extract distances, contacts, sequence, and coordinates from an AtomArray structure.
    Parameters:
    - structure (AtomArray): Biotite structure containing atomic data.
    - distance_threshold (float): Cutoff distance for contacts in Å.
    - chain (Optional[str]): Specific chain to process (if None, all chains are used).

    Returns:
    - dist (np.ndarray): Pairwise distance matrix of Cα atoms.
    - contacts (np.ndarray): Binary contact map (1 for contact, 0 otherwise).
    - pdb_seq (str): Protein sequence as a string.
    - good_res_ids (np.ndarray): Indices of valid residues.
    - CA_coords (np.ndarray): Coordinates of Cα atoms.
    """
    # Filter by chain ID if specified
    if chain is not None:
        structure = structure[structure.chain_id == chain]
    # Filter amino acids only
    amino_acid_filter = filter_amino_acids(structure)
    structure = structure[amino_acid_filter]
    # Get residues and their starting indices
    residues, residue_starts = get_residues(structure)
    # Map residues to single-letter codes (fallback to "X" for unknown residues)
    pdb_seq = "".join(aa_long_short.get(res,"X") for res in residue_starts)
    # Extract Cα coordinates
    ca_mask = structure.atom_name == "CA"
    CA_coords = structure.coord[ca_mask]
    if len(CA_coords) == 0:
        raise ValueError("No Cα atoms found in structure.")
    # Extract valid residue indices
    good_res_ids = residue_starts
    # Calculate pairwise distances for Cα atoms
    dist = squareform(pdist(CA_coords))
    # Create binary contact map based on distance threshold
    contacts = (dist < distance_threshold).astype(int)
    return dist, contacts, pdb_seq, good_res_ids, CA_coords

def compute_fsr_tmscore(pdb_file1, pdb_file2, fs_sequence, chain1=None, chain2=None):
    """
    Compute the local TM-score for a specific region (residues start_res to end_res) of pdb_seq2.
    Args:
        pdb_file1 (str): Path to the first PDB file or PDB ID.
        pdb_file2 (str): Path to the second PDB file or PDB ID.
        chain1 (str, optional): Chain ID for the first structure.
        chain2 (str, optional): Chain ID for the second structure.
        start_res (int): Start residue index for the local region.
        end_res (int): End residue index for the local region.
    Returns:
        float: Local TM-score for the specified region.
    """
    # Fetch or read structures
    if len(pdb_file1) == 4:  # PDB ID
        s1 = get_structure(rcsb.fetch(pdb_file1, "cif"), model=1)
    else:
        s1 = PDBFile.read(pdb_file1).get_structure(model=1)
    # Find all CA atoms
    ca_mask1 = (s1.atom_name == 'CA')
    # Find all residues that have a CA atom
    residues_with_ca = s1.res_id[ca_mask1]
    unique_residues_with_ca = np.unique(residues_with_ca)
    all_residues_mask = np.isin(s1.res_id, unique_residues_with_ca)
    # Filter the atom array to only include residues with a CA atom
    filtered_s1 = s1[all_residues_mask]
    # Process chains and sequences
    pdb_dists1, pdb_contacts1, pdb_seq1, pdb_good_res_inds1, coords1 = \
        read_seq_coord_contacts_from_pdb(s1, chain=chain1)
    #print(pdb_file1)
    #print(len(coords1 ))
    #print(len(pdb_seq1 ))
    fs_idx = align_fsr(pdb_seq1,fs_sequence)
    fs_idx[1] = min([fs_idx[1],len(coords1 ),len(pdb_seq1 ) ])
    coords1 = coords1[fs_idx[0]-1:fs_idx[1]]
    pdb_seq1 = pdb_seq1[fs_idx[0]-1:fs_idx[1]]
    if len(pdb_file2) == 4:  # PDB ID
        s2 = get_structure(rcsb.fetch(pdb_file2, "cif"), model=1)
    else:
        s2 = PDBFile.read(pdb_file2).get_structure(model=1)
    ca_mask2 = (s2.atom_name == 'CA')
    residues_with_ca = s2.res_id[ca_mask2]
    unique_residues_with_ca = np.unique(residues_with_ca)
    all_residues_mask = np.isin(s2.res_id, unique_residues_with_ca)
    # Filter the atom array to only include residues with a CA atom
    filtered_s2 = s2[all_residues_mask]
    pdb_dists2, pdb_contacts2, pdb_seq2, pdb_good_res_inds2, coords2 = \
        read_seq_coord_contacts_from_pdb(s2, chain=chain2)
    fs_idx = align_fsr(pdb_seq2,fs_sequence)
    fs_idx[1] = min([fs_idx[1],len(coords2 ),len(pdb_seq2 ) ])
    coords2 = coords2[fs_idx[0]-1:fs_idx[1]]
    pdb_seq2 = pdb_seq2[fs_idx[0]-1:fs_idx[1]]
    res = tm_align(coords1, coords2, pdb_seq1, pdb_seq2)
    return res.tm_norm_chain1

def align_fsr(pdb_seq,fsr_seq):
    a = pairwise2.align.localxs(pdb_seq,fsr_seq,-1,-0.5)
    i_start = None
    i_end = None
    leading_gaps = 0 # number of leading gaps of pdb_seq in alignment (correcting factor)
    read_status = {0: False, 1: False} 
    # find i_start
    for i in range(len(a[0][0])):
        # read statuses
        for j in read_status.keys():
            if not read_status[j] and a[0][j][i] != "-":
                read_status[j] = True
                if j == 0:
                    leading_gaps = i
        if read_status[0] and read_status[1]:
            i_start = i+1-leading_gaps
            break
    # find i_end
    read_status = {0: False, 1: False} 
    for i in reversed(range(len(a[0][0]))):
        # read statuses
        for j in read_status.keys():
            if not read_status[j] and a[0][j][i] != "-":
                read_status[j] = True
        if read_status[0] and read_status[1]:
            i_end = i+1-leading_gaps
            break
    return [i_start,i_end]


def plot_pca_medoids(x, y, df, medoids, plot_path):
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=x,y=y, hue='cluster', data=df, palette='tab10',linewidth=0)
    plt.scatter(medoids[x],medoids[y], color='black', marker='*', s=150, label='medoids')
    plt.legend(bbox_to_anchor=(1,1), frameon=False)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(plot_path+"PCA_with_medoids.png", bbox_inches='tight',dpi=300)
    plt.cla()
    plt.figure(figsize=(6,5))
    plt.scatter(df[x],df[y], c=df['plddt'],cmap='rainbow_r', s=50,vmin = 0.5,vmax=0.9)
    plt.colorbar(label='plddt')
    # norm = Normalize(vmin=0.5, vmax=0.9)
    # cmap = sns.color_palette("rainbow_r", as_cmap=True)
    # sns.scatterplot(x=x,y=y, hue='plddt', hue_norm=norm, data=df, palette=cmap)
    plt.scatter(medoids[x],medoids[y], color='black', marker='*', s=150, label='medoids')
    plt.legend(bbox_to_anchor=(1,1), frameon=False)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(plot_path+"PCA_with_medoids_plddt.png", bbox_inches='tight',dpi=300)
    plt.cla()

def subset_string(string, indices):
    return ''.join(string[i] for i in indices)

def split_fasta(input_file, output_dir):
    for record in SeqIO.parse(input_file, "fasta"):
        seq_id = record.id
        seq_file = os.path.join(output_dir, f"{seq_id}.fasta")
        with open(seq_file, "w") as f:
            SeqIO.write(record, f, "fasta")

def run_hhsearch(output_dir, hhsearch_db,cpu =2):
    print("Running HHsearch for each sequence...")
    for seq_file in os.listdir(output_dir):
        if seq_file.endswith(".fasta"):
            seq_path = os.path.join(output_dir, seq_file)
            output_hhr = os.path.join(output_dir, f"{os.path.splitext(seq_file)[0]}.hhr")
            command = f"hhblits -cpu {cpu} -i {seq_path} -d {hhsearch_db} -o {output_hhr} -cov 75 -qid 75"
            subprocess.run(command, shell=True, check=True)
    print("HHsearch completed.")

def extract_top_pdb_ids(output_dir,ntop=5,pvalue = 0.01):
    pdb_ids = []
    for hhr_file in os.listdir(output_dir):
        if hhr_file.endswith(".hhr"):
            hhr_path = os.path.join(output_dir, hhr_file)
            with open(hhr_path, "r") as f:
                start = 1
                for line in f:
                    if line.startswith("  %d"%start):  # Top hit line
                        pdb_id = line.split()[1]#.split("_")[0]  # Extract PDB ID
                        pvalues = float(line.split()[-8])
                        if pvalues>pvalue:
                            break
                        if start == ntop:
                            break
                        pdb_ids.append(pdb_id)
                        start = start+1
    return pdb_ids



def sym_w(w):
  '''symmetrize input matrix of shape (x,y,x,y)'''
  x = w.shape[0]
  w = w * np.reshape(1-np.eye(x),(x,1,x,1))
  w = w + tf.transpose(w,[2,3,0,1])
  return w

def opt_adam(loss, name, var_list=None, lr=1.0, b1=0.9, b2=0.999, b_fix=False):
  # adam optimizer
  # Note: this is a modified version of adam optimizer. More specifically, we replace "vt"
  # with sum(g*g) instead of (g*g). Furthmore, we find that disabling the bias correction
  # (b_fix=False) speeds up convergence for our case.
  
  if var_list is None: var_list = tf.trainable_variables() 
  gradients = tf.gradients(loss,var_list)
  if b_fix: t = tf.Variable(0.0,"t")
  opt = []
  for n,(x,g) in enumerate(zip(var_list,gradients)):
    if g is not None:
      ini = dict(initializer=tf.zeros_initializer,trainable=False)
      mt = tf.get_variable(name+"_mt_"+str(n),shape=list(x.shape), **ini)
      vt = tf.get_variable(name+"_vt_"+str(n),shape=[], **ini)
      
      mt_tmp = b1*mt+(1-b1)*g
      vt_tmp = b2*vt+(1-b2)*tf.reduce_sum(tf.square(g))
      lr_tmp = lr/(tf.sqrt(vt_tmp) + 1e-8)

      if b_fix: lr_tmp = lr_tmp * tf.sqrt(1-tf.pow(b2,t))/(1-tf.pow(b1,t))

      opt.append(x.assign_add(-lr_tmp * mt_tmp))
      opt.append(vt.assign(vt_tmp))
      opt.append(mt.assign(mt_tmp))
        
  if b_fix: opt.append(t.assign_add(1.0))
  return(tf.group(opt))

def GREMLIN(msa, opt_type="adam", opt_iter=100, opt_rate=1.0, batch_size=None,lamb_w = 1):
  ##############################################################
  # SETUP COMPUTE GRAPH
  ##############################################################
  # kill any existing tensorflow graph
  #tf.reset_default_graph()
  ncol = msa['ncol'] # length of sequence
  # msa (multiple sequence alignment) 
  MSA = tf.placeholder(tf.int32,shape=(None,ncol),name="msa")
  # one-hot encode msa
  OH_MSA = tf.one_hot(MSA,states)
  # msa weights
  MSA_weights = tf.placeholder(tf.float32, shape=(None,), name="msa_weights")
  # 1-body-term of the MRF
  V = tf.get_variable(name="V", 
                      shape=[ncol,states],
                      initializer=tf.zeros_initializer)
  # 2-body-term of the MRF
  W = tf.get_variable(name="W",
                      shape=[ncol,states,ncol,states],
                      initializer=tf.zeros_initializer)
  # symmetrize W
  W = sym_w(W)
  def L2(x): return tf.reduce_sum(tf.square(x))
  ########################################
  # V + W
  ########################################
  VW = V + lamb_w*tf.tensordot(OH_MSA,W,2)
  # hamiltonian
  H = tf.reduce_sum(tf.multiply(OH_MSA,VW),axis=(1,2))
  # local Z (parition function)
  Z = tf.reduce_sum(tf.reduce_logsumexp(VW,axis=2),axis=1)
  # Psuedo-Log-Likelihood
  PLL = H - Z
  # Regularization
  L2_V = 0.01 * L2(V)
  L2_W = 0.01 * L2(W) * 0.5 * (ncol-1) * (states-1)
  # loss function to minimize
  loss = -tf.reduce_sum(PLL*MSA_weights)/tf.reduce_sum(MSA_weights)
  loss = loss + (L2_V + L2_W)/msa["neff"]
  ##############################################################
  # MINIMIZE LOSS FUNCTION
  ##############################################################
  if opt_type == "adam":  
    opt = opt_adam(loss,"adam",lr=opt_rate)
  # generate input/feed
  def feed(feed_all=False):
    if batch_size is None or feed_all:
      return {MSA:msa["msa"], MSA_weights:msa["weights"]}
    else:
      idx = np.random.randint(0,msa["nrow"],size=batch_size)
      return {MSA:msa["msa"][idx], MSA_weights:msa["weights"][idx]}
  # optimize!
  with tf.Session() as sess:
    # initialize variables V and W
    sess.run(tf.global_variables_initializer())
    # initialize V
    msa_cat = tf.keras.utils.to_categorical(msa["msa"],states)
    pseudo_count = 0.01 * np.log(msa["neff"])
    V_ini = np.log(np.sum(msa_cat.T * msa["weights"],-1).T + pseudo_count)
    V_ini = V_ini - np.mean(V_ini,-1,keepdims=True)
    sess.run(V.assign(V_ini))
    # compute loss across all data
    get_loss = lambda: round(sess.run(loss,feed(feed_all=True)) * msa["neff"],2)
    print("starting",get_loss())
    if opt_type == "lbfgs":
      lbfgs = tf.contrib.opt.ScipyOptimizerInterface
      opt = lbfgs(loss,method="L-BFGS-B",options={'maxiter': opt_iter})
      opt.minimize(sess,feed(feed_all=True))
    if opt_type == "adam":
      for i in range(opt_iter):
        sess.run(opt,feed())  
        #if (i+1) % int(opt_iter/100) == 0:
          #print("iter",(i+1),get_loss())
    # save the V and W parameters of the MRF
    V_ = sess.run(V)
    W_ = sess.run(W)
  # only return upper-right triangle of matrix (since it's symmetric)
  #tri = np.triu_indices(ncol,1)
  #W_ = W_[tri[0],:,tri[1],:]
  mrf = {"v": V_,
         "w": W_,
         "v_idx": msa["v_idx"],
         "w_idx": msa["w_idx"]}
  return mrf

def GREMLIN_llik(msa, mrf):
    N, L = msa.shape
    OH_MSA = one_hot(msa)
    # 1-body-term of the MRF
    V = mrf["v"]
    # 2-body-term of the MRF
    W = mrf["w"]   
    ########################################
    # V + W
    ########################################
    VW = V + np.tensordot(OH_MSA,W,2)
    # hamiltonian
    H = np.sum(np.multiply(OH_MSA,VW),axis=(1,2))
    # local Z (parition function)
    Z = np.sum(np.logaddexp.reduce(VW,axis=2),axis=1)
    # Psuedo-Log-Likelihood
    PLL = H - Z
    return(PLL)

def fill_gaps(msa):
    '''fill gap with random imputation'''
    tmp = (msa == states-1).astype("int")
    seq_prob_all = seq_prob(msa)
    seq_prob_nongap = [seq_prob_all[i,0:states-1]/sum(seq_prob_all[i,0:states-1]) for i in range(seq_prob_all.shape[0])]
    random_impute = [np.random.choice(np.arange(states-1), size=msa.shape[0], p=seq_prob_nongap_i) for seq_prob_nongap_i in seq_prob_nongap]
    random_impute = (np.array(random_impute).T)
    msa_fill = msa*(1-tmp)+random_impute*tmp
    return msa_fill


def filt_gaps(msa,gap_cutoff=0.5):
  '''filters alignment to remove gappy positions'''
  tmp = (msa == states-1).astype("float")
  non_gaps = np.where(np.sum(tmp.T,-1).T/msa.shape[0] < gap_cutoff)[0]
  return msa[:,non_gaps],non_gaps

def get_eff(msa,eff_cutoff=0.8):
  '''compute effective weight for each sequence'''
  ncol = msa.shape[1]
  # pairwise identity
  msa_sm = 1.0 - squareform(pdist(msa,"hamming"))
  # weight for each sequence
  msa_w = (msa_sm >= eff_cutoff).astype("float")
  msa_w = 1/np.sum(msa_w,-1)
  return msa_w

def mk_msa_df(msa_ori):
  # '''converts list of sequences to msa'''
  # msa_ori = []
  # for seq in seqs:
  #   msa_ori.append([aa2num(aa) for aa in seq])
  # msa_ori = np.array(msa_ori)
  # # remove positions with more than > 50% gaps
  msa, v_idx = filt_gaps(msa_ori,1)#fill_gaps(msa_ori)#filt_gaps(msa_ori,0.99)
  # compute effective weight for each sequence
  msa_weights = get_eff(msa,0.8)
  # compute effective number of sequences
  ncol = msa.shape[1] # length of sequence
  w_idx = v_idx[np.stack(np.triu_indices(ncol,1),-1)]
  return {"msa":msa,
          "weights":msa_weights,
          "neff":np.sum(msa_weights),
          "v_idx":v_idx,
          "w_idx":w_idx,
          "nrow":msa.shape[0],
          "ncol":ncol,
          "ncol_ori":msa_ori.shape[1]}

###################
def normalize(x):
  x = stats.boxcox(x - np.amin(x) + 1.0)[0]
  x_mean = np.mean(x)
  x_std = np.std(x)
  return((x-x_mean)/x_std)

def get_mtx(mrf):
  '''get mtx given mrf'''
  # l2norm of 20x20 matrices (note: we ignore gaps)
  # aa_idx = [i for i in range(21) if i != 20]
  W_ = mrf["w"]
  tri = np.triu_indices(W_.shape[0],1)
  W_ = W_[tri[0],:,tri[1],:]
  raw = np.sqrt(np.sum(np.square(W_[:,:-1,:-1]),(1,2)))
  raw_sq = squareform(raw)
  # apc (average product correction)
  ap_sq = np.sum(raw_sq,0,keepdims=True)*np.sum(raw_sq,1,keepdims=True)/np.sum(raw_sq)
  apc = squareform(raw_sq - ap_sq, checks=False)
  mtx = {"i": mrf["w_idx"][:,0],
         "j": mrf["w_idx"][:,1],
         "raw": raw,
         "apc": apc,
         "zscore": normalize(apc)}
  return mtx

def plot_mtx(mtx,key="zscore",vmin=1,vmax=3):
  '''plot the mtx'''
  plt.figure(figsize=(5,5))
  plt.imshow(squareform(mtx[key]), cmap='Blues', interpolation='none', vmin=vmin, vmax=vmax)
  plt.grid(False)
  plt.show()

def get_mtx_w(W):
  '''
  ------------------------------------------------------
  inputs
  ------------------------------------------------------
   w           : coevolution   shape=(L,A,L,A)
  ------------------------------------------------------
  outputs 
  ------------------------------------------------------
   raw         : l2norm(w)     shape=(L,L)
   apc         : apc(raw)      shape=(L,L)
  ------------------------------------------------------
  '''
  # l2norm of 20x20 matrices (note: we ignore gaps)
  raw = np.sqrt(np.sum(np.square(W[:,:-1,:,:-1]),(1,3)))
  # apc (average product correction)
  ap = np.sum(raw,0,keepdims=True)*np.sum(raw,1,keepdims=True)/np.sum(raw)
  apc = raw - ap
  np.fill_diagonal(apc,0)
  return raw, apc


def seq_sample_all_bayes_weight(msa,prob_init,prob_weight, ntries = 10,lamb = 2,tau = 10,max_iter = 50):
    N = len(msa)
    if N<100:
        min_sample_size = 10
        max_sample_size = 50
    elif N<500:
        min_sample_size = 10
        max_sample_size = 100
    else:
        min_sample_size = 10
        max_sample_size = 200
    msa_samples = []
    n_iter = 0
    n_samples = 0
    c = 1 #control the overall sampling probs' scale
    while n_samples < ntries and n_iter < max_iter :
        n_iter+=1
        msa_samp = seq_sample_bayes_weight(msa,prob_init,prob_weight, seed=123*n_iter,lamb = lamb,tau = tau,c=c,max_sample_size = max_sample_size) # get a new sampled MSA
        if len(msa_samp)>= min_sample_size:
            n_samples+=1
            msa_samples.append(msa_samp)
        if n_iter%10 == 0:
            c *= 1.5
    return msa_samples

def seq_sample_bayes_weight(msa,prob_init,prob_weight,seed=123,lamb = 2,tau = 10,c=1,max_sample_size =512):
    # c controls the overall sampling probs
    shuf_msa = msa[1:].copy().tolist()
    shuf_idx = np.arange(len(msa)-1)
    random.shuffle(shuf_idx) 
    shuf_msa = [shuf_msa[idx] for idx in shuf_idx]# randomly permute MSA
    samp_msa = []
    cur_seq_prob = prob_init
    cur_entropy_vec,delta_p_exp,delta_p_sd_inv = seq_entropy_exp_sd(cur_seq_prob)
    M = abs(np.sum(abs(max_to_onehot(cur_seq_prob)-cur_seq_prob), 1) )
    for seq in shuf_msa:
        log_prob = 0
        prop_msa = samp_msa.copy()
        prop_msa.append(np.array(seq))
        delta_prob =  seq2mat(seq) -cur_seq_prob 
        delta_prob_abs = abs(np.sum(abs(delta_prob), 1) )
        delta_prob_rescaled = delta_prob_abs*delta_p_sd_inv
        log_prob -= lamb*np.mean(delta_prob_rescaled*prob_weight)/np.mean(prob_weight)
        prob = c*np.exp(log_prob)/np.exp(lamb*np.mean(-M*delta_p_sd_inv))
        if np.random.uniform() < prob:
            samp_msa.append(np.array(seq))
            cur_seq_prob = cur_seq_prob*(tau+len(samp_msa)-1)/(tau+len(samp_msa))+seq2mat(seq)/(tau+len(samp_msa))
            cur_entropy_vec,delta_p_exp,delta_p_sd_inv = seq_entropy_exp_sd(cur_seq_prob)
            M = abs(np.sum(abs(max_to_onehot(cur_seq_prob)-cur_seq_prob), 1) )
        if len(samp_msa)> max_sample_size:
            break
    return samp_msa

def plot_ss_MSA(msa_embedded,selected_indices,file_name):
    plt.figure(figsize=(8, 6),dpi = 300)
    plt.scatter(
        msa_embedded[:, 0],  # x-axis
        msa_embedded[:, 1],  
        c = "gray",
        alpha=0.7
    )
    plt.scatter(
        msa_embedded[selected_indices, 0],  # x-axis
        msa_embedded[selected_indices, 1],  # y-axis
        c = "black",
        alpha=0.7
    )
    plt.scatter(
        msa_embedded[0, 0],  # x-axis
        msa_embedded[0, 1],  
        c = "red",
        marker = '*',
        s = 30
    )
    plt.title("UMAP Visualization of Sampled MSA, seqs num (%d)"%len(selected_indices))
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.savefig(file_name)        
    plt.cla()

def run_BSS(msa, a3m_path, jobname, save_dir,lamb_list = [0,1,2,3]):
    random.seed(123)
    np.random.seed(123)
    prob_weight = 1-np.mean(msa==21,0)
    IDs,seqs = load_fasta(a3m_path)
    N, L = msa.shape
    ohe_msa = one_hot(msa).reshape(N,-1)
    if N<100:
        n_neighbors_list = [10]
    else:
        n_neighbors_list = [10,30]
    msa_dist = squareform(pdist(msa,"hamming"))
    umap_reducer = UMAP(n_components=2, random_state=42,n_neighbors=30,metric = "precomputed")  # Use UMAP with 2 components
    msa_embedded = umap_reducer.fit_transform(msa_dist)
    np.save(save_dir+"/msa_embedded.npy", msa_embedded)
    for n_neighbors in n_neighbors_list:  
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(ohe_msa)
        distances, indices = nn.kneighbors(ohe_msa)
        averaged_probs = np.zeros_like(ohe_msa, dtype=np.float32)
        for i in range(len(indices)):
            # Get indices of self + neighbors (indices[i] already includes self as first element)
            neighbor_indices = indices[i]
            neighbor_features = ohe_msa[neighbor_indices]
            averaged_probs[i] = neighbor_features.mean(axis=0)
        pca = PCA(n_components=0.9, random_state=42)  # Adjust n_components as needed
        pca_features = pca.fit_transform(averaged_probs)
        ### Get clustering on the prob vectors
        n_coreset = 9
        save_fig_cluster_dir = save_dir+"/res_fig/sub_MSA_cluster_probs_bayes_neighbors%d/"%n_neighbors
        save_msa_cluster_dir = save_dir+"/msa_cluster_probs_bayes_neighbors%d/"%n_neighbors
        os.makedirs(save_fig_cluster_dir, exist_ok=True)
        os.makedirs(save_msa_cluster_dir, exist_ok=True)
        kmedoids = KMedoids(n_clusters=n_coreset, metric='euclidean')
        kmedoids.fit(pca_features)
        # Get medoid indices (real data points serving as centers)
        medoid_indices = kmedoids.medoid_indices_
        # Get cluster centers in PCA space
        cluster_labels = kmedoids.labels_
        msa_clustered_indices = [[node for node, cluster in enumerate(cluster_labels) if cluster == c] for c in range(n_coreset)]
        clustered_msas = [np.array(msa)[indice,:] for indice in msa_clustered_indices]
        with open(save_msa_cluster_dir+"clustered_msas.pkl", "wb") as f:
            pickle.dump(clustered_msas, f)
        for k in range(len(clustered_msas)):
            msa_ss_seqs = [seqs[0]]
            msa_ss_seqs.extend([seqs[idx] for idx in msa_clustered_indices[k]])
            IDs_ss = [IDs[0]]
            IDs_ss.extend([IDs[idx] for idx in msa_clustered_indices[k]])
            write_fasta(IDs_ss, msa_ss_seqs, outfile=save_msa_cluster_dir+'cluster_%02d'% k +'.a3m') 
            plot_ss_MSA(msa_embedded,msa_clustered_indices[k],save_fig_cluster_dir +"msa_cluster_%d_umap.jpg"%k)
        for lamb in lamb_list:
            save_fig_ss_dir = save_dir+"/res_fig/sub_MSA_bayes_lamb%d_neighbors%d/"%(lamb,n_neighbors)
            save_msa_ss_dir = save_dir+"/msa_ss_bayes_lamb%d_neighbors%d/"%(lamb,n_neighbors)
            os.makedirs(save_fig_ss_dir, exist_ok=True)
            os.makedirs(save_msa_ss_dir, exist_ok=True)
            ### extract coresets for the prior's the concentration parameters
            msa_samples = []
            n_tries = 3
            averaged_probs_mat = averaged_probs.reshape(N,L,22)
            prob_init_list = [averaged_probs_mat[0]]
            prob_init_list.extend([averaged_probs_mat[medoid_indices[c_ind]] for c_ind in range(n_coreset)])# prior's concentration parameters
            neighbor_indices_list = [indices[0]]
            neighbor_indices_list.extend([indices[medoid_indices[c_ind]] for c_ind in range(n_coreset)]) # msa_sub used for calculate the prior's concentration parameters
            msa_samples_clabel = [] # save the indice of the prior's concentration parameters
            for c_ind in range(1+n_coreset):
                neighbor_indices = neighbor_indices_list[c_ind]
                msa_sub_initial = [msa[ind] for ind in neighbor_indices] 
                prob_init = prob_init_list[c_ind]
                msa_samples.append(msa_sub_initial) # include msa_subinitial as an extreme case
                msa_samples_c_ind = seq_sample_all_bayes_weight(msa,prob_init,prob_weight,n_tries,lamb = lamb,tau = n_neighbors/2)
                msa_samples.extend(msa_samples_c_ind)
                msa_samples_clabel.extend([c_ind]*(1+len(msa_samples_c_ind)) )
            if lamb ==0:
                msa_samples.append([seq for seq in msa])
                msa_samples_clabel.append(1+n_coreset)
            msa_samples_indices = [  [ msa.tolist().index(seq.tolist())  for seq in sample ]    for sample in msa_samples    ]
            for ss_idx in range(len(msa_samples)):
                msa_ss_seqs = [seqs[0]]
                msa_ss_seqs.extend([seqs[idx] for idx in msa_samples_indices[ss_idx]])
                IDs_ss = [IDs[0]]
                IDs_ss.extend([IDs[idx] for idx in msa_samples_indices[ss_idx]])
                write_fasta(IDs_ss, msa_ss_seqs, outfile=save_msa_ss_dir+'ss_%02d'% ss_idx +'.a3m') 

            with open(save_msa_ss_dir+"/msa_sample.pkl", "wb") as f:
                pickle.dump(msa_samples, f)
            with open(save_msa_ss_dir+"/msa_samples_clabel.pkl", "wb") as f:
                pickle.dump(msa_samples_clabel, f)
            save_plot_cluster_msas(msa_samples,save_fig_ss_dir+"msa_sample.jpg", sort_by_dist=False)
            for c_ind in np.arange(len(msa_samples)):
                msa_samp = msa_samples[c_ind]
                msa_samp = np.concatenate((msa[0].reshape((1,-1)),msa_samp))
                plot_ss_MSA(msa_embedded,msa_samples_indices[c_ind],save_fig_ss_dir +"msa_sample_%dth_prior_%d.jpg"%(c_ind,msa_samples_clabel[c_ind]))


def max_to_onehot(prob):
    one_hot = np.zeros_like(prob)
    one_hot[np.arange(len(prob)), np.argmax(prob, axis=1)] = 1
    return one_hot

def run_uniform(msa, a3m_path, jobname, save_dir, samp_size_list):
    # save_dir:  path for save the msa_samples and all other results
    save_msa_ss_dir = save_dir+"/msa_ss"
    os.makedirs(save_msa_ss_dir, exist_ok=True)
    random.seed(123)
    np.random.seed(123)
    prob_weight = 1-np.mean(msa==21,0)
    IDs,seqs = load_fasta(a3m_path)
    N, L = msa.shape
    n_tries = 300
    for samp_size in samp_size_list:  
        msa_samples_indices = [np.random.choice(np.arange(N),samp_size) for ii in range(n_tries)]
        msa_samples = [np.array(msa)[indice,:] for indice in msa_samples_indices]
        for k in range(len(msa_samples)):
            msa_ss_seqs = [seqs[0]]
            msa_ss_seqs.extend([seqs[idx] for idx in msa_samples_indices[k]])
            IDs_ss = [IDs[0]]
            IDs_ss.extend([IDs[idx] for idx in msa_samples_indices[k]])
            write_fasta(IDs_ss, msa_ss_seqs, outfile=save_msa_ss_dir+'/uniform_%02d_size_%03d'% (k ,samp_size)+'.a3m') 
    with open(os.path.join(save_dir, "success.txt"), "w") as f:
        f.write("All finished")
