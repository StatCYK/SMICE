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
    'font.size': 18,              # Base font size
    'axes.titlesize': 18,        # Axis titles
    'axes.labelsize': 18,        # Axis labels
    'xtick.labelsize': 18,       # X-axis tick labels
    'ytick.labelsize': 18,       # Y-axis tick labels
    'legend.fontsize': 18,       # Legend
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

def process_jobname_and_plot(jobname):
    # Load SMICE data
    outputs_bss = pd.read_json(f"{base_output_dir}{jobname}/bss_res/outputs_bss_alpha_{alpha_choice}.json.zip")
    outputs_enhanced = pd.read_json(f"{base_output_dir}{jobname}/outputs_enhanced_alpha_{alpha_choice}.json.zip")
    outputs_SMICE = pd.concat([outputs_bss, outputs_enhanced], ignore_index=True) 
    contacts_SMICE = np.load(f"{base_output_dir}{jobname}/contacts.npy")
    lamb_list = [int(pdb_file[pdb_file.index("lamb")+4:pdb_file.index("lamb")+5]) for pdb_file in outputs_bss["pdb_path"]]
    # Combine all contacts for PCA fitting
    n_SMICE_SS = len(outputs_bss)
    contacts_SS = contacts_SMICE[0:n_SMICE_SS,:] 
    contacts_SS = contacts_SS.reshape((contacts_SS.shape[0],-1))
    # Fit PCA on all contacts
    mdl = PCA(n_components=2, random_state=42)
    embedding_SMICE_SS = mdl.fit_transform(contacts_SS)
    # Create the figure with larger size
    fig = go.Figure()
    fig.update_layout(
    template="simple_white",  # Set template to simple_white
    width=650,  # Set width of the figure
    height=500,  # Set height of the figure
    showlegend=True)
    fig.add_trace(go.Scatter(
            x=embedding_SMICE_SS[:,0], y=embedding_SMICE_SS[:,1],
            mode='markers',
            marker=dict(color=lamb_list, size=3,opacity=0.8, colorscale='Inferno', showscale=True,
                       colorbar=dict(
                            title='λ values',   # Colorbar title
                            len=0.5,         # Length of colorbar (0-1, relative to plot height)
                            thickness=10,    # Thickness in pixels
                            x=1.1,           # Adjust position (x-axis anchor)
                        )) ,#,
            name="Sequential sampling(%d predicts)"%len(embedding_SMICE_SS)
    ))
    fig.update_xaxes(title_text='PC1 (on predicted contact maps)') 
    fig.update_yaxes(title_text='PC2 (on predicted contact maps)')  
    plt.cla()
    # Save the figure
    os.makedirs(f"{base_result_dir}analysis/lamb_effect/", exist_ok=True)
    output_plot_path = f"{base_result_dir}analysis/lamb_effect/pca_plot_{jobname}.png"
    fig.write_image(output_plot_path, scale=5)

def main():
    jobnames = ["1ceeB"]
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
