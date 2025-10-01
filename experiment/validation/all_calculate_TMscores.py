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

with open('../config/config_SMICE_benchmark.json', 'r') as f:
    config = json.load(f)

metadata_92 = pd.read_csv(config["meta_path"])
base_output_dir = config["base_output_dir"]
base_result_dir = config["base_result_dir"]
jobnames = config["jobnames"]
true_pdb_path = config["true_pdb_path"]
base_TMscores_output_dir = config['base_TMscores_output_dir']

calculate_TMscore = False #if TMscores are not computed,change this to True

jobnames = ["3jv6A"]


def process_jobname(jobname, save_fig_dir,seperate_color=False):
    """
    Process SMICE job results by analyzing predicted structures
    and comparing them to reference structures using TM-scores.
    Args:
        jobname (str): Name of the job to process
        save_fig_dir (str): Directory to save output figures
        seperate_color (bool): Whether to generate scatter plots colored by source of sampling (Sequential sampling and enhanced sampling)
    """
    try:
        meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
        ID1 = meta_info['Fold1']
        ID2 = meta_info['Fold2']
        Seq1 = meta_info['Seq1']
        Seq2 = meta_info['Seq2']
        ID1_dir = true_pdb_path+ID1[0:4]+"_"+ID1[4]+".pdb"
        ID2_dir = true_pdb_path+ID2[0:4]+"_"+ID2[4]+".pdb"
        fsr_seq = meta_info["Sequence of fold-switching region"]
        sequence = meta_info['sequences']
        fsr_seq_extend = extend_fsr_seq(fsr_seq,sequence,len(fsr_seq))
        TMscore12 = compute_fsr_tmscore(ID1_dir, ID2_dir,fsr_seq_extend )
        ## load pooled preds json file
        if calculate_TMscore == True:
            outputs_SMICE = pd.read_json(base_output_dir+jobname+f"/outputs_SMICE.json.zip")
            print(list(outputs_SMICE['pdb_path']))
            outputs_SMICE["TMscore1"] = [compute_fsr_tmscore(ID1_dir, pdb_file,fsr_seq_extend ) for pdb_file in list(outputs_SMICE['pdb_path'])]
            outputs_SMICE["TMscore2"] = [compute_fsr_tmscore(ID2_dir, pdb_file,fsr_seq_extend ) for pdb_file in list(outputs_SMICE['pdb_path'])]
            outputs_SMICE.to_json(base_TMscores_output_dir+jobname+f"/outputs_SMICE_TMscores.json.zip")
        else:
            TMscore_dir = base_TMscores_output_dir+jobname+f"/outputs_SMICE_TMscores.json.zip"
            if os.path.exists(TMscore_dir):
                outputs_SMICE = pd.read_json(TMscore_dir)
            else:
                print(f"TMscore file not found for {jobname}, change the calculate_TMscore to True")
        fig = go.Figure()
        fig.update_layout(
        template="simple_white",  # Set template to simple_white
        width=550,  # Set width of the figure
        height=500,  # Set height of the figure
        showlegend=True)
        fig.add_trace(go.Scatter(
        x=np.arange(20)/20, 
        y=np.arange(20)/20,
        mode='lines',
        line=dict(color='black', dash='dash'),  # Customize line color and style
        showlegend=False  # Name for the line trace
        ))
        sources = outputs_SMICE["source"].unique()
        # Define colors for each source
        if seperate_color == True:
            colors = ['#377EB8', '#E41A1C']  # Add more colors if you have more categories
        else:
            colors = ['red']
            sources = ['SMICE']
            outputs_SMICE["source"] = "SMICE"
        for i, source in enumerate(sources):
            mask = outputs_SMICE["source"] == source
            fig.add_trace(go.Scatter(
                x=outputs_SMICE["TMscore1"][mask], 
                y=outputs_SMICE["TMscore2"][mask],
                mode='markers',
                marker=dict(
                    color=colors[i],
                    size=5,
                    opacity=0.5
                ),
                name=source  # Name includes the source category
            ))
        fig.add_trace(go.Scatter(
            x=np.ones(100)*TMscore12, 
            y=np.arange(100)/100*TMscore12,
            mode='lines',
            line=dict(color='black'),  # Customize line color and style
            showlegend=False  # Name for the line trace
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(100)/100*TMscore12, 
            y=np.ones(100)*TMscore12,
            mode='lines',
            line=dict(color='black'),  # Customize line color and style
            showlegend=False  # Name for the line trace
        ))
        fig.update_xaxes(range=[0, 1],title_text='TMscore to %s'%ID1)  # Set x-axis limits from 0 to 6
        fig.update_yaxes(range=[0, 1],title_text='TMscore to %s'%ID2)  # Set y-axis limits from 8 to 16
        fig.update_xaxes()  # Set x-axis label
        os.makedirs(base_result_dir+"TMscore_fig/", exist_ok=True)
        fig.write_image(base_result_dir+"TMscore_fig/%s.png"%jobname, scale=5)
        plt.cla()
    except Exception as e:
        error_msg = f"Error processing {jobname}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        #raise  # Re-raise the exception after logging
        
def main():
    save_fig_dir = base_result_dir+"TMscore_fig/"
    os.makedirs(save_fig_dir, exist_ok=True)
    try:
        num_processes = multiprocessing.cpu_count() - 1  # Leave one core free
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(partial(process_jobname, save_fig_dir=save_fig_dir), jobnames)
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
    finally:
        # Clean up
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
    
