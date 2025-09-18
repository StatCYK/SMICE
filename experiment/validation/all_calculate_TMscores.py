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

ALPHA_VALUES = np.arange(0,2.1,0.1)
alpha_choice = 10#"adaptive"

metadata_92 = pd.read_csv(config["meta_path"])
base_output_dir = config["base_output_dir"]
base_result_dir = config["base_result_dir"]
jobnames = config["jobnames"]#config["FS"]#
true_pdb_path = config["true_pdb_path"]

def process_BSS_jobname(jobname, save_fig_dir):
    try:
        meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
        ID1 = meta_info['Fold1']
        ID2 = meta_info['Fold2']
        Seq1 = meta_info['Seq1']
        Seq2 = meta_info['Seq2']
        fsr_seq = meta_info["Sequence of fold-switching region"]#sequence#
        TMscores_fsr = np.load(f"/n/home13/yongkai/Yongkai/Alphafold2/SS_AF2/metadata/{jobname}/TMscores_fsr.npy")# the tmscores(fsr) between two folds
        if alpha_choice == "adaptive":
            alpha_indice = np.argmin(TMscores_fsr)
        else:
            alpha_indice = alpha_choice
        TMscore12 = TMscores_fsr[alpha_indice]
        sequence = meta_info['sequences']
        TMscore1 = []
        TMscore2 = []
        outputs=[]
        for ID in [ID1]:
            fsr_seq_extend = extend_fsr_seq(fsr_seq,sequence,int(len(fsr_seq)*ALPHA_VALUES[alpha_indice]))
            lamb_list = []
            for lamb in [0,1,2,3]:
                for n_neighbors in [10, 30]:
                    pdb_path = base_output_dir+ID+"/bss_res/pdb_ss_bayes_colab_lamb%d_neighbors%d/"%(lamb,n_neighbors)
                    msa_path = base_output_dir+ID+"/bss_res/msa_ss_bayes_lamb%d_neighbors%d/"%(lamb,n_neighbors)
                    ID1_dir = true_pdb_path+ID1[0:4]+"_"+ID1[4]+".pdb"
                    ID2_dir = true_pdb_path+ID2[0:4]+"_"+ID2[4]+".pdb"
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
                                tm_score1 = compute_fsr_tmscore(ID1_dir, pdb_files[0],fsr_seq_extend )
                                tm_score2 = compute_fsr_tmscore(ID2_dir, pdb_files[0],fsr_seq_extend )
                                score_file = glob.glob(os.path.join(pdb_path, score_file_pattern))[0]
                                pdb_file = pdb_files[0]
                                with open(score_file,"r") as f:
                                    plddt_scores = pd.read_json(f)
                                    avg_pae = np.mean(np.mean(np.array(plddt_scores["pae"])))
                                    max_pae = plddt_scores["max_pae"].iloc[0]
                                    ptm = plddt_scores["ptm"].iloc[0]
                                    avg_plddt = np.mean(plddt_scores["plddt"])/100
                                TMscore1.append(tm_score1)
                                TMscore2.append(tm_score2)
                                o.update({'msa_path': f"{msa_path}ss_{ss:02d}.a3m"})
                                o.update({'pdb_path': pdb_file})
                                o.update({'score_path': score_file})
                                o.update({'model': model})
                                o.update({'avg_plddt': avg_plddt})
                                o.update({'avg_pae': avg_pae })
                                o.update({'max_pae': max_pae})
                                o.update({'ptm': ptm})
                                o.update({'TMscore1': tm_score1})
                                o.update({'TMscore2': tm_score2})
                                o.update({'MSA_ID': ID})
                                o.update
                                outputs.append(o)  
            fig = go.Figure()
            fig.update_layout(
            template="simple_white",  # Set template to simple_white
            width=650,  # Set width of the figure
            height=500,  # Set height of the figure
            showlegend=True)
            fig.add_trace(go.Scatter(
                    x=TMscore1, y=TMscore2,
                    mode='markers',
                    marker=dict(color=lamb_list, size=3,opacity=0.8, colorscale='Inferno', showscale=True,
                               colorbar=dict(
                                    title='λ values',   # Colorbar title
                                    len=0.5,         # Length of colorbar (0-1, relative to plot height)
                                    thickness=10,    # Thickness in pixels
                                    x=1.1,           # Adjust position (x-axis anchor)
                                )) ,#,
                    name="Sequential sampling(%d predicts)"%len(TMscore1)
            ))
            fig.add_trace(go.Scatter(
                x=np.arange(20)/20, 
                y=np.arange(20)/20,
                mode='lines',
                line=dict(color='black', dash='dash'),  # Customize line color and style
                showlegend=False  # Name for the line trace
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
            fig.write_image(save_fig_dir+"%s_plddt(batch)(fsr).png"%ID, scale=5)
            plt.cla()
        outputs = pd.DataFrame.from_records(outputs)
        outputs.to_json(base_output_dir+jobname+f"/bss_res/outputs_bss_alpha_{alpha_choice}.json.zip")
    except Exception as e:
        error_msg = f"Error processing {jobname}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
def process_enhanced_jobname(jobname, save_fig_dir):
    try:
        n_coreset = 10
        os.makedirs(save_fig_dir, exist_ok=True)
        if jobname in list(metadata_92['jobnames']):
            meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
            ID1 = meta_info['Fold1']
            ID2 = meta_info['Fold2']
            Seq1 = meta_info['Seq1']
            Seq2 = meta_info['Seq2']
            fsr_seq = meta_info["Sequence of fold-switching region"]
            sequence = meta_info['sequences']
            TMscores_fsr = np.load(f"/n/home13/yongkai/Yongkai/Alphafold2/SS_AF2/metadata/{jobname}/TMscores_fsr.npy")
            if alpha_choice == "adaptive":
                alpha_indice = np.argmin(TMscores_fsr)
            else:
                alpha_indice = alpha_choice
            TMscore12 = TMscores_fsr[alpha_indice]
            TMscore1 = []
            TMscore2 = []
            outputs=[]
            for ID in [jobname]:
                fsr_seq_extend = extend_fsr_seq(fsr_seq,sequence,int(len(fsr_seq)*ALPHA_VALUES[alpha_indice]))
                ID1_dir = true_pdb_path+ID1[0:4]+"_"+ID1[4]+".pdb"
                ID2_dir = true_pdb_path+ID2[0:4]+"_"+ID2[4]+".pdb"
                for iter in [1,2]:
                    save_dir = base_output_dir+f"/{ID}/enhanced_iter{iter}_res"
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
                                        tm_score1 = compute_fsr_tmscore(ID1_dir, pdb_files[0],fsr_seq_extend )
                                        tm_score2 = compute_fsr_tmscore(ID2_dir, pdb_files[0],fsr_seq_extend )
                                        score_file = glob.glob(os.path.join(pdb_path, score_file_pattern))[0]
                                        pdb_file = pdb_files[0]
                                        TMscore1.append(tm_score1)
                                        TMscore2.append(tm_score2)
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
                                        o.update({'TMscore1': tm_score1})
                                        o.update({'TMscore2': tm_score2})
                                        o.update({'MSA_ID': ID})
                                        o.update
                                        outputs.append(o)
                fig = go.Figure()
                fig.update_layout(
                template="simple_white",  # Set template to simple_white
                width=550,  # Set width of the figure
                height=500,  # Set height of the figure
                showlegend=True)
                fig.add_trace(go.Scatter(
                        x=TMscore1, y=TMscore2,
                        mode='markers',
                        marker=dict(color="#FF0000", size=4,opacity=0.5) ,#,
                        name="Enhanced sampling(%d predicts)"%len(TMscore1)
                ))
                fig.update_xaxes(range=[0, 1],title_text='TMscore to %s'%ID1)  # Set x-axis limits from 0 to 6
                fig.update_yaxes(range=[0, 1],title_text='TMscore to %s'%ID2)  # Set y-axis limits from 8 to 16
                fig.update_xaxes()  # Set x-axis label
                fig.write_image(save_fig_dir+"%s_plddt(batch)(fsr_extend)_coevol_2way_iter%d.png"%(ID,iter), scale=5)
                plt.cla()
            TMscores = np.array([TMscore1,TMscore2])
            # Combine dataframes
            outputs_bss = pd.read_json(f"{base_output_dir}{jobname}/bss_res/outputs_bss_alpha_{alpha_choice}.json.zip")
            TMscores_bss = np.array([outputs_bss['TMscore1'],outputs_bss['TMscore2']])
            TMscores_combine = np.concatenate((TMscores_bss.T,TMscores.T))
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
            fig.add_trace(go.Scatter(
                    x=TMscores_bss[0,:], y=TMscores_bss[1,:],
                    mode='markers',
                    marker=dict(color="#377EB8", size=5,opacity=0.5) ,#,
                    name="SMICE_SeqSamp"
            ))
            fig.add_trace(go.Scatter(
                    x=TMscores[0,:], y=TMscores[1,:],
                    mode='markers',
                    marker=dict(color="#E41A1C" , size=5,opacity=0.5) ,#,
                    name="SMICE_enhanced"
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
            os.makedirs(base_result_dir+"TMscore_fig_combine/", exist_ok=True)
            fig.write_image(base_result_dir+"TMscore_fig_combine/%s_plddt(batch)(fsr_extend)_combine_seperate_color.png"%jobname, scale=5)
            plt.cla()
            outputs = pd.DataFrame.from_records(outputs)
            outputs.to_json(base_output_dir+jobname+f"/outputs_enhanced_alpha_{alpha_choice}.json.zip")

    except Exception as e:
        error_msg = f"Error processing {jobname}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        #raise  # Re-raise the exception after logging

def process_enhanced_nocevol_jobname(jobname, save_fig_dir):
    try:
        n_coreset = 10
        os.makedirs(save_fig_dir, exist_ok=True)
        if jobname in list(metadata_92['jobnames']):
            meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
            sequence = meta_info['sequences']
            ID1 = meta_info['Fold1']
            ID2 = meta_info['Fold2']
            Seq1 = meta_info['Seq1']
            Seq2 = meta_info['Seq2']
            fsr_seq = meta_info["Sequence of fold-switching region"]
            TMscores_fsr = np.load(f"/n/home13/yongkai/Yongkai/Alphafold2/SS_AF2/metadata/{jobname}/TMscores_fsr.npy")
            if alpha_choice == "adaptive":
                alpha_indice = np.argmin(TMscores_fsr)
            else:
                alpha_indice = alpha_choice
            TMscore12 = TMscores_fsr[alpha_indice]
            fsr_seq_extend = extend_fsr_seq(fsr_seq,sequence,int(len(fsr_seq)*ALPHA_VALUES[alpha_indice]))
            ID1_dir = true_pdb_path+ID1[0:4]+"_"+ID1[4]+".pdb"
            ID2_dir = true_pdb_path+ID2[0:4]+"_"+ID2[4]+".pdb"
            TMscore1 = []
            TMscore2 = []
            outputs = []
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
                                pattern = f"ss_MRF_{ii}_2_MRF_{jj}_*{set_size}_marginal_relaxed*model_{model}*.pdb"
                                pdb_files = glob.glob(os.path.join(pdb_path, pattern))
                                score_file_pattern = f"ss_MRF_{ii}_2_MRF_{jj}_*{set_size}_marginal_*model_{model}*.json"
                                if len(pdb_files)>0:
                                    tm_score1 = compute_fsr_tmscore(ID1_dir, pdb_files[0],fsr_seq_extend )
                                    tm_score2 = compute_fsr_tmscore(ID2_dir, pdb_files[0],fsr_seq_extend )
                                    score_file = glob.glob(os.path.join(pdb_path, score_file_pattern))[0]
                                    pdb_file = pdb_files[0]
                                    TMscore1.append(tm_score1)
                                    TMscore2.append(tm_score2)
                                    with open(score_file,"r") as f:
                                        plddt_scores = pd.read_json(f)
                                    avg_plddt = np.mean(plddt_scores["plddt"])/100
                                    o.update({'msa_path': f"{save_dir}/msa_ss/model_{model}/ss_MRF_{ii}_2_MRF_{jj}_*{set_size}_marginal.a3m"})
                                    o.update({'pdb_path': pdb_file})
                                    o.update({'score_path': score_file})
                                    o.update({'model': model})
                                    o.update({'avg_plddt': avg_plddt})
                                    o.update({'avg_pae': np.mean(np.mean(np.array(plddt_scores["pae"])))})
                                    o.update({'max_pae': plddt_scores["max_pae"].iloc[0]})
                                    o.update({'ptm': plddt_scores["ptm"].iloc[0]})
                                    o.update({'TMscore1': tm_score1})
                                    o.update({'TMscore2': tm_score2})
                                    o.update
                                    outputs.append(o)
            fig = go.Figure()
            fig.update_layout(
            template="simple_white",  # Set template to simple_white
            width=650,  # Set width of the figure
            height=500,  # Set height of the figure
            showlegend=True)
            fig.add_trace(go.Scatter(
                    x=TMscore1, y=TMscore2,
                    mode='markers',
                    marker=dict(color="#FF0000", size=4,opacity=0.5) ,#,
                    name="Enhanced sampling(%d predicts)"%len(TMscore1)
            ))
            fig.update_xaxes(range=[0, 1],title_text='TMscore to %s'%ID1)  # Set x-axis limits from 0 to 6
            fig.update_yaxes(range=[0, 1],title_text='TMscore to %s'%ID2)  # Set y-axis limits from 8 to 16
            fig.update_xaxes()  # Set x-axis label
            fig.write_image(save_fig_dir+"%s_plddt(batch)(fsr_extend)_coevol_2way_marginal_iter%d.png"%(jobname,iter), scale=5)
            plt.cla()
            TMscores = np.array([TMscore1,TMscore2])
            outputs_bss = pd.read_json(f"{base_output_dir}{jobname}/bss_res/outputs_bss_alpha_{alpha_choice}.json.zip")
            TMscores_bss = np.array([outputs_bss['TMscore1'],outputs_bss['TMscore2']])
            TMscores_combine = np.concatenate((TMscores_bss.T,TMscores.T))
            fig = go.Figure()
            fig.update_layout(
            template="simple_white",  # Set template to simple_white
            width=650,  # Set width of the figure
            height=500,  # Set height of the figure
            showlegend=True)
            fig.add_trace(go.Scatter(
            x=np.arange(20)/20, 
            y=np.arange(20)/20,
            mode='lines',
            line=dict(color='black', dash='dash'),  # Customize line color and style
            showlegend=False  # Name for the line trace
            ))
            fig.add_trace(go.Scatter(
                    x=TMscores_bss[0,:], y=TMscores_bss[1,:],
                    mode='markers',
                    marker=dict(color="#377EB8", size=4,opacity=0.5) ,#,
                    name="SMICE_SeqSamp"
            ))
            fig.add_trace(go.Scatter(
                    x=TMscores[0,:], y=TMscores[1,:],
                    mode='markers',
                    marker=dict(color="#E41A1C" , size=4,opacity=0.5) ,#,
                    name="SMICE_enhanced"
            ))
            # fig.add_trace(go.Scatter(
            #         x=TMscores_combine[:,0], y=TMscores_combine[:,1],
            #         mode='markers',
            #         marker=dict(color="#FF0000", size=4,opacity=0.5) ,#,
            #         name="SMICE(%d predicts)"%len(TMscores_combine[:,0])
            # ))
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
            os.makedirs(base_result_dir+"TMscore_fig_combine_withoutcoevol/", exist_ok=True)
            fig.write_image(base_result_dir+"TMscore_fig_combine_withoutcoevol/%s_plddt(batch)(fsr_extend)_combine_seperate_color.png"%jobname, scale=5)
            plt.cla()
            outputs = pd.DataFrame.from_records(outputs)
            outputs.to_json(base_output_dir+jobname+f"/outputs_enhanced_withoutcoevol_alpha_{alpha_choice}.json.zip")

    except Exception as e:
        error_msg = f"Error processing {jobname}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
def plot_tmscore_scatter_with_confidence(jobname, save_dir):
    """
    Plot TMscore scatter plot colored by confidence metric
    Args:
        jobname: Name of the job
        save_dir: Directory to save the plot
    """
    try:
        meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
        sequence = meta_info['sequences']
        ID1 = meta_info['Fold1']
        ID2 = meta_info['Fold2']
        os.makedirs(save_dir, exist_ok=True)
        outputs_bss = pd.read_json(f"{base_output_dir}{jobname}/bss_res/outputs_bss_alpha_{alpha_choice}.json.zip")
        outputs_enhanced = pd.read_json(f"{base_output_dir}{jobname}/outputs_enhanced_alpha_{alpha_choice}.json.zip")
        combined = pd.concat([outputs_bss, outputs_enhanced], ignore_index=True) 
        valid_data = combined.dropna(subset=['TMscore1','TMscore2', 'avg_plddt', 'avg_pae', 'ptm'])
        TMscore1_filtered = valid_data['TMscore1']
        TMscore2_filtered = valid_data['TMscore2']
        TMscores_fsr = np.load(f"/n/home13/yongkai/Yongkai/Alphafold2/SS_AF2/metadata/{jobname}/TMscores_fsr.npy")
        if alpha_choice == "adaptive":
            alpha_indice = np.argmin(TMscores_fsr)
        else:
            alpha_indice = alpha_choice
        TMscore12 = TMscores_fsr[alpha_indice]
        for metric in ['avg_plddt', 'avg_pae', 'ptm']:#,"Rosetta_energy",'DFIRE_energy']:
            confidence_filtered = valid_data[metric]
            if len(TMscore1_filtered)==0:
                print(f"No valid data to plot for {jobname} with {metric}")
                return
            if metric in ['avg_plddt', 'ptm']:
                V_min =0 
                V_max =1
                CMAP = plt.cm.rainbow.reversed()
            else:
                V_min = np.min(confidence_filtered)
                V_max = np.max(confidence_filtered)
                CMAP = plt.cm.rainbow
            plt.figure(figsize=(8, 6))
            sc = plt.scatter(TMscore1_filtered, TMscore2_filtered, 
                            c=confidence_filtered, cmap=CMAP, s =3,
                            vmin=V_min, vmax=V_max, alpha=0.6)
            plt.colorbar(sc, label=metric)
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.axvline(x=TMscore12,ymin=0, ymax=TMscore12, color='k', linestyle='-', alpha=0.5)
            plt.axhline(y=TMscore12, xmin=0, xmax=TMscore12,color='k', linestyle='-', alpha=0.5)
            plt.xlabel(f'TMscore to {ID1}')
            plt.ylabel(f'TMscore to {ID2}')
            plt.title(f'{jobname} - TMscore colored by {metric}')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            # Save the plot
            save_path = os.path.join(save_dir, f"{jobname}_tmscore_{metric}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        error_msg = f"Error processing {jobname}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
def main():
    save_fig_dir_ss = base_result_dir+f"/TMscore_fig_SS/"
    os.makedirs(save_fig_dir_ss, exist_ok=True)
    save_fig_dir_nocoevol = f"{base_result_dir}TMscore_fig_enhanced_withoutcoevol/"
    save_fig_dir_enhanced = f"{base_result_dir}TMscore_fig_enhanced/"
    save_fig_dir_confd = f"{base_result_dir}TMscore_colored_with_confd/"
    os.makedirs(save_fig_dir_confd, exist_ok=True)
    os.makedirs(save_fig_dir_nocoevol, exist_ok=True)
    os.makedirs(save_fig_dir_enhanced, exist_ok=True)
    jobnames = ["2c1uC"]
    #jobnames = ["1ceeB","2ougC"]
    try:
        num_processes = multiprocessing.cpu_count() - 1  # Leave one core free
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Process jobnames in parallel
            # pool.map(partial(process_BSS_jobname, save_fig_dir=save_fig_dir_ss), jobnames)
            # print("finish SS")
            pool.map(partial(process_enhanced_jobname, save_fig_dir=save_fig_dir_enhanced), jobnames)
            print("finish enhance")
            #pool.map(partial(process_enhanced_nocevol_jobname, save_fig_dir=save_fig_dir_nocoevol), jobnames)
            # print("finish no coevol")
            #pool.map(partial(plot_tmscore_scatter_with_confidence, save_dir= save_fig_dir_confd), jobnames)
            print("finish all")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
    finally:
        # Clean up
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
    
