import os
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
os.chdir("../../src/")

with open('../config/config_SMICE_benchmark.json', 'r') as f:
    config = json.load(f)

metadata_92 = pd.read_csv(config["meta_path"])
base_output_dir = config["base_output_dir"]
base_result_dir = config["base_result_dir"]
jobnames = config["jobnames"]
true_pdb_path = config["true_pdb_path"]
base_TMscores_output_dir = config['base_TMscores_output_dir']
base_dir_AFclust_random = config['base_dir_AFclust_random']


compare_coreset = False # whether compare coreset or full set of the SMICE's sampling step

def process_tmscores(TMscores, samp_size, nrep=500):
    np.random.seed(123)
    Precision = np.mean(np.max(TMscores, 0))
    Recall1 = sum(np.max(TMscores[0, np.random.choice(TMscores.shape[1], size=samp_size)]) 
              for _ in range(nrep))/nrep
    Recall2 = sum(np.max(TMscores[1, np.random.choice(TMscores.shape[1], size=samp_size)]) 
              for _ in range(nrep))/nrep
    return Precision, Recall1, Recall2

def process_jobname(jobname, cov, mode):
    try:
        if compare_coreset == False:
            output_SMICE = pd.read_json(f"{base_TMscores_output_dir}{jobname}/outputs_SMICE_TMscores.json.zip")
            TMscores_SMICE = np.array([output_SMICE['TMscore1'],output_SMICE['TMscore2']])
        else:
            coreset_dir = f"{base_TMscores_output_dir}{jobname}/coreset_greedy_fast"
            coreset_data = pd.read_json(f"{coreset_dir}/coreset.json.zip")
            TMscores_SMICE = np.array([coreset_data['TMscore1'],coreset_data['TMscore2']])
        if mode in ["clust", "random"]:
            tmscores_comp = np.load(f"{base_dir_AFclust_random}{jobname}_TMscores_{mode}.npy")
            tmscores_comp = tmscores_comp.reshape((2, len(tmscores_comp)//2))
        else:
            print(f"competing method: {mode} not recognized")
            return None
        if mode == "clust":
            samp_size = tmscores_comp.shape[1]
        else:
            samp_size = min(TMscores_SMICE.shape[1], tmscores_comp.shape[1] )
        if TMscores_SMICE.shape[1] > 0 and tmscores_comp.shape[1] > 0:
            print(TMscores_SMICE.shape)
            # Process precision/recall metrics
            prec_ss, rec1_ss, rec2_ss = process_tmscores(TMscores_SMICE, samp_size)
            prec_comp, rec1_comp, rec2_comp = process_tmscores(tmscores_comp, samp_size)
            num_ss = TMscores_SMICE.shape[1]
            num_comp = tmscores_comp.shape[1]
            return [prec_ss, rec1_ss, rec2_ss, 
                    prec_comp, rec1_comp, rec2_comp,
                    num_ss, num_comp]
        return None
    except Exception as e:
        print(jobname+ str(e))
        return None

   
def create_figure(Res_clust, Res_random, valid_jobnames, mode='markers'):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
    
    # Nature style colors
    metric_colors = {
        'max-TMscore1': '#F28E2B',
        'max-TMscore2': '#E15759',
        'minimax-TMscore': '#59A14F'
    }
    
    metrics = ['max-TMscore1', 'max-TMscore2', 'minimax-TMscore']
    
    # Create figure with proper height ratios (smaller histograms)
    fig = plt.figure(figsize=(14, 24))
    main_gs = GridSpec(3, 1, hspace=0.2, figure=fig)  # Normal spacing between metric rows
    
    for i, metric in enumerate(metrics):
        # Get data for this metric
        if metric == 'max-TMscore1':
            x_clust = Res_clust[:,4]
            y_clust = Res_clust[:,1]
            x_random = Res_random[:,4]
            y_random = Res_random[:,1]
        elif metric == 'max-TMscore2':
            x_clust = Res_clust[:,5]
            y_clust = Res_clust[:,2]
            x_random = Res_random[:,5]
            y_random = Res_random[:,2]
        else:
            x_clust = np.min(Res_clust[:,4:6], axis=1)
            y_clust = np.min(Res_clust[:,1:3], axis=1)
            x_random = np.min(Res_random[:,4:6], axis=1)
            y_random = np.min(Res_random[:,1:3], axis=1)
        
        
        # Calculate row positions
        scatter_row =  1
        hist_row =  0
        
        gs = GridSpecFromSubplotSpec(2, 4, subplot_spec=main_gs[i], 
                                        height_ratios=[1, 4],  # Hist row, scatter row
                                        width_ratios=[3, 0.7, 3, 0.7],  # Scatter, y-hist, scatter, y-hist
                                        hspace=0.015,  # Very small space between hist and scatter
                                        wspace=0.015)  # Very small space between scatter and y-hist
        
        # Create subplots for AF-Cluster (left side)
        ax_scatter1 = fig.add_subplot(gs[scatter_row, 0])
        ax_histx1 = fig.add_subplot(gs[hist_row, 0], sharex=ax_scatter1)
        ax_histy1 = fig.add_subplot(gs[scatter_row, 1], sharey=ax_scatter1)
        
        # Create subplots for Random (right side)
        ax_scatter2 = fig.add_subplot(gs[scatter_row, 2])
        ax_histx2 = fig.add_subplot(gs[hist_row, 2], sharex=ax_scatter2)
        ax_histy2 = fig.add_subplot(gs[scatter_row, 3], sharey=ax_scatter2)
        
        # Plot scatter and histograms
        color = metric_colors[metric]
        
        # AF-Cluster
        ax_scatter1.scatter(x_clust, y_clust, color=color, alpha=0.7, s=45)
        ax_histx1.hist(x_clust, bins=15, color=color, alpha=0.7, edgecolor='black')
        ax_histy1.hist(y_clust, bins=15, orientation='horizontal', color=color, alpha=0.7, edgecolor='black')
        
        # Random
        ax_scatter2.scatter(x_random, y_random, color=color, alpha=0.7, s=45)
        ax_histx2.hist(x_random, bins=15, color=color, alpha=0.7, edgecolor='black')
        ax_histy2.hist(y_random, bins=15, orientation='horizontal', color=color, alpha=0.7, edgecolor='black')
        
        # Remove ticks from histograms
        ax_histx1.tick_params(axis='x', labelbottom=False)
        ax_histx1.tick_params(axis='y', labelleft=False)
        ax_histy1.tick_params(axis='x', labelbottom=False)
        ax_histy1.tick_params(axis='y', labelleft=False)
        
        ax_histx2.tick_params(axis='x', labelbottom=False)
        ax_histx2.tick_params(axis='y', labelleft=False)
        ax_histy2.tick_params(axis='x', labelbottom=False)
        ax_histy2.tick_params(axis='y', labelleft=False)
        
        ax_scatter2.tick_params(axis='y', labelleft=False)
        
        # Share axes for proper alignment
        ax_histx1.sharex(ax_scatter1)
        ax_histy1.sharey(ax_scatter1)
        ax_histx2.sharex(ax_scatter2)
        ax_histy2.sharey(ax_scatter2)
         # Set white background for all subplots
        for ax in [ax_scatter1, ax_scatter2]:
            ax.set_facecolor('#f5f5f5')
        # Add titles
        ax_histx1.set_title(f"{metric}: SMICE vs AF-Cluster", fontsize=14, pad=10, fontweight='bold')
        ax_histx2.set_title(f"{metric}: SMICE vs Random Sampling", fontsize=14, pad=10, fontweight='bold')
        for ax in [ax_histx1, ax_histy1, ax_histx2, ax_histy2]:
            ax.set_axis_off()  # Completely remove axes
        
        # Add axis labels
        if i == 2:  # Bottom row
            ax_scatter1.set_xlabel('AF-Cluster', fontsize=14)
            ax_scatter2.set_xlabel('Random Sampling', fontsize=14)
        ax_scatter1.set_ylabel('SMICE', fontsize=14)
        
        # Add diagonal lines
        ax_scatter1.plot([0, 1], [0, 1], 'k--', linewidth=3, alpha=0.7)
        ax_scatter2.plot([0, 1], [0, 1], 'k--', linewidth=3, alpha=0.7)

        # Set limits for TM scores
        ax_scatter1.set_xlim(0, 1)
        ax_scatter1.set_ylim(0, 1)
        ax_scatter2.set_xlim(0, 1)
        ax_scatter2.set_ylim(0, 1)
        
        # Add grid lines
        ax_scatter1.grid(True, color='gray', linestyle='-', linewidth=0.7, alpha=0.7)
        ax_scatter2.grid(True, color='gray', linestyle='-', linewidth=0.7, alpha=0.7)
        
        # Set font sizes
        ax_scatter1.tick_params(labelsize=12)
        ax_scatter2.tick_params(labelsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_figure_num_preds(Res_clust, save_path=None):
    """
    Implementation using matplotlib with smaller histogram heights
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    # Create figure with proper height ratios (smaller histograms)
    fig = plt.figure(figsize=(9, 6))
    
    gs = GridSpec(2, 2,  
                height_ratios=[1, 4],  # Hist row, scatter row
                width_ratios=[3, 0.7],  # Scatter, y-hist, scatter, y-hist
                hspace=0.015,  # Very small space between hist and scatter
                wspace=0.015)  # Very small space between scatter and y-hist
        
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    # Plot scatter and histograms
    x_clust = Res_clust[:,7]
    y_clust = Res_clust[:,6]
    print(max(x_clust))
    # AF-Cluster
    ax_scatter.scatter(x_clust, y_clust, color='#B07AA1', alpha=0.7, s=45)
    ax_histx.hist(x_clust, bins=35, color='#B07AA1', alpha=0.7, edgecolor='black')
    ax_histy.hist(y_clust, bins=20, orientation='horizontal', color='#B07AA1', alpha=0.7, edgecolor='black')

    # Remove ticks from histograms
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histx.tick_params(axis='y', labelleft=False)
    ax_histy.tick_params(axis='x', labelbottom=False)
    ax_histy.tick_params(axis='y', labelleft=False)
    # Share axes for proper alignment
    ax_histx.sharex(ax_scatter)
    ax_histy.sharey(ax_scatter)
    ax_scatter.set_facecolor('#f5f5f5')
    for ax in [ax_histx, ax_histy]:
        ax.set_axis_off()  # Completely remove axes

    ax_scatter.set_xlabel('Num of Preds(AF-Cluster)', fontsize=14)
    ax_scatter.set_ylabel('Num of Preds(SMICE)', fontsize=14)
    ax_scatter.plot([0, 4600], [0, 460], 'k--', linewidth=2, alpha=0.7)

    # Set limits for pairwise-TMscore
    ax_scatter.set_xlim(0, 4600)
    ax_scatter.set_ylim(0, 200)

    # Add grid lines
    ax_scatter.grid(True, color='gray', linestyle='-', linewidth=0.7, alpha=0.7)
    
    # Set font sizes
    ax_scatter.tick_params(labelsize=12)
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to {save_path}")

def main():
    cov = 75
    with multiprocessing.Pool() as pool:
        Res_clust = pool.map(partial(process_jobname, cov=cov, mode='clust'), jobnames)
        Res_random = pool.map(partial(process_jobname, cov=cov, mode='random'), jobnames)

    valid_indices = [i for i, (c, r) in enumerate(zip(Res_clust, Res_random)) 
                   if c is not None and r is not None]
    valid_jobnames = [jobnames[i] for i in valid_indices]
    print((len(jobnames),len(valid_jobnames)))
    print(valid_jobnames)
    Res_clust = np.array([Res_clust[i] for i in valid_indices])
    Res_random = np.array([Res_random[i] for i in valid_indices])
    fig = create_figure(Res_clust, Res_random, valid_jobnames, 'markers')
    os.makedirs(base_result_dir+"compare_TopPred/", exist_ok=True)
    if compare_coreset == True:
        fig.savefig(f"{base_result_dir}compare_TopPred/comparison_plots_coreset.png",dpi=600)
        create_figure_num_preds(Res_clust, save_path=f"{base_result_dir}compare_TopPred/comparison_num_preds.png")
    else:
        fig.savefig(f"{base_result_dir}compare_TopPred/comparison_plots_fullset.png",dpi=600)

if __name__ == "__main__":
    main()
    
    
