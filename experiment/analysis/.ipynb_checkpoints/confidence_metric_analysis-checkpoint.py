from scipy import stats
import seaborn as sns
import os
import json
import glob
import multiprocessing
from functools import partial
import pandas as pd
import numpy as np
import random
import string
import sys
import pickle
import traceback
import plotly.graph_objects as go
import zipfile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

with open('../config/config_SMICE_benchmark.json', 'r') as f:
    config = json.load(f)

metadata_92 = pd.read_csv(config["meta_path"])
base_output_dir = config["base_output_dir"]
base_result_dir = config["base_result_dir"]
jobnames = config["jobnames"]
true_pdb_path = config["true_pdb_path"]

def plot_confidence_vs_tmscore(jobname, outputs_SMICE, save_dir):
    """Plot confidence metrics vs max TMscore with Spearman correlation"""
    try:
        # Filter out rows with None values
        valid_data = outputs_SMICE.dropna(subset=['max_TMscore', 'avg_plddt', 'avg_pae', 'ptm']).copy()
        meta_info = metadata_92[metadata_92['jobnames'] == jobname].iloc[0]
        sequence = meta_info['sequences']
        ID1 = meta_info['Fold1']
        ID2 = meta_info['Fold2']
        if len(valid_data) > 30:
            results = []
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Plot settings
            metrics = ['avg_plddt', 'ptm', 'avg_pae']
            titles = [f'Average pLDDT vs max TMscore of FS protein({ID1}/{ID2})', 
                    f'PTM vs max TMscore of FS protein({ID1}/{ID2})',
                    f'Average PAE vs max TMscore of FS protein({ID1}/{ID2})']
            ylabels = ['Average pLDDT', 'PTM', 'Average PAE']
            
            for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
                ax = axes[i]
                
                # Plot scatter points colored by source
                sources = valid_data['source'].unique()
                TM_score_diff = np.sign(valid_data['TMscore1']-valid_data['TMscore2'])*valid_data['max_TMscore']
                v_abs_max = np.max(np.max(TM_score_diff))
                scatter =ax.scatter(valid_data['max_TMscore'], valid_data[metric], 
                          c=TM_score_diff,cmap = "RdYlBu", vmin = -v_abs_max,vmax = v_abs_max,alpha=0.6, s=20)
                cbar = plt.colorbar(scatter)
                cbar.set_label('Signed Max-TMscore')
                # Calculate Spearman correlation
                x = valid_data['max_TMscore']
                y = valid_data[metric]
                spearman_corr, spearman_pvalue = stats.spearmanr(x, y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                # Plot regression line
                x_vals = np.linspace(0, 1, 100)
                y_vals = intercept + slope * x_vals
                ax.plot(x_vals, y_vals, color='red', linestyle='--')
                # Add correlation info to plot
                ax.text(0.05, 0.95 ,
                        f'spearman correlation = {spearman_corr:.3f}\np-value = {spearman_pvalue:.3e}',
                        transform=ax.transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel('max TMscore')
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.set_xlim(0,1)
                if metric in ['avg_plddt', 'ptm']:
                    ax.set_ylim(0,1)
                ax.legend()
                
                # Store results
                results.append({
                    'jobname': jobname,
                    'metric': metric,
                    'spearman_corr': spearman_corr,
                    'spearman_pvalue': spearman_pvalue
                })
            
            plt.tight_layout()
            
            # Ensure save directory exists
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, f"{jobname}_confidence_vs_max_TMscore.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            #results_path = os.path.join(save_dir, f"{jobname}.json")
            with open(os.path.join(save_dir, f"{jobname}.json"), 'w') as f:
                json.dump(results, f)

    except Exception as e:
        print(f"Error plotting for {jobname}: {str(e)}")
        

def plot_overlay_scatter_all_jobnames(jobnames, base_output_dir, save_dir, point_size=5, alpha=0.5):
    """
    Create overlay scatter plots for all jobnames for each metric.
    
    Args:
        jobnames (list): List of jobnames to include in the plots
        base_output_dir (str): Base directory where jobname results are stored
        save_dir (str): Directory to save the output plots
        point_size (int): Base size for scatter points
        alpha (float): Transparency level for points (0-1)
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize a dictionary to store all combined data
        all_data = []
        
        # Load data for all jobnames
        for jobname in jobnames:
            combined_path = f"{base_output_dir}{jobname}/outputs_SMICE.json.zip"
            if os.path.exists(combined_path):
                try:
                    df = pd.read_json(combined_path)
                    df['jobname'] = jobname
                    all_data.append(df[::10])
                except Exception as e:
                    print(f"Error loading data for {jobname}: {str(e)}")
        
        if not all_data:
            print("No valid data found for any jobname")
            return
        
        # Combine all data
        combined_all = pd.concat(all_data, ignore_index=True)
        
        # Filter out rows with None values
        valid_data = combined_all.dropna(subset=['TMscore1','TMscore2', 'avg_plddt', 'avg_pae', 'ptm'])
        
        if len(valid_data) == 0:
            print("No valid data points after filtering")
            return
        
        # Define metrics to plot
        metrics = [
            ('avg_plddt', 'Average pLDDT', (0, 1)),
            ('ptm', 'PTM', (0, 1)),
            ('avg_pae', 'Average PAE', (None, None))
        ]
        
        # Create a figure for each metric
        
        for metric, ylabel, ylim in metrics:
            for compared_metric in ['max_TMscore','TMscore1','TMscore2']:
                fig = plt.figure(figsize=(8, 6))
                gs = GridSpec(2, 2,  
                    height_ratios=[1, 4],  # Hist row, scatter row
                    width_ratios=[5, 1],  # Scatter, y-hist, scatter, y-hist
                    hspace=0.015,  # Very small space between hist and scatter
                    wspace=0.015)  # Very small space between scatter and y-hist
                ax_scatter = fig.add_subplot(gs[1, 0])
                ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
                ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

                ax_scatter.scatter(valid_data[compared_metric], valid_data[metric], label=metric,color = '#2E6DB4',
                           s=point_size, alpha=alpha, edgecolor='none')
                ax_scatter.set_xlabel(compared_metric, fontsize=14)
                ax_scatter.set_ylabel(ylabel, fontsize=14)
                ax_scatter.grid(True, color='gray', linestyle='-', linewidth=0.7, alpha=0.7)
                ax_scatter.set_facecolor('#f5f5f5')
                ax_histy.hist(valid_data[metric], color = '#2E6DB4', bins=40, alpha=0.5, edgecolor='black',density=True, orientation='horizontal')
                ax_histx.hist(valid_data[compared_metric],color = '#2E6DB4',bins=40, alpha=0.5, edgecolor='black',density=True)
                ax_histx.tick_params(axis='x', labelbottom=False)
                ax_histx.tick_params(axis='y', labelleft=False)
                ax_histy.tick_params(axis='x', labelbottom=False)
                ax_histy.tick_params(axis='y', labelleft=False)
                # Share axes for proper alignment
                ax_histx.sharex(ax_scatter)
                ax_histy.sharey(ax_scatter)
                if metric == "plddt":
                    ax_scatter.set_ylim(0.2,1)
                    ax_scatter.axhline(y=0.5, color='r', linestyle='--', linewidth=2)
                    ax_histy.axhline(y=0.5, color='r', linestyle='--', linewidth=2)
                elif metric == "ptm":
                    ax_scatter.set_ylim(0,1)
                else:
                    ax_scatter.set_ylim(0,np.max(valid_data[metric]))
                
                for ax in [ax_histx, ax_histy]:
                    ax.set_axis_off()  # Completely remove axes
            # Save the plot
                plt.tight_layout()
                plt.savefig(f"{save_dir}/overlay_{metric}_vs_{compared_metric}.png", dpi=300, bbox_inches='tight')
                plt.close()

    except Exception as e:
        print(f"Error in plot_overlay_scatter_all_jobnames: {str(e)}")
        traceback.print_exc()

def plot_summary_results(all_results, save_dir):
    """Plot text labels (jobnames) of Spearman correlation vs p-values (log scale) for each metric"""
    if not all_results:
        return
    
    try:
        # Prepare data for summary plots
        summary_df = pd.DataFrame(all_results)
        # Create a figure for each metric
        metrics = [
            ('avg_plddt', 'Average pLDDT','o'),
            ('ptm', 'PTM','s'),
            ('avg_pae', 'Average PAE','^')
        ]
        for compared_metric in ['max_TMscore','TMscore1','TMscore2']:
            plt.figure(figsize=(10, 6))
            for metric,metric_label,marker_type in metrics:
                # Filter data for current metric
                metric_data = summary_df[summary_df['metric'] == metric]
                # Create text plot (jobnames instead of points)
                plt.scatter(np.log10(metric_data['spearman_pvalue']), metric_data['spearman_corr'],
                         s=45,label=metric_label,marker = marker_type,
                         alpha=0.8)
                # Add significance threshold line
            plt.axvline(x=np.log10(0.05), color='r', linestyle='--', alpha=0.5, label='p-value=0.05')
            plt.axhline(y=0, color='k', linestyle='--', alpha=1, label='spearman correlation = 0')
            # Set log scale for x-axis (p-values)
            plt.xlim(-350,10)
            plt.ylim(-1.1, 1.1)  # Spearman correlation ranges from -1 to 1
            # Add labels and title
            plt.ylabel(f'Spearman Correlation to {compared_metric}', fontsize=15)
            plt.xlabel('p-value (log scale)', fontsize=15)
            #plt.title(f'Spearman Correlation vs p-value', fontsize=14)
            # Add grid and legend
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=15)
            # Adjust layout to prevent label clipping
            plt.tight_layout()
            # Save the plot
            #print(f"Saved labeled Spearman correlation vs p-value plot for {metric}")
            plt.savefig(f"{save_dir}/spearman_vs_pvalue_labeled_{compared_metric}.png", dpi=300)
            plt.close()
            # Save summary data
            summary_df.to_csv(f"{save_dir}/spearman_summary_{compared_metric}.csv", index=False)
        
    except Exception as e:
        print(f"Error creating summary plots: {str(e)}")
        import traceback
        traceback.print_exc()

def process_jobname_with_confidence(jobname, save_fig_dir, cov):
    """Process jobname including confidence metric analysis"""
    try:
        outputs_SMICE = pd.read_json(f"{base_output_dir}{jobname}/outputs_SMICE.json.zip")
        if outputs_SMICE is not None:
            # Create directory for confidence plots
            confidence_dir = os.path.join(save_fig_dir, "confidence_plots")
            os.makedirs(confidence_dir, exist_ok=True)
            # Generate confidence vs TMscore plots
            plot_confidence_vs_tmscore(jobname, outputs_SMICE, confidence_dir)
    except Exception as e:
        error_msg = f"Error processing {jobname}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise

def calculate_filter_ratios(jobname):
    try:
        outputs_SMICE = pd.read_json(f"{base_output_dir}{jobname}/outputs_SMICE.json.zip")
        filtered_data = outputs_SMICE[outputs_SMICE['avg_plddt'] > 0.5]
        # Calculate ratio of filtered to combined data
        ratio = len(filtered_data) / len(outputs_SMICE)
        return ratio
    except Exception as e:
        error_msg = f"Error processing {jobname}: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
def main():
    save_fig_dir = f"{base_result_dir}analysis/analyze_confidence/"
    os.makedirs(save_fig_dir, exist_ok=True)
    cov = 75
    # Create a pool of workers
    num_processes = multiprocessing.cpu_count() - 5  # Leave one core free
    pool = multiprocessing.Pool(processes=num_processes)
    try:
        # Process jobnames in parallel and collect results
        pool.map(partial(process_jobname_with_confidence, 
                                 save_fig_dir=save_fig_dir, 
                                 cov=cov), 
                          jobnames)
        result_files = glob.glob(f"{save_fig_dir}confidence_plots/*.json")
        jobnames_validate = [file[-10:-5] for file in result_files]
        all_results = []
        for file in result_files:
            with open(file, 'r') as f:
                data = json.load(f)
                all_results.extend(data)  # Append results to list
        # Create summary plots
        summary_dir = os.path.join(save_fig_dir, "summary_plots")
        os.makedirs(summary_dir, exist_ok=True)
        print("start plotting summary figure")
        plot_summary_results(all_results, summary_dir)
        plot_overlay_scatter_all_jobnames(jobnames_validate, base_output_dir, summary_dir, point_size=10, alpha=0.3)
        ratios = pool.map(partial(calculate_filter_ratios), jobnames_validate)
        plt.figure(figsize=(10, 6))
        plt.hist(ratios, bins=30, edgecolor='black')
        plt.title('Histogram of Filtered Data Ratios (Filtered/Combined)')
        plt.xlabel('Ratio of Filtered to Combined Data')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        # Save the histogram
        plt.savefig(f"{summary_dir}/filter_ratio_histogram.png")
        plt.close()
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
    finally:
        # Clean up
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()