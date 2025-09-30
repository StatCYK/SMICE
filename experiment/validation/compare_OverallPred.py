import os
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
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


# Color scheme
COLORS = {
    'SMICE': '#E15759',  # Red for SMICE
    'SMICE (no coevol)': '#FF9999',  # Light red for SMICE without coevolution
    'SMICE (not enhanced)': '#E15759',  # Red for SMICE (not enhanced)
    'AF-Cluster': 'blue',  # Blue for AF-Cluster
    'Random Sampling': '#000000'  # Black for Random
}

LINE_STYLES = {
    'SMICE': 'solid',
    'SMICE (no coevol)': 'dash',
    'SMICE (not enhanced)': 'dashdot',
    'AF-Cluster': 'solid',
    'Random Sampling': 'solid'
}

UPPER_QUANTILE_STYLE = 'dash'  # Style for 75% quantile lines
LOWER_QUANTILE_STYLE = 'dot'    # Style for 25% quantile lines


def calculate_Props(tmscores1, tmscores2, x_values):
    prop1 = []
    prop2 = []
    for x in x_values:
        # Prop where tmscores1 > tmscores2 AND tmscores1 > x
        cond1 = np.logical_and(tmscores1 > tmscores2, tmscores1 > x)
        p1 = np.mean(cond1) if len(cond1) > 0 else 0
        
        # Prop where tmscores2 > tmscores1 AND tmscores2 > x
        cond2 = np.logical_and(tmscores2 > tmscores1, tmscores2 > x)
        p2 = np.mean(cond2) if len(cond2) > 0 else 0
        
        prop1.append(p1)
        prop2.append(p2)
    return prop1, prop2

def process_jobname(jobname,  cov, fold_to_predict):
    try:
        output_SMICE = pd.read_json(f"{base_TMscores_output_dir}{jobname}/outputs_SMICE_TMscores.json.zip")
        output_SMICE = output_SMICE[output_SMICE['avg_plddt']>0.5]
        TMscores_SMICE = np.array([output_SMICE['TMscore1'],output_SMICE['TMscore2']])
        # Load AF-Cluster scores
        tmscores_clust = np.load(f"{base_dir_AFclust_random}{jobname}_TMscores_clust.npy")
        tmscores_clust = tmscores_clust.reshape((2, len(tmscores_clust)//2))
        
        # Load random scores
        tmscores_random = np.load(f"{base_dir_AFclust_random}{jobname}_TMscores_random.npy")
        tmscores_random = tmscores_random.reshape((2, len(tmscores_random)//2))
        
        if TMscores_SMICE.shape[1]>2 and tmscores_clust.shape[1]>0:
            # Get all TMscore values to determine x-axis range
            all_tmscores = np.concatenate([TMscores_SMICE.flatten(), tmscores_clust.flatten(), 
                                         tmscores_random.flatten()])
            x_values = np.linspace(max([0.2, np.min(all_tmscores)]), np.max(all_tmscores), 100)
            # Calculate Props for all methods
            results = {
                'jobname': jobname,
                'x_values': x_values,
                'fold': {},
                'max_prop': 0
            }

            # Calculate Props for all methods
            smice_p1, smice_p2 = calculate_Props(TMscores_SMICE[0], TMscores_SMICE[1], x_values)
            clust_p1, clust_p2 = calculate_Props(tmscores_clust[0], tmscores_clust[1], x_values)
            random_p1, random_p2 = calculate_Props(tmscores_random[0], tmscores_random[1], x_values)

            # Select which Prop to use based on fold_to_predict
            if fold_to_predict == "fold1":
                smice_min = [p1 for p1, p2 in zip(smice_p1, smice_p2)]
                clust_min = [p1 for p1, p2 in zip(clust_p1, clust_p2)]
                random_min = [p1 for p1, p2 in zip(random_p1, random_p2)]
                yaxis_title = "Prop of Predicting Fold1"
            elif fold_to_predict == "fold2":
                smice_min = [p2 for p1, p2 in zip(smice_p1, smice_p2)]
                clust_min = [p2 for p1, p2 in zip(clust_p1, clust_p2)]
                random_min = [p2 for p1, p2 in zip(random_p1, random_p2)]
                yaxis_title = "Prop of Predicting Fold2"
            elif fold_to_predict == "Mini":
                smice_min = [min(p1,p2) for p1, p2 in zip(smice_p1, smice_p2)]
                clust_min = [min(p1,p2) for p1, p2 in zip(clust_p1, clust_p2)]
                random_min = [min(p1,p2) for p1, p2 in zip(random_p1, random_p2)]
                yaxis_title = "Min-Prop of Predicting Two Folds"
            else:
                print("unregonize %s"%fold_to_predict)
                return None
            results['fold']['SMICE'] = smice_min
            results['fold']['AF-Cluster'] = clust_min
            results['fold']['Random Sampling'] = random_min
            results['yaxis_title'] = yaxis_title

            # Calculate max Prop for y-axis scaling
            all_props = smice_min + clust_min + random_min
            results['max_prop'] = max(max(all_props), 0.1)
            return results
        else:
            return None
    except Exception as e:
        print(f"{jobname} failed"+str(e))
        return None

def create_summary_plot(results, fold_to_predict):
    # Collect all x_values (they should be the same for all jobs)
    x_values = results[0]['x_values']
    # Initialize storage for all Props
    all_smice = []
    all_clust = []
    all_random = []
    
    for res in results:
        all_smice.append(res['fold']['SMICE'])
        all_clust.append(res['fold']['AF-Cluster'])
        all_random.append(res['fold']['Random Sampling'])
    
    # Convert to numpy arrays for easier calculations
    smice_arr = np.array(all_smice)
    clust_arr = np.array(all_clust)
    random_arr = np.array(all_random)
    
    # Calculate statistics
    def calculate_stats(arr):
        median = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)
        return median, q25, q75
    
    smice_median, smice_q25, smice_q75 = calculate_stats(smice_arr)
    clust_median, clust_q25, clust_q75 = calculate_stats(clust_arr)
    random_median, random_q25, random_q75 = calculate_stats(random_arr)
    
    # Create the figure
    fig = go.Figure()
    # Color scheme
    method_colors = {
        'SMICE': '#E15759',
        'AF-Cluster': 'blue',
        'Random Sampling': '#000000'
    }
    
    # Add a small constant to avoid log(0)
    epsilon = 1e-3
    
    # Add traces for each method with median line and quantile lines
    methods = [
        ('SMICE', smice_median, smice_q25, smice_q75),
        ('AF-Cluster', clust_median, clust_q25, clust_q75),
        ('Random Sampling', random_median, random_q25, random_q75)
    ]
    
    for name, median, q25, q75 in methods:
        # Apply log transformation with small offset
        log_median = np.log10(median + epsilon)
        log_q25 = np.log10(q25 + epsilon)
        log_q75 = np.log10(q75 + epsilon)
        
        # Add the median line (solid)
        fig.add_trace(go.Scatter(
            x=x_values,
            y=10**log_median,  # Show original values but plot in log space
            name=f'{name}',
            line=dict(color=method_colors[name], 
                      width=5,
                      dash='solid'),
            mode='lines',
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'TMscore: %{x:.2f}<br>' +
                         'Median: %{y:.3f}<extra></extra>'
        ))
        
    # Set y-axis title based on fold_to_predict
    yaxis_title = results[0]['yaxis_title']
    # Update layout for better visualization with log scale
    fig.update_layout(
        template='simple_white',
        width=600,
        height=400,
        margin=dict(l=80, r=40, t=60, b=60),
        font=dict(family="Arial", size=12, color="black"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font=dict(size=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,  # Move legend down to accommodate more entries
            xanchor="center",
            x=0.5,
            font=dict(size=16),  # Smaller font for more entries
            bgcolor='rgba(255,255,255,0.7)'
        ),
        xaxis_title="TMscore threshold",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        yaxis_type="log"
    )
    
    # Customize log axis to show original values
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='gray',
        title_text=yaxis_title,
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks='outside',
        tickwidth=1,
        tickcolor='black',
        #gridcolor='rgba(0,0,0,0.05)',
        title_font=dict(size=16),
        tickfont=dict(size=16),
        range=[np.log10(0.001), np.log10(1)] if fold_to_predict == "fold1" else [np.log10(0.001), np.log10(0.3)] ,  # Log range from 0.01 to 1
        tickvals=[0.0125, 0.025, 0.05, 0.1, 0.2,0.4,0.8],
        ticktext=["0.0125", "0.025", "0.05", "0.1", "0.2", "0.4", "0.8"],
        minor=dict(
            tickvals=np.logspace(-2,np.log10(1) , 21),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.02)',
            gridwidth=0.5
        )
    )
    
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='gray',
        title_text="TMscore threshold",
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        ticks='outside',
        tickwidth=1,
        tickcolor='black',
        #gridcolor='rgba(0,0,0,0.05)',
        title_font=dict(size=16),
        tickfont=dict(size=16),
        range=[min(x_values), max(x_values)]
    )
    
    return fig


def main():
    for fold in ["fold1","fold2","Mini"]:
        cov = 75
        with multiprocessing.Pool() as pool:
            results = pool.map(partial(process_jobname, cov=cov, fold_to_predict=fold), jobnames)

        # Filter out failed jobs
        valid_results = [res for res in results if res is not None]
        # Create output directory
        output_dir = f"{base_result_dir}compare_OverallPred/{fold}/"
        os.makedirs(output_dir, exist_ok=True)

        # Create and save summary plot
        figure_summary = create_summary_plot(valid_results, fold)
        figure_summary.write_image(f"{output_dir}Prop{fold}_summary.png", 
                           scale=2, width=600, height=400)

if __name__ == "__main__":
    main()