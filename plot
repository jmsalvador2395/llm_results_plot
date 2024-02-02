#!/usr/bin/env python

import os
import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd
import numpy as np
from  matplotlib.colors import LinearSegmentedColormap
from datasets import Dataset
from scipy.stats import spearmanr



# TODO delete these
from pprint import pprint

if __name__ == '__main__':

    # read dataset
    ds = Dataset.from_csv('results.csv')

    # collect sets
    models = set(ds['Model'])
    ds_names = set(ds['Dataset'])
    name_map = {name: name.replace('_', ' ').title() for name in ds_names}
    metrics = sorted({key for key, val in ds.features.items() if val.dtype == 'float64'})

    # convert to list split data into its respective datasets
    ds_list = ds.to_list() 
    ds_dict = {name: list(filter(lambda x: x['Dataset'] == name, ds_list)) for name in ds_names}

    trgt = 'figs'
    if not os.path.exists(trgt):
        os.mkdir(trgt)

    """
    def avg(ds_list, ds_name, metrics, levels):
        #print(f'{metrics[-3]}\t\t{metrics[-2]}\t{metrics[-1]}')
        for lv in levels:
            ds = Dataset.from_list(list(filter(lambda x: x['Dataset'] == ds_name and x['Level'] == lv, ds_list)))
            print(ds[metrics[-3]])
            #print(f'{np.mean(ds[metrics[-3]]):.3f}\t\t\t{np.mean(ds[metrics[-2]]):.3f}\t{np.mean(ds[metrics[-1]]):.3f}')
    avg(ds_list, 'privacy_policy', metrics, set(ds['Level']))
    breakpoint()
    """

    ##################################
    # average score per prompt group #
    ##################################
    lvs = sorted(set(ds['Level']))
    avg_scores = []
    for name in ds_names:
        for lv in lvs:
            lv_ds = Dataset.from_list(list(filter(
                lambda x: x['Level'] == lv and x['Dataset'] == name,
                ds_list
            )))
            scores = {'Dataset': name, 'Level': lv}
            scores.update({metric:np.mean(lv_ds[metric]) for metric in metrics})
            avg_scores.append(scores)

    avg_score_df = pd.DataFrame(avg_scores)
    with open(trgt + '/avg_scores.tex', 'w') as f:
        f.write(avg_score_df.to_latex(
            index=False,
            float_format='%.3f',
            column_format='| l | r | r | r | r | r | r | r | r | r |',
            bold_rows=True,
        ))

    ###########################
    # build correlation table #
    ###########################
    plt.rcParams.update({'font.size': 20})
    wh = 10.0, 10.0
    sns.set(rc={'figure.figsize': wh})
    cor_mat = np.zeros((len(metrics), len(metrics)))

    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            score = spearmanr(ds[m1], ds[m2]).statistic
            cor_mat[i, j] = score

    fig, ax = plt.subplots()
    sns.heatmap(
        cor_mat,
        annot=True,
        vmin=0,
        xticklabels=metrics,
        yticklabels=metrics,
    )
    plt.title(r"Spearman Correlation Matrix for Each Metric")
    ax.tick_params(axis='x', labelrotation=10)
    ax.tick_params(axis='y', labelrotation=10)
    plt.savefig(trgt + '/metric_correlation.png')

    ###############################################
    # get max scores over prompt levels per model #
    ###############################################
    #plt.rcParams.update({'font.size': 56})
    wh = 22.0, 12.0
    sns.set(rc={'figure.figsize': wh})
    sns.set(font_scale=1.5)
    max_per_model_data = {name: [] for name in ds_names}
    max_per_model_idxs = {name: [] for name in ds_names}
    counts_per_lvl = {lv: 0 for lv in set(ds['Level'])}

    for name, data in ds_dict.items():
        for model in sorted(models):
            model_data = list(filter(lambda x: x['Model'] == model, data))
            model_ds = Dataset.from_list(model_data)

            #dict_head = {'Model': model, 'Dataset': name}
            dict_head = {'Model': model}
            maxes = {metric: max(model_ds[metric]) for metric in metrics}
            max_indices = {metric: np.argmax(model_ds[metric]) for metric in metrics}

            new_samples = [{'Model': model, 'Score': maxes[metric], 'Metric': metric} for metric in metrics]
            max_per_model_data[name] += new_samples

            #max_per_model_data[name].append({**dict_head, **maxes})
            # update level counts
            max_per_model_idxs[name].append({**dict_head, **max_indices})
            for metric in metrics:
                counts_per_lvl[max_per_model_idxs[name][-1][metric]] += 1

    color_palette = {
        'BERTscore': 'yellow',
        'Rouge-1': 'aquamarine',
        'Rouge-2': 'turquoise',
        'Rouge-L': 'darkturquoise',
        'Rouge-L Sum': 'lightseagreen',
        'Sem-F1 (Distil)': 'thistle',
        'Sem-F1 (RoBERTa)': 'plum',
        'Sem-F1 (USE)': 'violet',
    }

            
    # plot data
    fig, ax = plt.subplots(2, 1, sharex=True)
    dataframes = {name: pd.DataFrame(max_per_model_data[name]) for name in ds_names}

    for i, (name, df) in enumerate(dataframes.items()):
        legend = True if i==0 else False
        sns.barplot(x='Model',
                    y='Score',
                    hue='Metric',
                    data=df,
                    ax=ax[i],
                    legend=legend,
                    palette=color_palette)
        ax[i].set_title(name_map[name])
        ax[i].tick_params(axis='x', labelrotation=15)

    plt.savefig(
        trgt + '/max_scores_per_model.png',
        bbox_inches='tight',
        pad_inches=0,
        dpi=400,
    )
    plt.show()

    # compute count of max scores per level
    wh = 12.0, 12.0
    sns.set(rc={'figure.figsize': wh})
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots()

    x = np.array(list(counts_per_lvl.keys()))
    x = [f'Lv{lv}' for lv in x]
    y = np.array(list(counts_per_lvl.values()))
    color_palette = sns.color_palette('dark')
    plt.pie(y, labels=x, colors=color_palette)
    ax.set_title(f'Highest Scoring TeLER Prompts For Each Model (n={sum(y)})')
    plt.savefig(
        trgt + '/top_prompts_by_lv.png',
        bbox_inches='tight',
        pad_inches=0,
        dpi=400,
    )
    plt.show()
