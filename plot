#!/usr/bin/env python

import os
import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd
import numpy as np
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

    # build correlation table
    cor_mat = np.zeros((len(metrics), len(metrics)))
    out_table =  '| | ' + ' | '.join(metrics) + ' |\n'
    out_table += '| - '*(out_table.count('|')-1) + '|\n'

    for i, m1 in enumerate(metrics):
        row = f'| {metrics[i]} |'
        for j, m2 in enumerate(metrics):
            score = spearmanr(ds[m1], ds[m2]).statistic
            cor_mat[i, j] = score
            if j >= i:
                row += f' - |'
            else:
                row += f' {score:.2f} |'
        out_table += row + '\n'
    with open(trgt + '/correlation_table.md', 'w') as f:
        f.write(out_table)

    cor_mat = np.tril(cor_mat)
    np.fill_diagonal(cor_mat, 0)

    ###############################################
    # get max scores over prompt levels per model #
    ###############################################
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
                    #palette='Blues')
                    palette=color_palette)
        ax[i].set_title(name_map[name])
        ax[i].tick_params(axis='x', labelrotation=20)

    plt.show()

    # compute count of max scores per level
    fig, ax = plt.subplots()
    ax.set_title(r'# Times Each TeLER Level Scored Highest')

    x = np.array(list(counts_per_lvl.keys()))
    x = [f'Lv{lv}' for lv in x]
    y = np.array(list(counts_per_lvl.values()))
    sns.barplot(x=x, y=y)
    plt.show()
