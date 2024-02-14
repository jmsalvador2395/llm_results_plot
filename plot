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
from colorama import Fore, Back, Style

# TODO delete these
from pprint import pprint

if __name__ == '__main__':

    # read dataset
    ds = Dataset.from_csv('results.csv')
    out_ext = '.png'

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

    hatch_patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

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
    wh = 15.0, 18.0
    sns.set(rc={'figure.figsize': wh}, font_scale=2.0)
    cor_mat = np.zeros((len(metrics), len(metrics)))

    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            score = spearmanr(ds[m1], ds[m2]).statistic
            cor_mat[i, j] = score

    fig, ax = plt.subplots()
    sns.heatmap(
        cor_mat[1:, :-1],
        #cor_mat,
        annot=True,
        vmin=0,
        xticklabels=metrics[:-1],
        yticklabels=metrics[1:],
        #xticklabels=metrics,
        #yticklabels=metrics,
        mask=np.triu(np.ones(cor_mat[1:, :-1].shape, dtype=bool), k=1),
        #mask=np.triu(np.ones(cor_mat.shape, dtype=bool)),
    )
    plt.title(r"Spearman Correlation Matrix for Each Metric")
    #ax.tick_params(axis='x', labelrotation=15)
    ax.tick_params(axis='y', labelrotation=15)
    plt.gcf().subplots_adjust(
        bottom=0.15,
        left=0.15,
    )
    plt.tight_layout()
    plt.savefig(trgt + '/metric_correlation' + out_ext)

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

    # truncate from model names
    for dsn in max_per_model_data.keys():
        for item in max_per_model_data[dsn]:
            item['Model'] = item['Model'].split('/')[-1]

    color_palette = {
        'BERTscore': 'red',
        'Rouge-1': 'green',
        'Rouge-2': 'limegreen',
        'Rouge-L': 'lime',
        'Rouge-L Sum': 'lightgreen',
        'Sem-F1 (Distil)': 'midnightblue',
        'Sem-F1 (RoBERTa)': 'royalblue',
        'Sem-F1 (USE)': 'skyblue',
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
        labels = []
        for model in df['Model']:
            if model not in labels:
                labels.append(model)
            
        ax[i].set_title(name_map[name])
        ax[i].set_xticklabels(labels, rotation=15, ha='right')

    plt.savefig(
        trgt + '/max_scores_per_model' + out_ext,
        bbox_inches='tight',
        pad_inches=0,
        dpi=400,
    )


    #################################################################################
    # get max scores over prompt levels per model for the largest models per family #
    #################################################################################
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

    # filter out smaller models
    keep_models = [
        'chat-bison-001',
        'gpt-4',
        'mistralai/Mistral-7B-Instruct-v0.2',
        'lmsys/vicuna-13b-v1.5-16k',
        'mosaicml/mpt-30b-instruct',
        'meta-llama/Llama-2-13b-chat-hf',
    ]
    keep_metrics = [
        'BERTscore',
        'Rouge-2',
        'Sem-F1 (Distil)',
    ]

    for dsn, dsl in max_per_model_data.items():
        max_per_model_data[dsn] = list(filter(
            lambda x: x['Model'] in keep_models, 
            dsl
        ))

    # truncate from model names
    for dsn in max_per_model_data.keys():
        for item in max_per_model_data[dsn]:
            item['Model'] = item['Model'].split('/')[-1]

    dataframes = {name: pd.DataFrame(max_per_model_data[name]) for name in ds_names}

    # plot data
    fig, ax = plt.subplots(2, 1, sharex=True)
    for i, (name, df) in enumerate(dataframes.items()):
        legend = True if i==0 else False
        for j, bar in enumerate(sns.barplot(x='Model',
                                            y='Score',
                                            hue='Metric',
                                            data=df,
                                            ax=ax[i],
                                            legend=legend,
                                            palette=color_palette).patches):
            #bar.set_hatch(hatch_patterns[j // 6])
            pass
        ax[i].set_title(name_map[name])
        #ax[i].tick_params(axis='x', labelrotation=15)

    plt.savefig(
        trgt + '/max_scores_per_model_family' + out_ext,
        bbox_inches='tight',
        pad_inches=0,
        dpi=400,
    )




    #########################################
    # compute count of max scores per level #
    #########################################
    wh = 12.0, 12.0
    pie_font=30
    plt.rcParams.update({'font.size': pie_font})
    fig, ax = plt.subplots()

    x = np.array(list(counts_per_lvl.keys()))
    x = [f'L{lv}' for lv in x]
    y = np.array(list(counts_per_lvl.values()))
    color_palette = sns.color_palette('bright')
    shadow_dist = 0.006
    _, _, autotexts = plt.pie(
        y,
        labels=x,
        colors=color_palette,
        textprops={'fontsize': pie_font},
        #autopct='%1.1f%%',
        autopct=lambda x: f'{x:.01f}%' if x > 3 else None,
        #shadow=False,
        shadow={
            'ox': -shadow_dist,
            'oy': shadow_dist,
            'edgecolor': 'none',
            'shade': 0.5},
        explode=(0.00, 0.05 , 0.00, 0.00, 0.00),
        startangle=45,
        #pctdistance=.8,
        wedgeprops={'edgecolor': 'k',
                    'linewidth': 0,
                    #'linestyle': 'dashed',
                    'antialiased': True},

    )
    for autotext in autotexts:
        autotext.set_color('white')
    ax.set_title(f'Highest Scoring TeLER Prompts For Each Model (n={sum(y)})', fontsize=pie_font)
    plt.savefig(
        trgt + '/top_prompts_by_lv' + out_ext,
        bbox_inches='tight',
        pad_inches=0,
        dpi=500,
    )
