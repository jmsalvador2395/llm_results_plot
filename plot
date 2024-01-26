#!/usr/bin/env python

import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd
import numpy as np
from datasets import Dataset


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

    ###############################################
    # get max scores over prompt levels per model #
    ###############################################
    max_per_model_data = {name: [] for name in ds_names}
    max_per_model_idxs = {name: [] for name in ds_names}

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
            max_per_model_idxs[name].append({**dict_head, **max_indices})

            
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
                    palette='Blues')
        ax[i].set_title(name_map[name])
        ax[i].tick_params(axis='x', labelrotation=20)

    plt.show()
    #breakpoint()



    """
    #for model in set(ds['model']):
    for lv in set(ds['level']):
        lv_ds = Dataset.from_list(list(
            filter(lambda x: x['level'] == lv, ds)
        ))
        breakpoint()

    df = ds.to_pandas()
    breakpoint()
    """
