#!/usr/bin/env python

import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datasets import Dataset

# TODO delete these
from pprint import pprint

if __name__ == '__main__':
    ds = Dataset.from_csv('results.csv')

    fnames = ds['Model']
    new_ds = []
    for fname in fnames:
        fsplit = fname.split('/')[1:]
        level = fsplit.pop(-1)
        level = level.replace('level', '').replace('.json', '')
        new_ds.append({
            'Dataset': fsplit.pop(0),
            'level': int(level),
            'model': '/'.join(fsplit),
        })
    new_ds = Dataset.from_list(new_ds)
    ds = datasets.concatenate_datasets([new_ds, ds], axis=1)
    ds = ds.remove_columns('Model')

    #for model in set(ds['model']):
    for lv in set(ds['level']):
        lv_ds = Dataset.from_list(list(
            filter(lambda x: x['level'] == lv, ds)
        ))
        breakpoint()

    df = ds.to_pandas()
    breakpoint()
