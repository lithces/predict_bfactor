#%%

import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

from streaming import LocalDataset, StreamingDataset
import torch
from torch.utils.data import random_split, ConcatDataset


class TokenTensorDataset(LocalDataset):
    def __init__(self, local, ctx_size):
        super().__init__(local=local)
        self.ctx_size = ctx_size+1 # need to add one for AR nature.


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        L = torch.tensor(obj['L'])
        y = obj['y']

        return L, y 
#%%
ds = TokenTensorDataset('ds/mds_train', 500)
dl = DataLoader(ds)

Ls = [xi[0] for xi in dl]
ys = [xi[1].view(-1,) for xi in dl]
# %%
s = torch.tensor(Ls).view(-1,)
#%%
ys_cat = torch.concat(ys)
# %%
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.DataFrame(data={'L': s})

# value_counts = df['L'].value_counts()
# plt.bar(value_counts.index, value_counts.values)

# # Adding labels and title
# plt.xlabel('Categories')
# plt.ylabel('Counts')
# plt.title('Value Counts')

# %%
df = pd.DataFrame(data={'y': ys_cat})
df['y_int'] = df['y'].map(lambda x: int(10*x))
value_counts = df['y_int'].value_counts()
plt.bar(value_counts.index, value_counts.values)

# # Adding labels and title
plt.xlabel('100*target')
plt.ylabel('Counts')
plt.title('Value Counts')

# %%
