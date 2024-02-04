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

from torch.utils.data import DataLoader
from streaming import LocalDataset, StreamingDataset
import torch
from torch.utils.data import random_split, ConcatDataset
import numpy as np

class OneSeqDataset(LocalDataset):
    def __init__(self, local):
        super().__init__(local=local)


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        return {'L': obj['L'] \
            ,'c_res_id' :obj['c_res_id'] \
            ,'c_ss_id' : obj['c_ss_id'] \
            ,'c_coord' : obj['c_coord'] \
            ,'c_chi' : obj['c_chi'] \
            ,'id_orig': obj['id_orig'] \
            ,'y': obj['y']
        }

class BatchSeqDataset(LocalDataset):
    def __init__(self, local, ctx_size=512):
        '''
        resulting tensors are padded to ctx_size
        '''
        super().__init__(local=local)
        self.ctx_size = ctx_size


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        L, res_id0, y0 = int(obj['L']), obj['c_res_id'], obj['y']
        pos1 = min(L, self.ctx_size)
        res_id = np.concatenate( (res_id0, np.array([21]*(self.ctx_size - pos1))))
        y = np.concatenate( (y0, np.array([0.0]*(self.ctx_size - pos1))))
        mask = np.concatenate( (np.array([0]*pos1, dtype=np.bool8), np.array([1]*(self.ctx_size - pos1),  dtype=np.bool8)))
        return {'L': obj['L'] \
            ,'c_res_id' :res_id \
            ,'mask': mask
            ,'y': y
        }

#%%
import tqdm
ds_train = BatchSeqDataset('./ds/mds_train', ctx_size=512)
ds_val = BatchSeqDataset('./ds/mds_valid', ctx_size=512)

dl_debug = DataLoader(ds_train, shuffle=True, batch_size=4)

for di in tqdm.tqdm(dl_debug):
    di
    break
    pass
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor

from transformermodel import *

#%%
vocab_sz = 22
output_dim = 1
hidden_dim = 512 // 8
num_layers = 8
num_heads = 8
dropout_rate = 0.1
batch_size = 64
max_epochs = 100


# dl_train = DataLoader(ds_train, shuffle=True, batch_size=batch_size)
dl_val = DataLoader(ds_val, shuffle=False, batch_size=batch_size)




#%%
# model = TransformerModel(vocab_sz, output_dim, hidden_dim, num_layers, num_heads, dropout_rate)

model = TransformerModel.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=73-step=65120.ckpt", \
    vocab_sz=vocab_sz, output_dim=output_dim, hidden_dim=hidden_dim,\
    num_layers=num_layers, num_heads=num_heads, dropout_rate=dropout_rate)
#%%
import tqdm
model.eval()
cos = torch.cos
def comp_corr(x,y):
    x = x - x.mean()
    y = y - y.mean()
    return torch.cosine_similarity(x,y, dim=0)
lst_corr = []

for batch in tqdm.tqdm(dl_val):
        src, tgt, mask, lens = batch['c_res_id'].to(model.device), batch['y'].to(model.device), batch['mask'].to(model.device), batch['L']
        N = len(lens)
        lens = lens.view(-1,)
        src = src.transpose(0,1).contiguous()
        keep_ind = (~mask).to(torch.float)
        
        output = model(src, src_key_padding_mask = mask).transpose(0,1)[:,:,0]

        corrs = [comp_corr(output[i][:lens[i]], tgt[i][:lens[i]]).cpu().item() for i in range(N)]
        lst_corr.extend(corrs)
        
# %%
corrs = np.array(lst_corr)
# %%
np.mean(np.array(corrs))
# %%
