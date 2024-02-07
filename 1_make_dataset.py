#%%
import numpy as np
X = np.load('x_61046.npy')
y = np.load('y_61046.npy')
#%%
idx_train = np.load('x_train_35_pri.npy')
idx_va = np.load('x_valid_35_pri.npy')
idx_te = np.load('x_test_35_pri.npy')


# %%
seqlen = X[:, -1, -1].astype(np.int32)
coord = X[:, :, [24,25,26]]
chi = X[:,:,27].astype(np.uint8)
residual_id = X[:, :, :21].argmax(axis=-1).astype(np.uint8)
ss_id = X[:,:,21:24].argmax(axis=-1).astype(np.uint8)
# %%
onehot_feat_chk1 = X[:, :, :21].sum(axis=-1).sum(axis=-1)
missing_idx = np.nonzero(onehot_feat_chk1==0)[0]
np.save('broken_id_orig.npy', missing_idx)
# %%

# %%
from shutil import rmtree
from uuid import uuid4
from streaming import MDSWriter
import tqdm 
out_root = './ds/mds_train/'
idxs = idx_train.reshape((-1,))

# out_root = './ds/mds_valid/'
# idxs = idx_va.reshape((-1,))

# out_root = './ds/mds_test/'
# idxs = idx_te.reshape((-1,))

columns = {'L': 'int' \
    ,'c_res_id' : 'pkl' \
    ,'c_ss_id' : 'pkl' \
    ,'c_coord' : 'pkl' \
    ,'c_chi' : 'pkl' \
    ,'id_orig': 'int'
    , 'y': 'pkl'
}

# Compression algorithm name
compression = None

ignore_idx = missing_idx
rmtree(out_root, ignore_errors=True)
with MDSWriter(out=out_root, columns=columns, compression=compression) as out:
    for ii in tqdm.tqdm(idxs):
        if ii in ignore_idx:
            continue
        c_L = seqlen[ii]
        ret = {'L': int(c_L) \
            ,'c_res_id' :residual_id[ii, :c_L] \
            ,'c_ss_id' : ss_id[ii, :c_L] \
            ,'c_coord' : coord[ii, :c_L] \
            ,'c_chi' : chi[ii, :c_L] \
            ,'id_orig': int(ii) \
            ,'y': y[ii, :c_L, 0]
        }
        out.write(ret)


# %%
