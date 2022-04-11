from __future__ import print_function
import os
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import numpy as np
from math import pi
from IPython.core.debugger import set_trace
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# sample data with a given fraction via binomial mask
def sample_data(data, fraction):
    mask = np.random.binomial(1, fraction, data.shape[0]).astype(bool)
    return data[mask]
 
def intrinsic_dim_sample_wise(X, k=5, neighb=None):
    if neighb is None:
        neighb = NearestNeighbors(n_neighbors=k+1).fit(X)
    dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1:]
    dist = dist[:, 0:k]
    assert dist.shape == (X.shape[0], k)
    try:
        assert np.all(dist > 0)
    except:
        set_trace()
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k-1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    return intdim_sample

def intrinsic_dim_scale_interval(X, k1=10, k2=20, step=5):
    X = pd.DataFrame(X).drop_duplicates() # remove duplicates in case you use bootstrapping
    
    #########################
    # hack to drop all dubl #
    #########################
    pdist = pairwise_distances(X.values)
    pdist = (pdist == 0)
    pdist[np.triu(pdist)] = False
    drop_mask = pdist.sum(1).astype(bool)
    X = X.drop(index=X.index[drop_mask]).values
    ###############################
    
    intdim_k = []
    
    for k in range(k1, k2 + 1, step):
#         print("calculating neighbours for: " + str(k))
        neighb = NearestNeighbors(n_neighbors=k+1, n_jobs=32).fit(X)
#         print("done")
        m = intrinsic_dim_sample_wise(X, k, neighb).mean()
        intdim_k.append(m)
    return intdim_k
 
def repeated(func, X, nb_iter=100, random_state=None, mode='bootstrap', **func_kw):
    '''
    The goal is to estimate intrinsic dimensionality of data, the estimation of dimensionality is scale dependent
    (depending on how much you zoom into the data distribution you can find different dimesionality), so they
    propose to average it over different scales, the interval of the scales [k1, k2] are the only parameters of the algorithm.
    '''
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    results = []
    
    iters = range(nb_iter) 
    for i in tqdm_notebook(iters):
        if mode == 'bootstrap':
            Xr = X[rng.randint(0, nb_examples, size=nb_examples)]
        elif mode == 'shuffle':
            ind = np.arange(nb_examples)
            rng.shuffle(ind)
            Xr = X[ind]
        elif mode == 'same':
            Xr = X
        else:
            raise ValueError('unknown mode : {}'.format(mode))
        results.append(func(Xr, **func_kw))
    return results