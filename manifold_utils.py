from __future__ import print_function
import os
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import numpy as np
from math import pi
from IPython.core.debugger import set_trace
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed, pool


def intrinsic_dim_sample_wise(X, neighb, k=5):
    
    dist, _ = neighb.kneighbors(X)
        
    dist = dist[:, 1:k+1] 
    assert np.all(dist > 0)
        
    d = np.log(dist[:,k - 1: k] / dist[:,:k-1]) 
    d = d.sum(axis=1) / (k - 2) 
    d = 1. / (d + 1e-10)
    intdim_sample = d
    
    return intdim_sample

def intrinsic_dim_scale_interval(X, neighbours_range):
    
    intdim_k = []
    
    for k in neighbours_range:

        neighb = NearestNeighbors(n_neighbors=k+1, 
                                  n_jobs=-1).fit(X)
        m = intrinsic_dim_sample_wise(X, 
                                      neighb, 
                                      k=k).mean()
        intdim_k.append(m)
    return intdim_k
 
    
def repeated(func, 
             X, 
             nb_iter=100, 
             random_state=None, 
             duplicates_thresh=0.,
             n_jobs=-1,
             **func_kw):
    '''
    The goal is to estimate intrinsic dimensionality of data, 
    the estimation of dimensionality is scale dependent
    (depending on how much you zoom into the data distribution 
    you can find different dimesionality), so they
    propose to average it over different scales, 
    the interval of the scales [k1, k2] are the only 
    parameters of the algorithm.
    '''
        
    results = []
    N = X.shape[0]
    
    results = Parallel(n_jobs=n_jobs,verbose=0)(delayed(func)(X[np.unique(np.random.choice(np.arange(N), 
                                                                                       size=N))],
                                                          **func_kw) for i in tqdm(range(nb_iter)))
    return np.array(results)