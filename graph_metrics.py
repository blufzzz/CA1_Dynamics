import numpy as np
import torch
from scipy import sparse

def get_first_k_eig(w,v,k=5):
    w_argsort = np.argsort(w)
    k_index = w_argsort[:k]
    k_vector = v[:,k_index].real
    return k_vector

def compute_norm_laplacian(A):
    
    # Laplacian
    d = A.sum(0)
    D = np.diag(d)
    L = D - A
    
    D_sqrt_diag = np.sqrt(d)
    D_sqrt_diag[D_sqrt_diag>0] = 1./D_sqrt_diag[D_sqrt_diag>0]
    D_sqrt_diag = np.diag(D_sqrt_diag)
    L = D_sqrt_diag@(L@D_sqrt_diag)
    
    return L


def compute_norm_laplacian_eig(A, k=None, full=False):
    
    L = compute_norm_laplacian(A)
    if full:
        w,v = np.linalg.eig(L) # full decomposition ~O(n^2.5)
    else:
        L_sparse = sparse.csr_matrix(A)
        w,v = sparse.linalg.eigs(L_sparse, k, which='SM')
    
    return w.real,v.real


def laplacian_keigv_similarity(p1, 
                               p2, 
                               neurons_selected=None, 
                               return_nonsingle=False, 
                               full_decomposition=False,
                               k=15):
    
    if isinstance(p1,str):
        A1 = np.load(p1)
        A2 = np.load(p2)
    else:
        A1=p1
        A2=p2
        
    if neurons_selected is not None:
        A1 = A1[neurons_selected][:,neurons_selected]
        A2 = A2[neurons_selected][:,neurons_selected]
    
    N = A1.shape[0]
    non_single_nodes = np.arange(N)[(A1.sum(0)>0)*(A2.sum(0)>0)]

    if len(non_single_nodes) >= 3:

        A1_n = A1[non_single_nodes][:,non_single_nodes]
        A2_n = A2[non_single_nodes][:,non_single_nodes]
        
        d1 = compute_norm_laplacian_eig(A1_n, k, full=full_decomposition)
        d2 = compute_norm_laplacian_eig(A2_n, k, full=full_decomposition)
        
        if full_decomposition:
            v1 = get_first_k_eig(*d1, k=k)
            v2 = get_first_k_eig(*d2, k=k)
        
        n = v1.shape[0]
        d = np.linalg.norm((np.ones((n,n)) - v1@v1.T)@v2@v2.T)

        return (d, non_single_nodes) if return_nonsingle else d
    else:
        return (np.nan, non_single_nodes) if return_nonsingle else np.nan


def l2_distance(p1,p2, neurons_selected=None, use_nonsinge_nodes=False):

    if isinstance(p1,str):
        s = np.load(p1)
        p = np.load(p2)
    else:
        s=p1
        p=p2
    
    if neurons_selected is not None:
        s = s[neurons_selected][:,neurons_selected]
        p = p[neurons_selected][:,neurons_selected]
    
    N = s.shape[0]
    non_single_nodes = np.arange(N)
    if use_nonsinge_nodes:
        non_single_nodes = non_single_nodes[(s.sum(0)>0)*(p.sum(0)>0)]
    
    if len(non_single_nodes) >= 2:
        s = s[non_single_nodes][:,non_single_nodes]
        p = p[non_single_nodes][:,non_single_nodes]
        m = np.linalg.norm(s - p) / s.shape[0]
    else:
        m = np.nan
        
    return m