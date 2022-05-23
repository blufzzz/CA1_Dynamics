import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal
import math
from scipy.stats import kstest
import copy
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


TRIMS = {
    1:{'22':[2000,10], # done 
       '23':[60,200],  # done
       '24':[2500,1600],# done
       '25':[50,800]}, # done
    2:{'22':[100,20], # done
       '23':[10,1], # done
       '24':[50,900], # done
       '25':[500,800]}, # done
    3:{'22':[100,100], # done
       '23':[50,800], # done 
       '24':[50,1000], # done 
       '25':[50,600]} # done
}

def calculate_position_variables(coords):
    
    # x, y
    coords -= coords.mean(0)[None,:]
    coords_ = MinMaxScaler((-1,1)).fit_transform(coords)

    # angle
    phi = np.arctan2(coords_[:,1], coords_[:,0])
    phi[phi < 0] = 2*np.pi + phi[phi < 0]
    phi_ = MinMaxScaler().fit_transform(phi[:,None]).flatten()

    # speed sign
    dphi = np.diff(phi, prepend=phi[0])
    jump_mask = np.abs(dphi) > 6 # jump through breakpoint
    dphi[jump_mask] = -1 * np.sign(dphi[jump_mask]) * np.abs(dphi[jump_mask] - 2*np.pi)
    circle_sign = np.sign(dphi)

    # speed
    shift = np.diff(coords_, prepend=[coords_[0]], axis=0)
    speed = np.sqrt((shift**2).sum(1)) * circle_sign
    speed_ = MinMaxScaler((-1,1)).fit_transform(speed[:,None]).flatten()

    # acceleration
    acceleration = np.diff(speed, prepend=speed[0])
    acceleration_ = MinMaxScaler((0,1)).fit_transform(acceleration[:,None]).flatten()
    
    return coords_, phi_, speed_, acceleration_



def load_dataset(root, mouse=22, day=1, track='Circle', trims=None):

    '''
    Loading dataset of cells activity for 
    mouse: int [22,23,24,25]
    day: [1,2,3]
    track: ['Circle', 'Holy']
    trim0: trim values from the beginning
    trim1: trim values from the end
    '''

    ###################
    # NEURAL ACTIVITY #
    ###################
    calcium_df = pd.read_csv(os.path.join(root,f"{track}/data/CA1_{mouse}_{day}D_initial_data.csv"), 
                             index_col=0)
    spikes_df = pd.read_csv(os.path.join(root,f"{track}/spikes/CA1_{mouse}_{day}D_initial_data_spikes.csv"), 
                            index_col=0)
    rears_events = pd.read_csv(os.path.join(root,f'CA1_22-25_rears/CA1_{mouse}_{day}D_rears_from_npz.csv'), 
                               index_col=0)

    if trims is not None:
        trim0, trim1 = trims
    else:
        trim0, trim1 = 1,1
    
    cadata = calcium_df.iloc[:,7:][trim0:-trim1].values # [T, n_neurons]
    spdata = spikes_df.iloc[:,1:][trim0:-trim1].values # [T, n_neurons]
    time = calcium_df['time_s'][trim0:-trim1].values # 21 FPS
    rears_events = rears_events[trim0:-trim1]

    rear_times = rears_events['time_s'].values
    rears_indicators = rears_events['rear_1_outward_-1_inward'].values
    
    # at least one spike per timeseries
    cells_with_spikes = np.sum(spdata, axis=0) > 1.

    cadata = cadata[:,cells_with_spikes]
    spdata = spdata[:,cells_with_spikes]

    T,N = spdata.shape
    neurons = np.arange(N)

    ###########
    # TARGETS #
    ###########
    coords = calcium_df[['x','y']][trim0:-trim1].values # [T,2]
    coords_, phi_, speed_, acceleration_ = calculate_position_variables(coords)
    
    # rears indicators
    rears_indicators_abs = np.pad(np.abs(rears_indicators), pad_width=(T - rears_indicators.shape[0])//2)

    targets = {
        'x': coords_[:,0],
        'y': coords_[:,1],
        'v': speed_,
        'a': acceleration_,
        'phi': phi_
    }
    
    data = {'cadata': cadata,
            'spdata': spdata,
            'rears_events':rears_events,
            'rear_times':rear_times,
            'rears_indicators':rears_indicators,
            'cells_with_spikes':cells_with_spikes,
            'time':time
           }

    return data, targets


def plot_coords_on_circle(coords, ax=None):
    angle_range = np.linspace(0, 2*np.pi, 1000)
    ax.scatter(np.cos(angle_range), np.sin(angle_range),color='grey', alpha=0.1, marker='.')
    ax.scatter(coords[:,0], coords[:,1], color='red', alpha=0.6, marker='.')    

def coords2phi(coords):
    phi = np.arctan2(coords[:,1], coords[:,0])
    phi[phi < 0] = 2*np.pi + phi[phi < 0]
    return phi

def calc_spike_similarity_cuda(sp_torch_batch, 
                               spdata_torch, 
                               save=True, 
                               corr_dir=None, 
                               i=None, 
                               timewise=True):
    
    dt = spdata_torch.shape[-1]
    kernel_size = sp_torch_batch.shape[-1]
    result_torch = torch.conv1d(input=spdata_torch.unsqueeze(1), weight=sp_torch_batch, padding=kernel_size//2).squeeze(1)
    
    if timewise:
        result_torch = result_torch - result_torch.mean(1).unsqueeze(1)
        norms = torch.norm(result_torch, p=2, dim=1).unsqueeze(1)
        sim_map = result_torch@result_torch.transpose(0,1) / (norms@norms.transpose(0,1) + 1e-10)
    else:
        result_torch = result_torch - result_torch.mean(0).unsqueeze(0)
        norms = torch.norm(result_torch, p=2, dim=0).unsqueeze(1)
        sim_map = result_torch.transpose(0,1)@result_torch / (norms@norms.transpose(0,1) + 1e-10)
        
    sm = np.abs(sim_map.detach().cpu().numpy())
    
    # delete diagonal
    sm[np.diag_indices_from(sm)] = 0
    
    sm[sm > 1] = 1
    # save
    if save:
        path = os.path.join(corr_dir, str(i))
        np.save(path, sm)
        return path
    else:
        return sm

    
def place_field_correlation(spikes_bool, coordinates, pf_dir='', i='', bins=25, save=False):
    path = os.path.join(pf_dir, str(i))
    if save and os.path.exists(path):
        return None
    else:
        neuron_pfs = []
        for neuron_mask in spikes_bool:
            coordinates_subsample = coordinates[neuron_mask]
            if len(coordinates_subsample) > 0:
                H,_,_ = np.histogram2d(coordinates_subsample[:,0], 
                                       coordinates_subsample[:,1], 
                                       bins=bins, 
                                       range=[[-1, 1], [-1, 1]], 
                                       density=True)
                neuron_pfs.append(H.flatten()/H.sum())
            else:
                H = np.zeros((bins**2,))
                neuron_pfs.append(H)

        neuron_pfs = np.array(neuron_pfs)
        norms = np.linalg.norm(neuron_pfs, axis=1, keepdims=True)
        S = neuron_pfs @ neuron_pfs.T / (norms @ norms.T + 1e-10)
        
        S[np.diag_indices_from(S)] = 0

        if save:
            np.save(path, S)
            return path
        else:
            return S
    
def spike_form(t):
    return (1-np.exp(-t/T_RISE))*np.exp(-t/T_OFF)

def kth_diag_indices(rows, cols, k):
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols