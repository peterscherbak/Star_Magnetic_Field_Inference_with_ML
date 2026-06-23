#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 00:37:49 2026

@author: peterscherbak
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
#from torchvision import datasets
import os
import numpy as np
import matplotlib.pyplot as plt


import matplotlib as mpl
mpl.rcParams['figure.dpi']= 400

_HERE = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(_HERE, 'training_data_random.npz'))



n_modes_max = 100

train_mode = 'both'   # 'P', 'dP', or 'both'

use_cnn = True
use_mask_channel = False   # add binary real/pad channel as 3rd CNN input


use_mdn           = True
n_mdn_components  = 5
assert not (use_mdn and not use_cnn), "MDN only implemented for CNN (use_cnn=True)"

#window_width      = 0.3    # width of the observation window (days)

window_width_lo = 0.05   # minimum window width (days)
window_width_hi = 0.3   # maximum window width (days)
window_width    = 0.3   # used only for 'fixed'/'pmin' modes and debug plots

fixed_window = False

if fixed_window:
    window_width_lo = window_width_hi = window_width
no_overhang = False

modeuse = "random"
#modeuse = "fixed"
#modeuse = "full"


P_max_hard_cut = None
if modeuse == 'random':
    P_max_hard_cut = 1.0
    


Pcut_val_lo       = 0.0    # lower bound for 'fixed' eval window (days)
Pcut_val_up       = 10 #Pcut_val_lo + window_width   # upper bound for 'fixed' eval window (days)

debug_plot_batch  = False  # set True to plot first batch entering prepare_batch then stop
debug_prot_targets = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3, 4.0, 5, 6.0, 8.0]  # Prot (days) of stars to visualize



#dP_with_mask = False
do_3_split = True

do_unique    = False
do_unique_2d = False   # 2D rectangular holdout: test on unseen Omega AND unseen Bc
test_frac_2d = 0.2    # fraction of unique Omega/Bc values held out for test
frac_val = 0.1 #0


do_dropout = False
p_drop     = 0.2

do_noise = False
noise_std = 0.0005

do_sin_noise         = False   # set True to add sinusoidal physics-gap noise (experimental)
#sin_noise_amp        = 0.001   # amplitude (days)
#sin_noise_freq_lo    = 1.0     # cycles/day, lower bound
#sin_noise_freq_hi    = 5.0    # cycles/day, upper bound

sin_noise_amp = 0.01  # slightly larger baseline
sin_noise_freq_lo = 1.0     # cycles/day (lambda ~ 0.14d)
sin_noise_freq_hi = 5.0    # cycles/day (lambda ~ 0.02d)
debug_plot_sin_noise = False   # plot clean vs noisy ΔP–P on first batch, then stop




early_stop = True

torch.manual_seed(42)
np.random.seed(42)


def periods_to_spacings_np(P_aug_norm, Pcut=None):
    P_phys = np.where(P_aug_norm != 0, P_aug_norm * P_std + P_mean, 0.0)
    dP = np.diff(P_phys, axis=1) * 24  # hours, physical
    P_left = P_phys[:, :-1]
    valid = (P_left > 0) & (P_phys[:, 1:] > 0)
    if Pcut is not None:
        valid = valid & (P_left < Pcut)
    return np.where(valid, dP, 0.0)  # physical, unnormalized
##%%


'''
om_t = 62.77854967626196e-6
om_t = 57.7e-6

bc_t = 54010.23990300822
idx = np.argmin(np.abs(data['Bc'] - bc_t) + np.abs(data['Omega'] - om_t)*1e6)
print(f"In may14: Bc={data['Bc'][idx]:.5f}  Om={data['Omega'][idx]*1e6:.5f} uHz")
print(f"  P={data['P_raw'][idx][data['P_raw'][idx]>0]}")

print(idx)
P_v_test   = data['P_raw'][idx]
dP_v_test  = np.diff(P_v_test) * 24
'''


#%%



# Derive Prot from Omega on the fly
P_arr = data["P_raw"]
mask_arr = data["mask"]
y_arr = np.column_stack((data['Bc'], data['Omega']))


valid = (P_arr[:, :-1] > 0) & (P_arr[:, 1:] > 0)
dP_scrch = np.diff(P_arr, axis=1)
violations = (dP_scrch < 0) & valid
print(f'Violations: {violations.sum()}')






if False:
    Prot_target  = .11  # days — change to explore different rotation rates
    Omega_target = 2*np.pi / (Prot_target * 86400)
    tol          = Omega_target * 0.05
    
    same_Om  = np.abs(data['Omega'] - Omega_target) < tol
    idx_same = np.where(same_Om)[0]
    Bc_same  = data['Bc'][idx_same]
    
    print(f'Found {len(idx_same)} examples near Prot={Prot_target}d')
    print(f'Bc range: {Bc_same.min()/1e3:.1f} – {Bc_same.max()/1e3:.1f} kG')
    
    # Pick a few spanning the Bc range
    n_show = 6
    bc_order = np.argsort(Bc_same)
    idx_same_sorted = idx_same[bc_order]
    Bc_same_sorted  = Bc_same[bc_order]
    
    # Pick evenly spaced across sorted range
    n_show   = 6
    bc_pick  = np.linspace(0, len(idx_same_sorted)-1, n_show, dtype=int)
    colors   = plt.cm.plasma(np.linspace(0, 1, n_show))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, j in enumerate(bc_pick):
        i = idx_same_sorted[j]
        P_i   = P_arr[i]
        valid = (P_i > 0) & (P_i < Pcut_val_up)
        P_v   = P_i[valid]
        dP_v  = np.diff(P_v) * 24
        ax.scatter(P_v[:-1], dP_v, s=8, color=colors[k],
                   label=f'Bc={Bc_same_sorted[bc_pick[k]]/1e3:.0f} kG')
    
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('ΔP (hours)')
    Prot = 2*np.pi / closest_om / 86400 if Om_target > 0 else np.inf
    ax.set_title(f'$\Omega$={closest_om*1e6:.2f} μHz  '
               f'(P$_{{rot}}$={Prot:.2f} d)')    
    ax.set_xlim(0, Pcut_val_up)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
if True:
    
    Bc_all = data['Bc']
    Om_all = data['Omega']

    bc_sorted_all = np.argsort(Bc_all)
    om_sorted_all = np.argsort(Om_all)
    N = len(Bc_all)

    # Pick 3 Omega values: low, mid, high
    #om_targets = [om_sorted_all[100], om_sorted_all[N//2], om_sorted_all[-1]]
    om_targets = [8, 60, 140]
    #om_targets = [60]
    om_targets = [15, 60, 130]  # μHz 8, 60, 140


    n_bc_show = 6
    colors = plt.cm.plasma(np.linspace(0, 1, n_bc_show))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for col, om_idx in enumerate(om_targets):
        ax       = axes[col]
        #Om_target = Om_all[om_idx]
        Om_target = om_idx * 1e-6  # convert to rad/s
        tol      = Om_target * 0.05 if Om_target > 0 else 1e-8

        # Find all examples near this Omega
        closest_om = Om_all[np.argmin(np.abs(Om_all - Om_target))]
        same_Om    = (Om_all == closest_om)
        idx_same   = np.where(same_Om)[0]

        bc_order    = np.argsort(Bc_all[idx_same])
        idx_sorted  = idx_same[bc_order]

        # Pick n_bc_show evenly spaced across Bc range
        bc_pick = np.linspace(0, len(idx_sorted)-1, n_bc_show, dtype=int)
        #bc_pick = [bc_pick2[4]]
        for k, j in enumerate(bc_pick):
            i     = idx_sorted[j]
            P_i   = P_arr[i]
            Pcut_val_up_test = Pcut_val_up
            Pcut_val_up_test = np.inf

            valid = (P_i > 0) & (P_i < Pcut_val_up_test)
            P_v   = P_i[valid]
            dP_v  = np.diff(P_v) * 24

            ax.scatter(P_v[:-1], dP_v, s=8, color=colors[k],
                       label=f'Bc={Bc_all[i]/1e3:.0f} kG')
            #ax.scatter(P_v_test[:-1], dP_v_test, color='black')
            #if (abs(Om_all[i] - om_t) < 1e-5): #and (abs(Bc_all[i] - bc_t)<1e-1):
                #print(idx_same[0])
                #print(idx_same[1])

            #print(Om_all[i]*1e6, str(Bc_all[i]))
            #print(P_v)

        Prot = 2*np.pi / closest_om / 86400 if Om_target > 0 else np.inf
        ax.set_xlabel('P (days)')
        ax.set_ylabel('ΔP (h)')
        ax.set_title(f'$\Omega$={closest_om*1e6:.2f} μHz  '
                     f'(P$_{{rot}}$={Prot:.2f} d)')

        ax.set_xlim(0,2)
        ax.set_ylim(0, 1)

        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        #ax.set_ylim(0.3, 0.9)

    plt.suptitle('ΔP vs P — fixed Ω, varying Bc', fontsize=10)
    plt.tight_layout()
    #plt.show()
    plt.close()




N_data = len(P_arr)
#size_data_start = len(P_arr[0])


if not do_unique and not do_unique_2d:
    idx     = np.random.permutation(N_data)
    n_val     = int(0.1 * N_data) #IMPORTANT!
    
    
    if do_3_split:
        n_test  = int(0.1 * N_data)
    else:
        n_test = 0
    
    
    n_train = N_data - n_val - n_test
    
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val] #FIXME
    
    if do_3_split:
        test_idx  = idx[n_train+n_val:]
    
    else:
        test_idx = val_idx
elif do_unique_2d:
      unique_omegas = np.unique(data['Omega'])
      unique_bcs    = np.unique(data['Bc'])
      perm_om       = np.random.permutation(len(unique_omegas))
      perm_bc       = np.random.permutation(len(unique_bcs))

      if do_3_split:
          # 2D test holdout — strictly unseen Omega AND Bc
          n_test_om   = max(1, int(test_frac_2d * len(unique_omegas)))
          n_test_bc   = max(1, int(test_frac_2d * len(unique_bcs)))
          test_omegas = set(unique_omegas[perm_om[:n_test_om]])
          test_bcs    = set(unique_bcs[perm_bc[:n_test_bc]])
          test_mask   = np.isin(data['Omega'], list(test_omegas)) & np.isin(data['Bc'], list(test_bcs))
          test_idx    = np.where(test_mask)[0]
          pool_mask   = (~np.isin(data['Omega'], list(test_omegas))) & \
                        (~np.isin(data['Bc'],    list(test_bcs)))
      else:
          pool_mask = np.ones(N_data, dtype=bool)

      pool_idx  = np.where(pool_mask)[0]
      perm_pool = np.random.permutation(len(pool_idx))
      n_val     = int(frac_val * len(pool_idx)) #FIXME
      val_idx   = pool_idx[perm_pool[:n_val]]
      train_idx = pool_idx[perm_pool[n_val:]] #FIXME

      if not do_3_split:
          test_idx = val_idx
  
      print(f'2D split: {len(train_idx)} train  {len(val_idx)} val  {len(test_idx)} test')


else:
    unique_omegas = np.unique(data['Omega'])
    perm_om       = np.random.permutation(len(unique_omegas))
    n_val_om      = max(1, int(0.1 * len(unique_omegas)))                                                                                                                                                        
                                                         
    val_omegas   = set(unique_omegas[perm_om[:n_val_om]])                                                                                                                                                        
    val_mask     = np.isin(data['Omega'], list(val_omegas))                                                                                                                                                      
                                                                                                                                                                                                                 
    if do_3_split:                                                                                                                                                                                               
        n_test_om   = max(1, int(0.1 * len(unique_omegas)))
        test_omegas = set(unique_omegas[perm_om[n_val_om:n_val_om + n_test_om]])
        test_mask   = np.isin(data['Omega'], list(test_omegas))                                                                                                                                                  
        train_mask  = ~val_mask & ~test_mask                                                                                                                                                                     
        test_idx    = np.where(test_mask)[0]                                                                                                                                                                     
    else:                                                                                                                                                                                                        
        train_mask = ~val_mask
        test_idx   = np.where(val_mask)[0]                                                                                                                                                                       
                                          
    train_idx = np.where(train_mask)[0]                                                                                                                                                                          
    val_idx   = np.where(val_mask)[0]  



'''
perm      = np.random.permutation(N_data)
train_idx = perm[n_val:]
test_idx  = perm[:n_val]
'''

print(len(train_idx), len(val_idx), len(test_idx))
print(N_data)
print(len(train_idx) + len(val_idx), len(train_idx) + len(val_idx)+len(test_idx))

y_norm_raw = np.stack([np.log10(y_arr[:, 0]), y_arr[:, 1]], axis=1)
y_mean = y_norm_raw[train_idx].mean(axis=0)
y_std  = y_norm_raw[train_idx].std(axis=0)
y_norm = (y_norm_raw - y_mean) / y_std





# No outer cut — store all modes; window is applied per-star in prepare_batch
P_cut_arr = np.where(P_arr > 0, P_arr, 0.0)
cols_with_data = (P_cut_arr[train_idx] > 0).any(axis=0)
P_cut_arr = P_cut_arr[:, cols_with_data]

if n_modes_max is not None:                                                                                                        
      P_cut_arr = P_cut_arr[:, :n_modes_max]   # keep first N (shortest periods)


train_cut = P_cut_arr[train_idx]
P_mean    = train_cut[train_cut > 0].mean()
P_std     = train_cut[train_cut > 0].std()
P_norm    = np.where(P_cut_arr > 0, (P_cut_arr - P_mean) / P_std, 0.0)

P_max_data = train_cut[train_cut > 0].max()
P_min_data = train_cut[train_cut > 0].min()
print(f'Period range (train): {P_min_data:.3f} – {P_max_data:.3f} days')

P_phys    = np.where(P_norm != 0, P_norm * P_std + P_mean, 0)

# dP stats computed from full period range (window applied per-batch at train time)
dP_arr   = np.diff(P_phys, axis=1) * 24
P_left   = P_phys[:, :-1]
mask_dP  = (P_phys[:, :-1] != 0) & (P_phys[:, 1:] != 0)
dP_phys  = np.where((P_left > 0) & (mask_dP > 0), dP_arr, 0.0)
train_dP = dP_phys[train_idx]
dP_mean  = train_dP[train_dP != 0].mean()
dP_std   = train_dP[train_dP != 0].std()
dP_norm  = np.where(dP_phys != 0, (dP_phys - dP_mean) / dP_std, 0.0)


'''
train_dP_window = dP_phys[train_idx]
mask_win = (P_phys[train_idx, :-1] > Pcut_val_lo) & (P_phys[train_idx, :-1] < Pcut_val_up)
dP_in_window = dP_phys[train_idx][mask_win]
print(f'dP_mean full={dP_mean:.4f}h  dP_mean in window={dP_in_window[dP_in_window!=0].mean():.4f}h')
'''





'''
train_P_norm = P_norm[train_idx][mask_arr[train_idx] == 1] #FIXME
print(f'Normalized train P mean: {train_P_norm.mean():.4f}  (should be ~0)')
print(f'Normalized train P std:  {train_P_norm.std():.4f}   (should be ~1)')
'''

# Then build X from periods or spacings
'''
if do_P_train:
    X = torch.tensor(P_norm, dtype=torch.float32)

elif do_dP_train:
    dP_arr   = np.diff(P_arr, axis=1) * 24  # full columns
    P_left   = P_arr[:, :-1]
    mask_dP  = (mask_arr[:, :-1] * mask_arr[:, 1:])
    in_window = (P_left > 0) & (P_left < (Pcut_val_up if do_Pcut else np.inf)) & (mask_dP > 0)

    #in_window = (P_left > 0) & (P_left < Pcut_val_up if do_Pcut else np.inf) & (mask_dP > 0)
    dP_phys  = np.where(in_window, dP_arr, 0.0)
    train_dP = dP_phys[train_idx]
    dP_mean  = train_dP[train_dP != 0].mean()
    dP_std   = train_dP[train_dP != 0].std()
    dP_norm  = np.where(dP_phys != 0, (dP_phys - dP_mean) / dP_std, 0.0)
    if not dP_with_mask:
        X = torch.tensor(dP_norm, dtype=torch.float32)
    else:
        #in_window = dP_phys != 0  # valid spacing mask
        X = torch.tensor(
            np.concatenate([dP_norm, in_window.astype(float)], axis=1),
            dtype=torch.float32)
        
        
'''
        
  
    
   
y = torch.tensor(y_norm, dtype=torch.float32)     

#X_full = X.numpy().copy()   # shape (N, size_data)
y_full = y_norm.copy()     # shape (N, 2)

#print(f'X mean: {X_full.mean():.3f}  std: {X_full.std():.3f}')

n_P = P_norm.shape[1]
size_data = {'P': n_P, 'dP': n_P - 1, 'both': 2*n_P - 1}[train_mode]
if use_mask_channel and train_mode == 'both':
    size_data += n_P
print(f'n_P={n_P}  size_data={size_data}  (train_mode={train_mode!r}  window_width={window_width}d  val_cut_test={Pcut_val_up}d)')





# ── Scalar tensors for on-the-fly period→spacing conversion ───────────────
P_mean_t = torch.tensor(P_mean, dtype=torch.float32)
P_std_t  = torch.tensor(P_std,  dtype=torch.float32)
if train_mode in ('dP', 'both'):
    dP_mean_t = torch.tensor(dP_mean, dtype=torch.float32)
    dP_std_t  = torch.tensor(dP_std,  dtype=torch.float32)

def _plot_batch_debug(P_phys_raw, P_batch_packed, mode, lo_i=None, hi_i=None, P_min_i=None, P_max_i=None, n_show=6, y_batch=None):
    """Plot full mode range vs windowed modes actually fed to the model."""
    n_show   = min(n_show, P_phys_raw.shape[0])
    ncols    = 3
    nrows    = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes     = np.array(axes).flatten()

    P_raw_np  = P_phys_raw[:n_show].numpy()
    P_pack_np = P_batch_packed[:n_show].numpy()
    P_std_val = float(P_std_t.item())
    P_mean_val= float(P_mean_t.item())
    # de-normalize packed batch back to physical days
    P_win_phys = np.where(P_pack_np != 0, P_pack_np * P_std_val + P_mean_val, 0.0)

    lo_np = lo_i[:n_show].numpy() if lo_i is not None else None
    hi_np = hi_i[:n_show].numpy() if hi_i is not None else None

    y_phys = None
    if y_batch is not None:
        y_np = y_batch[:n_show].detach().numpy()
        y_phys = y_np * y_std + y_mean   # columns: [log10(Bc), Omega rad/s]

    for k in range(n_show):
        ax = axes[k]

        # full mode range (gray)
        P_full = P_raw_np[k]; P_full = P_full[P_full > 0]
        if len(P_full) > 1:
            ax.scatter(P_full[:-1], np.diff(P_full) * 24,
                       s=8, color='lightgray', zorder=1, label='all modes')

        # window shading (random mode only)
        if lo_np is not None:
            ax.axvspan(lo_np[k], hi_np[k], alpha=0.12, color='steelblue')
            ax.axvline(lo_np[k], color='steelblue', lw=0.8, ls='--')
            ax.axvline(hi_np[k], color='steelblue', lw=0.8, ls='--')
            
        #print(P_min_i)
            
        if P_min_i is not None:
            ax.axvline(P_min_i[k], ls='--', color='red')
            ax.axvline(P_max_i[k], ls='--', color='pink')


        # windowed modes fed to model (blue)
        P_win = P_win_phys[k]; P_win = P_win[P_win > 0]
        n_win = len(P_win)
        if n_win > 1:
            ax.scatter(P_win[:-1], np.diff(P_win) * 24,
                       s=14, color='steelblue', zorder=2, label=f'{n_win} modes in window')

        if y_phys is not None:
            Bc_k  = 10 ** y_phys[k, 0]
            Om_k  = y_phys[k, 1] * 1e6   # μHz
            Prot_k = 2 * np.pi / (y_phys[k, 1] * 86400)  # days
            title = f'Bc={Bc_k/1e3:.0f} kG  Ω={Om_k:.1f} μHz  Prot={Prot_k:.2f}d  |  {n_win} modes'
        else:
            title = f'star {k}  |  {n_win} modes to model'
        ax.set_title(title, fontsize=7)
        ax.set_xlabel('P (days)', fontsize=8)
        ax.set_ylabel('ΔP (h)', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)

    for k in range(n_show, len(axes)):
        axes[k].set_visible(False)

    plt.suptitle(f'prepare_batch debug  mode={mode!r}  window_width={window_width}d', fontsize=10)
    plt.tight_layout()
    plt.show()

def _apply_sin_noise_to_packed(P_packed):
    """Sinusoidal noise on left-packed normalized periods.
    Frequency drawn uniformly per call; phase drawn independently per star."""
    #freq_i    = sin_noise_freq_lo + torch.rand(1).item() * (sin_noise_freq_hi - sin_noise_freq_lo)
    #phase_i   = torch.rand(P_packed.shape[0]) * 2 * np.pi
    P_phys    = torch.where(P_packed != 0,
                            P_packed * P_std_t + P_mean_t,
                            torch.zeros_like(P_packed))
    
    log_amp  = torch.randn(1).item() * 0.5
    amp_i    = 10**log_amp * sin_noise_amp
    
    # sum 3 sinusoids with random frequencies -> looks like irregular bumps
    n_components = 3
    delta_norm = torch.zeros_like(P_packed)
    for _ in range(n_components):
        f = sin_noise_freq_lo + torch.rand(1).item() * (sin_noise_freq_hi - sin_noise_freq_lo)
        ph = torch.rand(P_packed.shape[0]) * 2 * np.pi
        delta_norm += (amp_i / P_std_t) * torch.sin(2 * np.pi * f * P_phys + ph.unsqueeze(1))
    delta_norm /= n_components  # keep amplitude calibrated
    
    return P_packed + delta_norm * (P_packed != 0).float()

def _plot_sin_noise_debug(P_phys_raw, P_clean_pack, P_noisy_pack,
                          lo_i=None, hi_i=None, n_show=6, y_batch=None):
    """Overlay clean vs sinusoidally-noised ΔP–P for a mini-batch."""
    n_show  = min(n_show, P_phys_raw.shape[0])
    ncols   = 3
    nrows   = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes    = np.array(axes).flatten()

    P_std_val  = float(P_std_t.item())
    P_mean_val = float(P_mean_t.item())
    P_raw_np     = P_phys_raw[:n_show].numpy()
    P_clean_phys = np.where(P_clean_pack[:n_show].numpy() != 0,
                            P_clean_pack[:n_show].numpy() * P_std_val + P_mean_val, 0.0)
    P_noisy_phys = np.where(P_noisy_pack[:n_show].numpy() != 0,
                            P_noisy_pack[:n_show].numpy() * P_std_val + P_mean_val, 0.0)

    lo_np = lo_i[:n_show].numpy() if lo_i is not None else None
    hi_np = hi_i[:n_show].numpy() if lo_i is not None else None

    y_phys = None
    if y_batch is not None:
        y_np   = y_batch[:n_show].detach().numpy()
        y_phys = y_np * y_std + y_mean

    for k in range(n_show):
        ax = axes[k]

        P_full = P_raw_np[k]; P_full = P_full[P_full > 0]
        if len(P_full) > 1:
            ax.scatter(P_full[:-1], np.diff(P_full) * 24,
                       s=6, color='lightgray', zorder=1, label='all modes')

        if lo_np is not None:
            ax.axvspan(lo_np[k], hi_np[k], alpha=0.12, color='steelblue')
            ax.axvline(lo_np[k], color='steelblue', lw=0.8, ls='--')
            ax.axvline(hi_np[k], color='steelblue', lw=0.8, ls='--')

        P_c = P_clean_phys[k]; P_c = P_c[P_c > 0]
        if len(P_c) > 1:
            ax.scatter(P_c[:-1], np.diff(P_c) * 24,
                       s=16, color='steelblue', zorder=2, label='clean')

        P_n = P_noisy_phys[k]; P_n = P_n[P_n > 0]
        if len(P_n) > 1:
            ax.scatter(P_n[:-1], np.diff(P_n) * 24,
                       s=8, color='tomato', alpha=0.8, zorder=3, label='+ sin noise')

        n_c = len(P_c)
        if y_phys is not None:
            Bc_k   = 10 ** y_phys[k, 0]
            Om_k   = y_phys[k, 1] * 1e6
            Prot_k = 2 * np.pi / (y_phys[k, 1] * 86400)
            title  = f'Bc={Bc_k/1e3:.0f} kG  Ω={Om_k:.1f} μHz  Prot={Prot_k:.2f}d  |  {n_c} modes'
        else:
            title  = f'star {k}  |  {n_c} modes'
        ax.set_title(title, fontsize=7)
        ax.set_xlabel('P (days)', fontsize=8)
        ax.set_ylabel('ΔP (h)', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)

    for k in range(n_show, len(axes)):
        axes[k].set_visible(False)

    plt.suptitle(
        f'sin noise diagnostic  amp={sin_noise_amp:.4f}d  '
        f'freq=[{sin_noise_freq_lo:.1f}, {sin_noise_freq_hi:.1f}] cy/d',
        fontsize=10)
    plt.tight_layout()
    plt.show()

#%%
def prepare_batch(P_batch, mode='random', y_batch=None, do_noise_param=True):
    """Apply window cut, dropout, then build model input.

    mode: 'random' — per-star random window (training + val)
          'full'   — no cut, all modes (diagnostic upper bound)
          'pmin'   — fixed [P_min_i, P_min_i + window_width] per star
          'fixed'  — fixed [Pcut_val_lo, Pcut_val_up] absolute window
    """
    P_phys_raw = torch.where(P_batch != 0, P_batch * P_std_t + P_mean_t, torch.zeros_like(P_batch))
    lo_i = hi_i = None  # set in 'random' mode; used by debug plot
    
    P_min_i = P_max_i = None
    
    
    if fixed_window:
        window_width_i = window_width_hi
    else:
        window_width_i = window_width_lo + torch.rand(P_batch.shape[0], device=P_batch.device) * (window_width_hi - window_width_lo)
        #window_width_i = window_width_lo + torch.rand(1).item() * (window_width_hi - window_width_lo)
        
    #print(f'window_width_i = {window_width_i:.4f}d')  # temporary diagnostic

    if mode == 'random':
        big      = torch.full_like(P_phys_raw, 1e9)
        P_min_i  = torch.where(P_phys_raw > 0, P_phys_raw, big).min(dim=1).values
        P_max_i  = P_phys_raw.max(dim=1).values
        
        if no_overhang:
            P_max_eff = torch.clamp(P_max_i, max=P_max_hard_cut)
            lo_range = (P_max_eff - P_min_i - window_width_i).clamp(min=0.0)
            lo_i     = P_min_i + torch.rand(P_batch.shape[0], device=P_batch.device) * lo_range
        
        elif True:
            min_coverage = 0.05  # days — ensure at least some modes in window
            # cap lo_range so window start stays below P_max_hard_cut
            lo_range = torch.min(
                P_max_i - P_min_i - min_coverage,
                torch.full_like(P_max_i, P_max_hard_cut - min_coverage) - P_min_i,
            ).clamp(min=0.0)
            lo_i     = P_min_i + torch.rand(P_batch.shape[0], device=P_batch.device) * lo_range
        hi_i     = lo_i + window_width_i
        in_win   = (P_phys_raw > lo_i.unsqueeze(1)) & (P_phys_raw < hi_i.unsqueeze(1))  & (P_phys_raw < P_max_hard_cut)
        P_batch  = P_batch.masked_fill(~in_win, 0.0)
    elif mode == 'pmin':
        big      = torch.full_like(P_phys_raw, 1e9)
        P_min_i  = torch.where(P_phys_raw > 0, P_phys_raw, big).min(dim=1).values
        hi_i     = P_min_i + window_width
        in_win   = (P_phys_raw >= P_min_i.unsqueeze(1)) & (P_phys_raw < hi_i.unsqueeze(1))
        P_batch  = P_batch.masked_fill(~in_win, 0.0)
    elif mode == 'fixed':
        in_win  = (P_phys_raw >= Pcut_val_lo) & (P_phys_raw < Pcut_val_up)
        P_batch = P_batch.masked_fill(~in_win, 0.0)
    # mode == 'full': no masking, use all modes

    # left-pack surviving modes
    order   = torch.argsort((P_batch == 0).float(), dim=1, stable=True)
    P_batch = torch.gather(P_batch, 1, order)
    
    if do_noise and do_noise_param:
      #noise = torch.randn_like(P_batch) * noise_std / P_std
      #P_batch = P_batch + noise * (P_batch != 0).float()
      P_clean_for_debug = P_batch.clone() if debug_plot_sin_noise else None

      if do_sin_noise:
          P_batch = _apply_sin_noise_to_packed(P_batch)

      if debug_plot_sin_noise:
          _plot_sin_noise_debug(P_phys_raw, P_clean_for_debug, P_batch,
                                lo_i=lo_i, hi_i=hi_i, n_show=9, y_batch=y_batch)
          raise SystemExit('debug_plot_sin_noise: stopping after sin noise diagnostic')

    if debug_plot_batch:
        _plot_batch_debug(P_phys_raw, P_batch, mode, lo_i, hi_i, P_min_i, P_max_i, y_batch=y_batch, n_show=15)
        raise SystemExit('debug_plot_batch: stopping after first batch')

    if do_dropout:
        mask    = (P_batch != 0) & (torch.rand_like(P_batch) < p_drop)
        dropped = P_batch.masked_fill(mask, 0.0)
        order   = torch.argsort((dropped == 0).float(), dim=1, stable=True)
        P_batch = torch.gather(dropped, 1, order)

    if train_mode in ('dP', 'both'):
        P_phys  = torch.where(P_batch != 0, P_batch * P_std_t + P_mean_t, torch.zeros_like(P_batch))
        dP      = torch.diff(P_phys, dim=1) * 24
        mask_dP = (P_phys[:, :-1] != 0) & (P_phys[:, 1:] != 0)
        dP_norm = torch.where(mask_dP, (dP - dP_mean_t) / dP_std_t, torch.zeros_like(dP))
        if train_mode == 'both':
            out = torch.cat([P_batch, dP_norm], dim=1)
            if use_mask_channel:
                out = torch.cat([out, (P_batch != 0).float()], dim=1)
            return out
        return dP_norm
    return P_batch

##%%



if False:
    Prot_target  = 1.5
    Omega_target = 2*np.pi / (Prot_target * 86400)
    tol          = Omega_target * 0.1
    same_Om      = np.abs(data['Omega'] - Omega_target) < tol
    idx_same     = np.where(same_Om)[0]
    Bc_same      = data['Bc'][idx_same]

    # ── Specify your two Bc values here ───────────────────────
    Bc_val1 = 10e3    # kG
    Bc_val2 = 470e3   # kG

    # Find closest available Bc to each target
    idx1 = idx_same[np.argmin(np.abs(Bc_same - Bc_val1))]
    idx2 = idx_same[np.argmin(np.abs(Bc_same - Bc_val2))]

    print(f'Bc1 = {data["Bc"][idx1]/1e3:.1f} kG')
    print(f'Bc2 = {data["Bc"][idx2]/1e3:.1f} kG')
    print(f'Omega = {Omega_target*1e6:.3f} μHz')

    n_axis = np.arange(2, 60 + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(n_axis, P_arr[idx1], s=10, label=f'Bc={data["Bc"][idx1]/1e3:.1f} kG')
    axes[0].scatter(n_axis, P_arr[idx2], s=10, label=f'Bc={data["Bc"][idx2]/1e3:.1f} kG')
    axes[0].set_xlabel('n'); axes[0].set_ylabel('Period (days)')
    axes[0].set_title(f'Prot={Prot_target}d')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].scatter(n_axis, P_arr[idx2] - P_arr[idx1], s=10, color='tomato')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('P2 - P1 (days)')
    axes[1].set_title(f'{data["Bc"][idx2]/1e3:.0f} kG minus {data["Bc"][idx1]/1e3:.0f} kG')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    dP1 = np.diff(P_arr[idx1]) * 24   # hours, consecutive n
    dP2 = np.diff(P_arr[idx2]) * 24
    
    n_axis_dP = np.arange(2, 60+1)[:-1] + 0.5  # midpoints
    
    fig, ax = plt.subplots(figsize= (8, 4))
    ax.scatter(P_arr[idx1][:-1], dP1, s=10, label=f'Bc={data["Bc"][idx1]/1e3:.0f} kG')
    ax.scatter(P_arr[idx1][:-1], dP2, s=10, label=f'Bc={data["Bc"][idx2]/1e3:.0f} kG')
    ax.set_xlabel('n'); ax.set_ylabel('dP (h)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.show()
#%%


P_norm_t      = torch.tensor(P_norm, dtype=torch.float32)
train_dataset = TensorDataset(P_norm_t[train_idx], y[train_idx])
val_dataset   = TensorDataset(P_norm_t[val_idx],   y[val_idx])
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

for X, y in val_dataloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break



#%%
device = "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(size_data, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        n_in = 2 + int(use_mask_channel)
        self.conv = nn.Sequential(
            nn.Conv1d(n_in, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.feat = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        if use_mdn:
            self.out_bc = nn.Linear(64, n_mdn_components * 3)  # pi + mu_bc + log_sigma_bc
            self.out_om = nn.Linear(64, 1)                      # Omega MSE head
        else:
            self.out = nn.Linear(64, 2)

    def forward(self, x):
        P_part  = x[:, :n_P].unsqueeze(1)
        dP_part = F.pad(x[:, n_P:2*n_P-1].unsqueeze(1), (0, 1))
        channels = [P_part, dP_part]
        if use_mask_channel:
            channels.append(x[:, 2*n_P-1:].unsqueeze(1))
        x = torch.cat(channels, dim=1)            # (batch, 2 or 3, n_P)
        x = self.conv(x).mean(dim=-1)             # global avg pool → (batch, 128)
        feat = self.feat(x)
        if use_mdn:
            K   = n_mdn_components
            raw = self.out_bc(feat)
            return raw[:, :K], raw[:, K:2*K], raw[:, 2*K:], self.out_om(feat).squeeze(1)
        return self.out(feat)
          
model = (CNNNetwork() if use_cnn else NeuralNetwork()).to(device)
print(model)



#model = NeuralNetwork().to(device)
#print(model)


#%%

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

_LOG2PI = 0.5 * np.log(2 * np.pi)

def mdn_loss_1d(pi_logits, mu_bc, log_sigma_bc, y_bc):
    """1D MDN NLL for Bc only."""
    log_pi    = F.log_softmax(pi_logits, dim=1)              # (B, K)
    sigma     = torch.exp(log_sigma_bc).clamp(min=1e-6)      # (B, K)
    y_exp     = y_bc.unsqueeze(1).expand_as(mu_bc)           # (B, K)
    log_gauss = (-0.5 * ((y_exp - mu_bc) / sigma) ** 2
                 - torch.log(sigma) - _LOG2PI)               # (B, K)
    return -torch.logsumexp(log_pi + log_gauss, dim=1).mean()

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X    = prepare_batch(X, mode=modeuse, y_batch=y)
        if use_mdn:
            pi_logits, mu_bc, log_sig_bc, om_pred = model(X)
            loss = mdn_loss_1d(pi_logits, mu_bc, log_sig_bc, y[:, 0]) + loss_fn(om_pred, y[:, 1])
        else:
            loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{(batch+1)*len(X):>5d}/{size:>5d}]")
    return total_loss / len(dataloader)

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X    = prepare_batch(X, mode=modeuse, y_batch=y)
            if use_mdn:
                pi_logits, mu_bc, log_sig_bc, om_pred = model(X)
                test_loss += (mdn_loss_1d(pi_logits, mu_bc, log_sig_bc, y[:, 0]) + loss_fn(om_pred, y[:, 1])).item()
            else:
                test_loss += loss_fn(model(X), y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss
    
#%%

model_save = 'model_bestD.pth'

if debug_plot_batch and True:
    Om_train = y_arr[train_idx, 1]
    pick = [train_idx[np.argmin(np.abs(Om_train - 2*np.pi / (p * 86400)))]
            for p in debug_prot_targets]
    P_dbg = P_norm_t[pick]
    y_dbg = torch.tensor(y_norm[pick], dtype=torch.float32)
    torch.manual_seed(42)
    prepare_batch(P_dbg, mode=modeuse, y_batch=y_dbg)
    # ^ plots and raises SystemExit

train_losses = []
val_losses   = []

best_val_loss = float('inf')
patience_count = 0
patience = 50
epochs = 1000
#epochs = 500

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainloss = train(train_dataloader, model, loss_fn, optimizer)
    val_loss = float('nan') if len(val_idx) == 0 else test(val_dataloader, model, loss_fn)
    
    train_losses.append(trainloss)
    val_losses.append(val_loss)
    
    print(f'train loss: {trainloss:.6f}  val loss: {val_loss:.6f}')
    
    # early stopping + save best
    window = 10
    smoothed_val = np.mean(val_losses[-window:]) if len(val_losses) >= window else val_loss
   
    if val_loss < best_val_loss: #val_loss
        best_val_loss = val_loss #was val_loss
        torch.save(model.state_dict(), model_save)
        patience_count = 0
    else:
        patience_count += 1
    
    if patience_count >= patience and t>100 and early_stop and len(val_idx) > 0:
        do_nada=True
        print(f'Early stopping at epoch {t+1}') #FIXME
        break
    
    

print("Done!")
##%%
if len(val_idx) == 0:
      torch.save(model.state_dict(), model_save)


# Load best model
model.load_state_dict(torch.load(model_save, weights_only=True))

# Plot learning curve

##%%
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label='train')
ax.plot(val_losses,   label='val')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)
#ax.set_ylim(.15, .5)
#ax.set_xlim(0, 1000)

plt.tight_layout()
plt.show()



'''

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
'''









#%%
#'''

use_mixture_sigma = False
test_seed = 42
torch.manual_seed(test_seed)
X_test = prepare_batch(P_norm_t[test_idx], mode=modeuse, do_noise_param=False)

model.eval()
with torch.no_grad():
    if use_mdn:
        pi_logits, mu_bc, log_sig_bc, om_pred = model(X_test)
        best_k  = pi_logits.argmax(dim=1)
        bc_pred = mu_bc[torch.arange(len(mu_bc)), best_k]
        # mixture std via law of total variance — reflects full bimodal spread
        pi_k    = F.softmax(pi_logits, dim=1)
        sig_k   = torch.exp(log_sig_bc)
        if use_mixture_sigma:
            mu_mix  = (pi_k * mu_bc).sum(dim=1)
            var_mix = (pi_k * (sig_k**2 + mu_bc**2)).sum(dim=1) - mu_mix**2
            sigma_best = var_mix.clamp(min=0).sqrt().numpy() * y_std[0]
        else:
            sigma_best = sig_k[torch.arange(len(sig_k)), best_k].numpy() * y_std[0]


        y_pred_norm = torch.stack([bc_pred, om_pred], dim=1).numpy()
    else:
        y_pred_norm = model(X_test).numpy()
        sigma_best  = None

y_pred = y_pred_norm * y_std + y_mean
y_true = y_full[test_idx] * y_std + y_mean

#'''


#%%





Bc_pred = 10 ** y_pred[:, 0]
Bc_true = 10 ** y_true[:, 0]
Om_pred = y_pred[:, 1]
Om_true = y_true[:, 1]


if modeuse == 'full':
    win_str = 'win=full'
elif modeuse == 'fixed':
    win_str = f'win={Pcut_val_lo}–{Pcut_val_up}d'
elif modeuse == 'pmin':
    win_str = f'win=pmin+{window_width}d'
else:  # random
    win_str = (f'win={window_width}d' if fixed_window
               else f'win={window_width_lo}–{window_width_hi}d')
overhang_str = 'no_overhang' if no_overhang else 'overhang'
split_str    = ('unique2d' if do_unique_2d else 'unique_om' if do_unique else 'random')
arch_str     = ('CNN' if use_cnn else 'MLP') + ('+mask' if use_mask_channel else '')
loss_str     = f'MDN(K={n_mdn_components}) ' if use_mdn else 'MSE'
config_str   = (f'{arch_str} '
                f'{loss_str}'
                f'mode={train_mode} '
                f'{win_str} {overhang_str} '
                f'Pcuthard={P_max_hard_cut} ' 
                f'split={split_str} '
                f'dropout={do_dropout} '
                f'noise={do_noise} '
                f'n_max={n_modes_max}')



'''
config_str = (f'mode={train_mode} '
              f'3split={do_3_split} '
              #f'Pcut={do_Pcut} Pcut_val={Pcut_val_up}d '
              f'dropout={do_dropout} '
              f'p_drop={p_drop if do_dropout else 0}')
'''




Prot_true = 2 * np.pi / (Om_true  * 86400) # days (assuming Om in rad/s * 86400... check units)


prot_mask = (Prot_true >= 0.5) & (Prot_true <= 2.0)
prot_mask = True

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(config_str, fontsize=9, color='gray')
if True:
    sc0 = axes[0].scatter(Bc_true[prot_mask], Bc_pred[prot_mask], s=2, alpha=0.5,
                          c=Prot_true[prot_mask], cmap='viridis', vmin=.5, vmax=1)
                          
    #n_total = len(Bc_true)
    #idx = np.random.choice(n_total, size=int(n_total * 1), replace=False)
    #sc0 = axes[0].scatter(Bc_true[idx], Bc_pred[idx], s=2, alpha=0.5,   c=Prot_true[idx], cmap='viridis', vmin=.5, vmax=1)
else:
    axes[0].scatter(Bc_true[prot_mask], Bc_pred[prot_mask], s=5, alpha=0.5)
axes[0].plot([Bc_true[prot_mask].min(), Bc_true[prot_mask].max()],
             [Bc_true[prot_mask].min(), Bc_true[prot_mask].max()], 'r--')
axes[0].set_xscale('log'); axes[0].set_yscale('log')
axes[0].set_ylim(2e5, 8e5)
axes[0].set_ylim(1e4, 1e6)
#axes[0].set_ylim(5e4, 1e6)
#axes[0].set_xlim(5e4, 1e6)
#axes[0].set_xlim(2e5, 8e5)


print(max(Prot_true), min(Prot_true))
print(np.sort(Prot_true)[-9])
axes[0].set_xlabel('True Bc (G)'); axes[0].set_ylabel('Predicted Bc (G)')


plt.colorbar(sc0, ax=axes[0], label='Prot (days)')

axes[1].scatter(Om_true * 1e6, Om_pred * 1e6, s=2, alpha=0.5)
axes[1].plot([Om_true.min()*1e6, Om_true.max()*1e6],
             [Om_true.min()*1e6, Om_true.max()*1e6], 'r--')
axes[1].set_xlabel('True Ω (μHz)')
axes[1].set_ylabel('Predicted Ω (μHz)')
plt.tight_layout()
#plt.savefig('/Users/peterscherbak/Research/machine_learning/pics_may6_claude_code/for_github/comp_2.png', dpi=400)

plt.show()

if use_mdn and sigma_best is not None:
    sigma_logBc = sigma_best                              # already in log10(Bc) space
    Bc_err_up   = 10**(np.log10(Bc_pred) + sigma_logBc) - Bc_pred
    Bc_err_lo   = Bc_pred - 10**(np.log10(Bc_pred) - sigma_logBc)
    inf_str = win_str

    # ── Figure 1: full scatter, all stars, no errorbars, colored by sigma / Prot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(config_str + f'  [{inf_str}-window]', fontsize=9, color='gray')
    for ax, c_vals, cmap, clabel in [
        (axes[0], sigma_logBc, 'plasma',  'σ(log₁₀ Bc)'),
        (axes[1], Prot_true,   'viridis', 'Prot (days)'),
    ]:
        sc = ax.scatter(Bc_true, Bc_pred, s=3, c=c_vals, cmap=cmap, alpha=0.8,
                        vmin=(0.5 if clabel == 'Prot (days)' else None),
                        vmax=(2.0 if clabel == 'Prot (days)' else None))
        ax.plot([Bc_true.min(), Bc_true.max()],
                [Bc_true.min(), Bc_true.max()], 'r--', lw=1)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('True Bc (G)'); ax.set_ylabel('Predicted Bc (G)')
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label=clabel)
    plt.tight_layout()
    plt.show()

    # ── Figure 2: subset with errorbars, colored to match dots ───────────────────
    n_eb = 200
    rng  = np.random.default_rng(1)
    samp = rng.choice(len(Bc_true), size=min(n_eb, len(Bc_true)), replace=False)
    samp = samp[np.argsort(Bc_true[samp])]   # sort by true Bc for visual clarity

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(config_str + f'  [{inf_str}]  errorbars: 200 stars', fontsize=9, color='gray')

    for ax, c_vals, cmap, clabel in [
        (axes[0], sigma_logBc, 'plasma',  'σ(log₁₀ Bc)'),
        (axes[1], Prot_true,   'viridis', 'Prot (days)'),
    ]:
        vmin = 0.5 if clabel == 'Prot (days)' else c_vals.min()
        vmax = 2.0 if clabel == 'Prot (days)' else c_vals.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        for i in samp:
            color = cmap_obj(norm(c_vals[i]))
            ax.errorbar(Bc_true[i], Bc_pred[i],
                        yerr=[[Bc_err_lo[i]], [Bc_err_up[i]]],
                        fmt='o', ms=4, lw=0.8, color=color, alpha=0.7, capsize=0)
        ax.plot([Bc_true.min(), Bc_true.max()],
                [Bc_true.min(), Bc_true.max()], 'r--', lw=1)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('True Bc (G)'); ax.set_ylabel('Predicted Bc (G)')
        ax.grid(alpha=0.3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label=clabel)
    plt.tight_layout()
    plt.show()

    # ── sigma vs window position diagnostic (multiple widths) ────────────────
    Prot_t = 2*np.pi / (y_full[test_idx, 1] * y_std[1] + y_mean[1]) / 86400
    Bc_t   = 10 ** (y_full[test_idx, 0] * y_std[0] + y_mean[0])
    slow   = np.where(Prot_t > 0.7)[0]
    picks  = slow[np.linspace(0, len(slow)-1, 4, dtype=int)]

    sweep_widths = [0.05, 0.1, 0.2, 0.3]
    width_colors = ['C0', 'C1', 'C2', 'C3']
    n_steps = 60
    model.eval()
    fig2, axes2 = plt.subplots(2, len(picks), figsize=(14, 6))

    for col, idx in enumerate(picks):
        gi       = test_idx[idx]
        P_star   = P_norm_t[gi:gi+1]
        P_crit_v = float(data['P_crit'][gi])

        P_phys_s = torch.where(P_star != 0, P_star * P_std_t + P_mean_t, torch.zeros_like(P_star))
        P_min_s  = P_phys_s[P_phys_s > 0].min().item()

        for width, color in zip(sweep_widths, width_colors):
            lo_vals = np.linspace(P_min_s, P_max_hard_cut - width, n_steps)
            sigmas, mu_bcs = [], []

            with torch.no_grad():
                for lo in lo_vals:
                    hi     = lo + width
                    in_win = (P_phys_s > lo) & (P_phys_s < hi)
                    P_win  = P_star.masked_fill(~in_win, 0.0)
                    order  = torch.argsort((P_win == 0).float(), dim=1, stable=True)
                    P_win  = torch.gather(P_win, 1, order)
                    P_pw   = torch.where(P_win != 0, P_win * P_std_t + P_mean_t, torch.zeros_like(P_win))
                    dP     = torch.diff(P_pw, dim=1) * 24
                    mdP    = (P_pw[:, :-1] != 0) & (P_pw[:, 1:] != 0)
                    dPn    = torch.where(mdP, (dP - dP_mean_t) / dP_std_t, torch.zeros_like(dP))
                    X      = torch.cat([P_win, dPn], dim=1)
                    if use_mask_channel:
                        X  = torch.cat([X, (P_win != 0).float()], dim=1)

                    pi_l, mu_b, ls_b, _ = model(X)
                    pi   = F.softmax(pi_l, dim=1)[0].numpy()
                    sig  = torch.exp(ls_b)[0].numpy()
                    best = pi.argmax()
                    sigmas.append(sig[best] * y_std[0])
                    mu_bcs.append(10 ** (mu_b[0, best].item() * y_std[0] + y_mean[0]))

            ctr = lo_vals + width / 2
            axes2[0, col].plot(ctr, sigmas, color=color, label=f'{width:.2f}d')
            axes2[1, col].plot(ctr, np.array(mu_bcs) / 1e3, color=color, label=f'{width:.2f}d')

        if 0 < P_crit_v < P_max_hard_cut:
            axes2[0, col].axvline(P_crit_v, color='red', ls='--', lw=1, label='P_crit')
            axes2[1, col].axvline(P_crit_v, color='red', ls='--', lw=1, label='P_crit')
        axes2[1, col].axhline(Bc_t[idx] / 1e3, color='k', ls='--', lw=1, label='true Bc')

        axes2[0, col].set_xlabel('Window centre (days)')
        axes2[0, col].set_ylabel('σ(log₁₀ Bc)')
        axes2[0, col].set_title(f'Bc={Bc_t[idx]/1e3:.0f}kG  Prot={Prot_t[idx]:.2f}d', fontsize=8)
        axes2[0, col].legend(fontsize=6); axes2[0, col].grid(alpha=0.3)
        axes2[1, col].set_xlabel('Window centre (days)')
        axes2[1, col].set_ylabel('Predicted Bc (kG)')
        axes2[1, col].legend(fontsize=6); axes2[1, col].grid(alpha=0.3)

    plt.suptitle('MDN σ and Bc vs window position — multiple widths  (slow rotators)', fontsize=10)
    plt.tight_layout()
    plt.close()


if False:
    
    
    Bc_true_all = 10 ** (y_full[test_idx][:, 0] * y_std[0] + y_mean[0])
    Om_true_all = y_full[test_idx][:, 1] * y_std[1] + y_mean[1]
    Bc_pred_all = 10 ** y_pred[:, 0]
    Om_pred_all = y_pred[:, 1]
    
    bc_targets_kG = [1*Bc_true_all.min()/1e3, 
                 np.median(Bc_true_all)/1e3, 
                 Bc_true_all.max()/1e3]  # kG
    #bc_targets_kG = [100, 300, 600]
    om_targets_uHz = [15, 60, 130]  # μHz 8, 60, 140
    
    def find_pair_sep(bc_target_kG, om_target_uHz):
        bc_val = bc_target_kG * 1e3   # G
        om_val = om_target_uHz * 1e-6  # rad/s
        bc_norm = (Bc_true_all - bc_val) / Bc_true_all.std()
        om_norm = (Om_true_all - om_val) / (Om_true_all.std() * 0.01)  # tight Omega
        return np.argmin(bc_norm**2 + om_norm**2)
    
    pick_indices = [
        [find_pair_sep(bc_targets_kG[0], om_targets_uHz[0]), 
         find_pair_sep(bc_targets_kG[0], om_targets_uHz[1]), 
         find_pair_sep(bc_targets_kG[0], om_targets_uHz[2])],
        [find_pair_sep(bc_targets_kG[1], om_targets_uHz[0]), 
         find_pair_sep(bc_targets_kG[1], om_targets_uHz[1]), 
         find_pair_sep(bc_targets_kG[1], om_targets_uHz[2])],
        [find_pair_sep(bc_targets_kG[2], om_targets_uHz[0]), 
         find_pair_sep(bc_targets_kG[2], om_targets_uHz[1]), 
         find_pair_sep(bc_targets_kG[2], om_targets_uHz[2])],
    ]
    
    Bc_min = Bc_true_all.min() * 0.5
    Bc_max = Bc_true_all.max() * 2.0
    Om_min = Om_true_all.min() * 0.8 * 1e6
    Om_max = Om_true_all.max() * 1.2 * 1e6
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            i  = pick_indices[row][col]
    
            ax.scatter(Bc_true_all[i], 1e6*Om_true_all[i], color='red',
                       s=100, zorder=5, label='true', marker='*')
            ax.scatter(Bc_pred_all[i], 1e6*Om_pred_all[i], color='blue',
                       s=50, zorder=5, label='pred')
            ax.set_xscale('log')
            ax.set_xlim(Bc_min, Bc_max)
            ax.set_ylim(Om_min, Om_max)
            ax.set_xlabel('Bc (G)')
            ax.set_ylabel('Omega (μHz)')
            P_days = 2*np.pi / (Om_true_all[i] * 86400)   # Om in Hz → P in days

            ax.set_title(f'Bc={Bc_true_all[i]/1e3:.1f} kG  '
                         f'Om={Om_true_all[i]*1e6:.1f} μHz'
                         f'P={P_days:.2f} d')
            #ax.set_title(f'P={P_days:.2f} d  Bc={Bc_true_all[i]/1e3:.1f} kG')
            ax.legend(fontsize=7)
    
    plt.suptitle(config_str, fontsize=8, color='gray')
    plt.tight_layout()
    #plt.savefig('/Users/peterscherbak/Research/machine_learning/pics_may6_claude_code/for_github/scatter_2.png', dpi=400)
    plt.show()










#%%

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(Bc_true, Bc_pred, s=2, alpha=0.5, c=Om_true, cmap='viridis')
ax.plot([Bc_true.min(), Bc_true.max()], [Bc_true.min(), Bc_true.max()], 'r--')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('True Bc (G)'); ax.set_ylabel('Predicted Bc (G)')
plt.colorbar(sc, ax=ax, label='Omega')
plt.tight_layout()
plt.show()


#%%
#OTHER


'''

#%%

model.load_state_dict(torch.load("model.pth", weights_only=True))

#%%
X_all = torch.tensor(P_norm, dtype=torch.float32)


model.eval()
with torch.no_grad():
    y_pred_norm = model(X_all).numpy()


y_pred = y_pred_norm * y_std + y_mean
y_true = y_norm * y_std + y_mean 

#%%
'''

