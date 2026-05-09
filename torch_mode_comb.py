#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 00:37:49 2026

@author: peterscherbak
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset#, random_split
#from torchvision import datasets
import os
import numpy as np
import matplotlib.pyplot as plt


data = np.load('training_data_precomputed.npz')


do_P_train = False
do_dP_train = not do_P_train

do_Pcut = False
Pcut_val_up = 0.5


do_3_split = True

do_unique = False


do_dropout = False
p_drop     = 0.2

torch.manual_seed(42)
np.random.seed(42)


#SEED?


def periods_to_spacings_np(P_aug_norm, Pcut=None):
    P_phys = np.where(P_aug_norm != 0, P_aug_norm * P_std + P_mean, 0.0)
    dP = np.diff(P_phys, axis=1) * 24  # hours, physical
    P_left = P_phys[:, :-1]
    valid = (P_left > 0) & (P_phys[:, 1:] > 0)
    if Pcut is not None:
        valid = valid & (P_left < Pcut)
    return np.where(valid, dP, 0.0)  # physical, unnormalized


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
    ax.set_title(f'$\Omega$ ≈ {Omega_target*1e6:.1f} μHz  (P$_{{rot}}$ ≈ {Prot_target:.1f} d) — varying B$_c$')   
    ax.set_xlim(0, Pcut_val_up)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
if False:
    
    Bc_all = data['Bc']
    Om_all = data['Omega']

    bc_sorted_all = np.argsort(Bc_all)
    om_sorted_all = np.argsort(Om_all)
    N = len(Bc_all)

    # Pick 3 Omega values: low, mid, high
    om_targets = [om_sorted_all[100], om_sorted_all[N//2], om_sorted_all[-1]]
    om_targets = [8, 60, 661]

    n_bc_show = 6
    colors = plt.cm.plasma(np.linspace(0, 1, n_bc_show))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for col, om_idx in enumerate(om_targets):
        ax       = axes[col]
        #Om_target = Om_all[om_idx]
        Om_target = om_idx * 1e-6  # convert to rad/s
        tol      = Om_target * 0.05 if Om_target > 0 else 1e-8

        # Find all examples near this Omega
        same_Om     = np.abs(Om_all - Om_target) < tol
        idx_same    = np.where(same_Om)[0]
        bc_order    = np.argsort(Bc_all[idx_same])
        idx_sorted  = idx_same[bc_order]

        # Pick n_bc_show evenly spaced across Bc range
        bc_pick = np.linspace(0, len(idx_sorted)-1, n_bc_show, dtype=int)
        #bc_pick = [bc_pick2[4]]

        for k, j in enumerate(bc_pick):
            i     = idx_sorted[j]
            P_i   = P_arr[i]
            valid = (P_i > 0) & (P_i < Pcut_val_up)
            P_v   = P_i[valid]
            dP_v  = np.diff(P_v) * 24

            ax.scatter(P_v[:-1], dP_v, s=8, color=colors[k],
                       label=f'Bc={Bc_all[i]/1e3:.0f} kG')

        Prot = 2*np.pi / Om_target / 86400 if Om_target > 0 else np.inf
        ax.set_xlabel('P (days)')
        ax.set_ylabel('ΔP (h)')
        ax.set_title(f'$\Omega$={Om_target*1e6:.1f} μHz  '
                     f'(P$_{{rot}}$={Prot:.1f} d)')
        #ax.set_xlim(0, 0.5)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        #ax.set_ylim(0.3, 0.9)

    plt.suptitle('ΔP vs P — fixed Ω, varying Bc', fontsize=10)
    plt.tight_layout()
    plt.show()




N_data = len(P_arr)
#size_data_start = len(P_arr[0])


if not do_unique:
    idx     = np.random.permutation(N_data)
    n_val     = int(0.1 * N_data) #IMPORTANT!
    
    
    if do_3_split:
        n_test  = int(0.1 * N_data)
    else:
        n_test = 0
    
    
    n_train = N_data - n_val - n_test
    
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    
    if do_3_split:
        test_idx  = idx[n_train+n_val:]
    
    else:
        test_idx = val_idx
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





print(len(train_idx), len(val_idx), len(test_idx))
print(N_data)
print(len(train_idx) + len(val_idx), len(train_idx) + len(val_idx)+len(test_idx))

y_norm_raw = np.stack([np.log10(y_arr[:, 0]), y_arr[:, 1]], axis=1)
y_mean = y_norm_raw[train_idx].mean(axis=0)
y_std  = y_norm_raw[train_idx].std(axis=0)
y_norm = (y_norm_raw - y_mean) / y_std





P_cut = Pcut_val_up if do_Pcut else np.inf

P_cut_arr = np.where((P_arr > 0) & (P_arr < P_cut), P_arr, 0.0)
cols_with_data = (P_cut_arr[train_idx] > 0).any(axis=0)
P_cut_arr = P_cut_arr[:, cols_with_data]

train_cut = P_cut_arr[train_idx]
P_mean    = train_cut[train_cut > 0].mean()
P_std     = train_cut[train_cut > 0].std()
P_norm    = np.where(P_cut_arr > 0, (P_cut_arr - P_mean) / P_std, 0.0)


P_phys    = np.where(P_norm != 0, P_norm * P_std + P_mean,0)

dP_arr   = np.diff(P_phys, axis=1) * 24  # full columns
P_left   = P_phys[:, :-1]
mask_dP  = (P_phys[:, :-1] != 0) & (P_phys[:, 1:] != 0)  
in_window = (P_left > 0) & (P_left < (Pcut_val_up if do_Pcut else np.inf)) & (mask_dP > 0)

#in_window = (P_left > 0) & (P_left < Pcut_val_up if do_Pcut else np.inf) & (mask_dP > 0)
dP_phys  = np.where(in_window, dP_arr, 0.0)
train_dP = dP_phys[train_idx]
dP_mean  = train_dP[train_dP != 0].mean()
dP_std   = train_dP[train_dP != 0].std()
dP_norm  = np.where(dP_phys != 0, (dP_phys - dP_mean) / dP_std, 0.0)






  
    
   
y = torch.tensor(y_norm, dtype=torch.float32)     

#X_full = X.numpy().copy()   # shape (N, size_data)
y_full = y_norm.copy()     # shape (N, 2)

#print(f'X mean: {X_full.mean():.3f}  std: {X_full.std():.3f}')

size_data = P_norm.shape[1] - 1 if do_dP_train else P_norm.shape[1]

#print(f'X shape: {X.shape}')





# ── Scalar tensors for on-the-fly period→spacing conversion ───────────────
P_mean_t = torch.tensor(P_mean, dtype=torch.float32)
P_std_t  = torch.tensor(P_std,  dtype=torch.float32)
if do_dP_train:
    dP_mean_t = torch.tensor(dP_mean, dtype=torch.float32)
    dP_std_t  = torch.tensor(dP_std,  dtype=torch.float32)

def prepare_batch(P_batch):
    """Apply mode dropout then convert to model input (periods or spacings)."""
    if do_dropout:
        mask    = (P_batch != 0) & (torch.rand_like(P_batch) < p_drop)
        dropped = P_batch.masked_fill(mask, 0.0)
        order   = torch.argsort((dropped == 0).float(), dim=1, stable=True)
        P_batch = torch.gather(dropped, 1, order)
    if do_dP_train:
        P_phys    = torch.where(P_batch != 0, P_batch * P_std_t + P_mean_t, torch.zeros_like(P_batch))
        dP        = torch.diff(P_phys, dim=1) * 24
        P_left    = P_phys[:, :-1]
        mask_dP   = (P_phys[:, :-1] != 0) & (P_phys[:, 1:] != 0)
        in_window = (P_left > 0) & mask_dP
        if do_Pcut:
            in_window = in_window & (P_left < Pcut_val_up)
        return torch.where(in_window, (dP - dP_mean_t) / dP_std_t, torch.zeros_like(dP))
    return P_batch





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

model = NeuralNetwork().to(device)
print(model)


#%%

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        X    = prepare_batch(X)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
   

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss = total_loss / len(dataloader)
    return train_loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X    = prepare_batch(X)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #correct += 
            
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    #correct /= size
    
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return (test_loss)
    
#%%

train_losses = []
val_losses   = []

best_val_loss = float('inf')
patience_count = 0
patience = 50
epochs = 1000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainloss = train(train_dataloader, model, loss_fn, optimizer)
    val_loss = test(val_dataloader, model, loss_fn)
    
    train_losses.append(trainloss)
    val_losses.append(val_loss)
    
    print(f'train loss: {trainloss:.6f}  val loss: {val_loss:.6f}')
    
    # early stopping + save best
    window = 10
    smoothed_val = np.mean(val_losses[-window:]) if len(val_losses) >= window else val_loss
   
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model_best.pth')
        patience_count = 0
    else:
        patience_count += 1
    
    if patience_count >= patience:        
        print(f'Early stopping at epoch {t+1}')
        break
    
    

print("Done!")



# Load best model
model.load_state_dict(torch.load('model_best.pth'))

# Plot learning curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label='train')
ax.plot(val_losses,   label='val')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#%%


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")










#%%
X_test = prepare_batch(P_norm_t[test_idx])

model.eval()
with torch.no_grad():   
    y_pred_norm = model(X_test).numpy()

y_pred = y_pred_norm * y_std + y_mean
y_true = y_full[test_idx] * y_std + y_mean

#%%





Bc_pred = 10 ** y_pred[:, 0]
Bc_true = 10 ** y_true[:, 0]
Om_pred = y_pred[:, 1]
Om_true = y_true[:, 1]


config_str = (f'P={do_P_train} dP={do_dP_train} '
              f'3split={do_3_split} '
              f'Pcut={do_Pcut} Pcut_val={Pcut_val_up}d '
              f'dropout={do_dropout} '
              f'p_drop={p_drop if do_dropout else 0}')

Prot_true = 2 * np.pi / (Om_true  * 86400) # days (assuming Om in rad/s * 86400... check units)


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(config_str, fontsize=9, color='gray')
if False:
    sc0 = axes[0].scatter(Bc_true, Bc_pred, s=2, alpha=0.5, c=Prot_true, cmap='viridis',vmin=.5, vmax=5)
else:
    axes[0].scatter(Bc_true, Bc_pred, s=5, alpha=0.5)
axes[0].plot([Bc_true.min(), Bc_true.max()], [Bc_true.min(), Bc_true.max()], 'r--')
axes[0].set_xscale('log'); axes[0].set_yscale('log')
axes[0].set_ylim(2e5, 8e5)
axes[0].set_ylim(1e4, 1e6)

#axes[0].set_xlim(2e5, 8e5)


print(max(Prot_true), min(Prot_true))
print(np.sort(Prot_true)[-9])
axes[0].set_xlabel('True Bc (G)'); axes[0].set_ylabel('Predicted Bc (G)')


#plt.colorbar(sc0, ax=axes[0], label='Prot (days)')

axes[1].scatter((Om_true), (Om_pred), s=2, alpha=0.5)
axes[1].plot([Om_true.min(), Om_true.max()], [Om_true.min(), Om_true.max()], 'r--')
axes[1].set_xlabel('True Omega'); axes[1].set_ylabel('Predicted Omega')
plt.tight_layout()
plt.show()





if True:
    
    
    Bc_true_all = 10 ** (y_full[test_idx][:, 0] * y_std[0] + y_mean[0])
    Om_true_all = y_full[test_idx][:, 1] * y_std[1] + y_mean[1]
    Bc_pred_all = 10 ** y_pred[:, 0]
    Om_pred_all = y_pred[:, 1]
    
    bc_targets_kG = [Bc_true_all.min()/1e3, 
                 np.median(Bc_true_all)/1e3, 
                 Bc_true_all.max()/1e3]  # kG
    om_targets_uHz = [8, 60, 661]  # μHz
    
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
    Om_min = Om_true_all.min() * 0.8
    Om_max = Om_true_all.max() * 1.2
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            i  = pick_indices[row][col]
    
            ax.scatter(Bc_true_all[i], Om_true_all[i], color='red',
                       s=100, zorder=5, label='true', marker='*')
            ax.scatter(Bc_pred_all[i], Om_pred_all[i], color='blue',
                       s=50, zorder=5, label='pred')
            ax.set_xscale('log')
            ax.set_xlim(Bc_min, Bc_max)
            ax.set_ylim(Om_min, Om_max)
            ax.set_xlabel('Bc (G)')
            ax.set_ylabel('Omega (rad/s)')
            ax.set_title(f'Bc={Bc_true_all[i]/1e3:.1f} kG  '
                         f'Om={Om_true_all[i]*1e6:.1f} μHz')
            ax.legend(fontsize=7)
    
    plt.suptitle(config_str, fontsize=8, color='gray')
    plt.tight_layout()
    plt.show()








