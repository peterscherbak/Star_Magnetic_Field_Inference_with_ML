#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 19:36:30 2026

@author: peterscherbak
"""
#%%


import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import os


_HERE = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════
# 1. LOAD STELLAR MODEL
# ══════════════════════════════════════════════════════════════

with open(os.path.join(_HERE, 'MS-1.5-young.data.GYRE')) as f:
    header = f.readline().split()
    n_pts  = int(header[0])
    M_star = float(header[1])   # g
    R_star = float(header[2])   # cm
    model  = np.array([f.readline().split() 
                       for _ in range(n_pts)], dtype=float)

G      = 6.674e-8          # CGS
r_sun = 6.957e10   # cm
Omega_0 = np.sqrt(G * M_star / R_star**3)   # rad/s, GYRE time unit



Pmin_days = 0.1
Pmax_days = 2.0

Prot_min_days = 0.5
Prot_max_days = 8

omega_scan_min =  1.0/(Pmax_days*86400)  * 2*np.pi
omega_scan_max =  1.0/(Pmin_days*86400)   * 2*np.pi



n_save_lo = 2
n_save_hi = 60
n_len     = n_save_hi - n_save_lo + 1

r    = model[:, 1]    # cm
r[0] = r[1] * 1e-10 
#M_r  = model[:, 2]    # gx
#P    = model[:, 4] * 1    # dyne/cm^2 ###GYRE INPUT IS OFF!! #IMPORTANT!
#T    = model[:, 5]    # K
rho  = model[:, 6]    # g/cm^3
N2   = model[:, 8]    # rad^2/s^2
#G1   = model[:, 9]    # Gamma_1



delta_P = 5.76346
beta_P  = 1.31765




def j1(z):
    """Spherical Bessel j1"""
    
    out =    np.sin(z)/z**2 - np.cos(z)/z
    return out

j1_delta = j1(delta_P)

def Br_shape(x_val):
    """Dimensionless Br profile (Prendergast), no Bc or cos(theta)"""
    #x_val = np.atleast_1d(np.float64(x_val))
    out =  (2.0/x_val**2) * (beta_P/(delta_P**2.0)) *    (x_val**2 - x_val * j1(delta_P*x_val)/j1_delta)
    return out



x    = r / R_star     # dimensionless radius

Br = Br_shape(x)   # dimensionless shape on model grid


#Bc_G   =  475e3 # 0 #475e3   # Gauss
#Bc_G   =  0

N_arr  = np.where(N2 > 0, np.sqrt(N2), 0.0)

# Two-part Bc sweep split at geometric midpoint — 199 unique values
mid         = np.sqrt(10e3 * 700e3)
Bc_sweep_1  = np.logspace(np.log10(10e3), np.log10(mid),   100)
Bc_sweep_2  = np.logspace(np.log10(mid),  np.log10(700e3), 100)
Bc_sweep    = np.concatenate([Bc_sweep_1, Bc_sweep_2[1:]])  # 199 unique values

# 100 log-spaced rotation periods → convert to angular velocity (199*100 = ~19 900 models)
Prot_sweep  = np.logspace(np.log10(Prot_min_days), np.log10(Prot_max_days), 100)
Omega_sweep = 2*np.pi / (Prot_sweep * 86400)   # rad/s

print(f'Bc range: {Bc_sweep.min()/1e3:.1f} – {Bc_sweep.max()/1e3:.0f} kG, {len(Bc_sweep)} pts')
print(f'Prot range: {Prot_sweep.min():.1f} – {Prot_sweep.max():.1f} days, {len(Omega_sweep)} pts')
print(f'Total models: {len(Bc_sweep) * len(Omega_sweep)}')

#%%%
#Omega_sweep = np.array([2*np.pi/(Prot_min*86400)])


n_Bc    = len(Bc_sweep)
n_Omega = len(Omega_sweep)


MAX_MODES = 150

all_rows = []

mv_arr = [-1]


# ── Load tables once outside function ─────────────────────────────────────
from scipy.interpolate import LinearNDInterpolator

TABLE_FILES = {
    -1: os.path.join(_HERE, 'l1_m-1.txt'),   # prograde
     0: os.path.join(_HERE, 'l1_m0.txt'),    # zonal
    +1: os.path.join(_HERE, 'l1_m+1.txt'),   # retrograde
}

INTERP_LAM = {}   # cache interpolators so they're built once

def load_tables(mv_list=[-1]):
    """Build and cache lambda interpolators for requested m values."""
    for mv in mv_list:
        if mv in INTERP_LAM:
            continue
        data_tab  = np.loadtxt(TABLE_FILES[mv], skiprows=1)
        a_tab     = data_tab[:, 0]
        q_tab     = data_tab[:, 1]
        lam_tab   = data_tab[:, 3]
        points_aq = np.column_stack([a_tab, q_tab])
        INTERP_LAM[mv] = {
            'interp' : LinearNDInterpolator(points_aq, lam_tab),
            'a_tab'  : a_tab,
            'q_tab'  : q_tab,
            'lam_tab': lam_tab,
        }
        print(f'Loaded m={mv:+d}: {len(a_tab)} rows, '
              f'a=[{a_tab.min():.3f},{a_tab.max():.3f}], '
              f'q=[{q_tab.min():.3f},{q_tab.max():.3f}]')

# Call once at startup
load_tables(mv_list=[-1, 0,  1])   # add 0, +1 when needed


# ── Main solver function ───────────────────────────────────────────────────
def solve_tarm_asym(Omega_rot, omB_phys, Bc_G,
                    mv_list=[-1],
                    omega_scan_min=None, omega_scan_max=None,
                    eps_g=0.0, n_scan=100):
    """
    Solve Eq. (58) of Rui+2023 for g-mode frequencies under TARM.

    Parameters
    ----------
    Omega_rot     : float, rotation rate (rad/s)
    omB_phys      : array, ωB(r) = sqrt(N·vAr/r) profile (rad/s), shape (n_pts,)
    mv_list       : list of int, azimuthal orders to solve (default prograde only)
    omega_scan_min: float, min inertial frequency to report (rad/s)
    omega_scan_max: float, max inertial frequency to report (rad/s)
    eps_g         : float, Tassoul phase offset (default 0)
    n_scan        : int, number of points in log scan per n_g (default 100)

    Returns
    -------
    results : dict keyed by mv, each value is list of mode dicts:
              {'n_g', 'omega_bar', 'omega', 'q'}
    scalars : dict of scalar diagnostics:
              {'I_buoy0', 'delta_Pi0_s', 'P_crit_days' per mv}
    """
    # Use module-level stellar structure
   # global N2, N_arr, r

    omB2 = omB_phys**2

    # ── Reference integral (B=0, Ω=0) ─────────────────────────
    sqrt_lam0  = np.sqrt(2.0)                # l(l+1) = 2 for l=1
    
    cav0   = N2 > 0
    I_buoy0 = sqrt_lam0 * trapezoid(N_arr[cav0] / r[cav0], r[cav0])

    results = {}
    scalars = {'I_buoy0': I_buoy0}

    for mv in mv_list:
        #print(mv)
        if mv not in INTERP_LAM:
            raise ValueError(f'm={mv} table not loaded — call load_tables([{mv}]) first')

        tab       = INTERP_LAM[mv]
        interp_lam = tab['interp']
        q_tab     = tab['q_tab']
        a_tab     = tab['a_tab']

        def get_lambda(a_arr, q_scalar):
           """
           Vectorised λ lookup for array of local a-values at fixed q.
           Returns nan for suppressed modes (outside convex hull of table).
           """
           pts = np.column_stack([a_arr, np.full(len(a_arr), q_scalar)])
           return interp_lam(pts)

        def buoyancy_integral(omega_bar):
            """
            I(omega_bar) = integral sqrt(lambda(a(r), q)) * N/r dr
            over the g-mode cavity (N > omega_bar).
        
            Returns nan if any part of the cavity is in the suppressed regime.
            """
            if omega_bar <= 0:
                return np.nan
        
            q_val = 2.0 * Omega_rot / omega_bar   # uniform spin parameter #IMPORTANT
            
            
            #print(omega_bar, q_val)
            q_val = min(2.0 * Omega_rot / omega_bar, q_tab.max() * 0.999)
        
            # g-mode cavity: ω̄ < N  (asymptotic formula ignores ω̄-dep of bounds,
            # but including it is more physical and costs nothing)
            cav = (N2 > 0) & (N_arr > np.abs(omega_bar))   #IMPORTANT!
            if cav.sum() < 4:
                #print('badI 2 ' + str(omega_bar))
                return np.nan
        
            a_arr   = omB2[cav] / omega_bar**2    # local a(r)
            lam_arr = get_lambda(a_arr, q_val)
            
            #print(lam_arr)
        
            if np.any(~np.isfinite(lam_arr)) or np.any(lam_arr <= 0):
                #print('badI ' + str(omega_bar))
                return np.nan                      # suppressed or out of table
        
            integrand = np.sqrt(lam_arr) * N_arr[cav] / r[cav]
            return trapezoid(integrand, r[cav])

        # ── n_g range — wide enough for all m ─────────────────
        n_at_max = I_buoy0 / (np.pi * (omega_scan_max + Omega_rot)    ) - eps_g
        n_at_min = I_buoy0 / (np.pi * max(omega_scan_min - Omega_rot,          omega_scan_min * 0.1)) - eps_g
        n_lo = max(1, int(np.floor(n_at_max)) - 2)   # small buffer
        n_hi = int(np.ceil(n_at_min)) + 2

        modes_asym = []
        
        #for n_g in range(n_save_lo, n_save_hi + 1):
        for n_g in range(n_lo, n_hi + 1):
            if not n_g == 24:
                do_nada=True
                #continue
            #print(n_g)
            phi_g = np.pi * (n_g + eps_g)
            om0   = I_buoy0 / phi_g        # zero-B, zero-Ω estimate
        
            def F(omega_bar, phi_g=phi_g):
                I = buoyancy_integral(omega_bar)
                
                if not np.isfinite(I):
                    return np.nan
                Fret = omega_bar * phi_g - I
                #print(omega_bar, I, Fret)
                return Fret
        
            # Bracket symmetrically around om0; widen if needed
            if False:
                for half_width in [0.3, 0.6, 2.0]:
                    om_lo_b = om0 * (1 - half_width)
                    om_hi_b = om0 * (1 + half_width)
                    om_lo_b = max(om_lo_b, omega_scan_min * 0.5)
                    om_hi_b = min(om_hi_b, omega_scan_max * 2.0)
                    F_lo = F(om_lo_b)
                    F_hi = F(om_hi_b)
                    if np.isfinite(F_lo) and np.isfinite(F_hi) and F_lo * F_hi < 0:
                        break
                else:
                    print(n_g)
                    continue   # couldn't bracket
            else:
                    ng_good = False
                    om_scan = np.logspace(
                    np.log10(max(om0 * 0.05, 1e-12)),
                    np.log10(om0 * 5.0),
                    n_scan)
                    
                    F_scan = np.array([F(o) for o in om_scan])
                    finite = np.isfinite(F_scan)
                    
                    if n_g == 32 and False:
                        print("***")
                        print(n_g, F_scan)
                        print(finite)
                        print(finite.sum())
                    if finite.sum() < 2:
                        print(f'all nan: n_g={n_g}, om0={om0*1e6:.2f} uHz, '
                              f'q={2*Omega_rot/om0:.2f}')
                        continue
            
                    idx_finite   = np.where(finite)[0]
                    sign_changes = idx_finite[np.where(
                        np.diff(np.sign(F_scan[idx_finite])) != 0)[0]]
            
                    if len(sign_changes) == 0:
                        do_nada=True
                        #print(f'no sign change: n_g={n_g}, om0={om0*1e6:.2f} uHz')
                        
                        continue
                    else:
                        ng_good = True
                        
                    om_lo_b = om_scan[sign_changes[0]]
                    om_hi_b = om_scan[sign_changes[0] + 1]
        
            try:
                #print('about to start brent')
                #continue
                om_bar_sol = brentq(F, om_lo_b, om_hi_b, xtol=1e-12, maxiter=200)
        
                # Convert to inertial frame: ω = ω̄ − m·Ω   (Eq. 38)
                # Consistent with shooting: omega_bar = omega + mv*Omega_rot
                om_inertial = om_bar_sol - mv * Omega_rot
                
                
                    
        
                if omega_scan_min <= om_inertial <= omega_scan_max: #FIXME
                    modes_asym.append({
                        'n_g'      : n_g,
                        'omega_bar': om_bar_sol,
                        'omega'    : om_inertial,
                        'q'        : 2.0 * Omega_rot / om_bar_sol,
                    })
                    
                    
            except Exception as e:
                print(f'  n_g={n_g}: brentq failed — {e}')
        print(f'Found {len(modes_asym)} asymptotic (TARM) modes in scan range')

        results[mv] = modes_asym
       # print('testy here')
        # ── P_crit diagnostic per m ────────────────────────────
        if Bc_G > 0:
            #print('omb here')
            cav_mask    = N2 > 0
            om_bar_crit = omB_phys[cav_mask].max() / np.sqrt(a_tab.max())
            for _ in range(20):
                q_est    = min(2.0 * Omega_rot / om_bar_crit,
                               q_tab.max() * 0.999)
                q_mask   = np.abs(q_tab - q_est) < 0.05
                a_crit_q = a_tab[q_mask].max() if q_mask.sum() > 0 else a_tab.max()
                om_new   = np.sqrt(omB2[cav_mask].max() / a_crit_q)
                if abs(om_new - om_bar_crit) / om_bar_crit < 1e-6:
                    break
                om_bar_crit = om_new
            om_crit_iner = om_bar_crit - mv * Omega_rot
            if om_crit_iner > 0:
                scalars[f'P_crit_days_m{mv:+d}'] = \
                    2*np.pi / om_crit_iner / 86400
                    
        if False:
            fig, ax = plt.subplots(figsize=(8, 4))
            oms_asym = np.array([md['omega'] for md in modes_asym])
            P_asym   = np.sort(1.0 / (oms_asym / (2*np.pi)) / 86400)   # days
            dP_asym  = np.diff(P_asym) * 24                              # hours
        
           
            sc1 = ax.scatter(P_asym[:-1], dP_asym, s=1, marker='s', #IMPORTANT
                       label=f'TARM  m={mv:+d}')
            try:
                sc2 = ax.scatter(P_asym[:MAX_MODES-1], dP_asym[:MAX_MODES], s=1, marker='s', #IMPORTANT
                           label=f'TARM  m={mv:+d}')
            except Exception as e:
                print(e)
                do_nada=True
            #P_grid  = np.linspace(0.05, 1, 100)
            #f       = interp1d(P_asym[:-1], dP_asym, bounds_error=False, fill_value=np.nan)
            #dP_grid = f(P_grid)
            
            #sc1 = ax.scatter(P_grid, dP_grid, s=1, marker='s', color='red')
        
            # Reference: flat ΔΠ from zero-B/Ω formula
            dPi0_hr = 2*np.pi**2 / I_buoy0 / 3600
            if mv == 0:
                ax.axhline(dPi0_hr, color='gray', ls='--', lw=1,
                           label=f'ΔΠ₀ = {dPi0_hr*3600:.1f} s  (B=0, Ω=0)')
        
            ax.axvline(Pmin_days, color='salmon', ls=':', lw=1)
            ax.axvline(Pmax_days, color='salmon', ls=':', lw=1)
            colora = sc1.get_facecolor()[0]
            #print(colora)
            if False:
                if mv == 0:
                    data = np.load("period_data0.npz")
    
                    Pgyre = data["P"]
                    dPgyre = data["dP"]
                    ax.scatter(Pgyre[:-1]/(86400), dPgyre/3600, s=10, color=colora, alpha=0.5)
                if mv == 1:
                    data = np.load("period_data-1.npz")
    
                    Pgyre = data["P"]
                    dPgyre = data["dP"]
                    ax.scatter(Pgyre[:-1]/(86400), dPgyre/3600, s=10, color=colora, alpha=0.5)
                if mv == -1:
                    data = np.load("period_data1.npz")
    
                    Pgyre = data["P"]
                    dPgyre = data["dP"]
                    ax.scatter(Pgyre[:-1]/(86400), dPgyre/3600, s=10, color=colora, alpha=0.5)
            ax.set_xlabel('Period (days)')
            ax.set_ylabel('ΔP (h)')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            #plt.show()
            ax.set_xlim(0, 2)
            ax.set_ylim(.8, 1.0)
            ax.set_ylim(-0.05, 1.5)
            ax.set_ylim(.3, .9)
            ax.set_ylim(0, 1)

            if Bc_G > 0:
                #print('plot here')
                P_crit_days = 2*np.pi / om_crit_iner / 86400
                ax.axvline(P_crit_days, color=colora, ls='dotted', lw=1.5,
                           label=f'a=a_crit(q) m={mv:+d} ({P_crit_days:.2f}d)')
            plt.title(f"General asym, B = {Bc_G:.3e}, Omega={Omega_rot:.3e}, modes = {len(modes_asym)}")
        if False:   # set to False when running full sweep
            fig_diag, ax_diag = plt.subplots(figsize=(7, 4))
            
            # Plot found modes
            n_found = [md['n_g'] for md in modes_asym]
            P_found = [2*np.pi / md['omega'] / 86400 for md in modes_asym]
            ax_diag.scatter(n_found, P_found, s=15, color='steelblue', 
                            label='found modes', zorder=3)
            
            # Mark the fixed save range
            ax_diag.axvline(n_save_lo, color='gray', ls='--', lw=1, 
                            label=f'n_save range [{n_save_lo},{n_save_hi}]')
            ax_diag.axvline(n_save_hi, color='gray', ls='--', lw=1)
            
            # Mark scan period window
            '''
            ax_diag.axhline(Pmin_days, color='salmon', ls=':', lw=1,
                            label=f'scan window [{Pmin_days},{Pmax_days}]d')
            ax_diag.axhline(Pmax_days, color='salmon', ls=':', lw=1)
            '''
            
            # Mark missing modes in save range in red
            if False:
                all_n_in_range = set(range(n_save_lo, n_save_hi + 1))
                missing_n = all_n_in_range - set(n_found)
                if missing_n:
                    # estimate period for missing n using zero-B/Omega asymptotic
                    for n_miss in sorted(missing_n):
                        P_est = I_buoy0 / (np.pi * n_miss) / (2*np.pi) * 86400  # rough
                        ax_diag.scatter(n_miss, P_est, s=15, color='red', 
                                       marker='x', zorder=3)
            
            ax_diag.set_xlabel('Radial order n_g')
            ax_diag.set_ylabel('Period (days)')
            ax_diag.set_title(f'Bc={Bc_G/1e3:.0f}kG  '
                              f'Prot={"inf" if Omega_rot==0 else f"{2*np.pi/Omega_rot/86400:.2f}d"}  '
                              f'mv={mv}  '
                              f'found={len(modes_asym)}')
            ax_diag.legend(fontsize=8)
            ax_diag.grid(alpha=0.3)
            plt.tight_layout()
            #plt.show()
            plt.xlim(0, 60)
            plt.ylim(0, 1)
          

    return results, scalars







for i_Bc, Bc_G in enumerate(Bc_sweep):
    # recompute B-dependent quantities inside sweep
    B0r = Bc_G * Br
    vAr    = np.abs(B0r) / np.sqrt(4*np.pi * rho)   # cm/s
    omB_phys = np.sqrt(np.sqrt(N2) * vAr / r      )          
    
    for i_Om, Omega_rot in enumerate(Omega_sweep):
        Prot_str = f'{2*np.pi/Omega_rot/86400:.2f}d' if Omega_rot > 0 else 'inf'
        print(f'[{i_Bc+1}/{n_Bc} Bc] [{i_Om+1}/{n_Omega} Om]  '
              f'Bc={Bc_G/1e3:.0f}kG  Prot={Prot_str}', flush=True)
        results, scalars = solve_tarm_asym(
            Omega_rot    = Omega_rot,
            omB_phys     = omB_phys,
            Bc_G = Bc_G,
            mv_list      = mv_arr,       # prograde only for now
            omega_scan_min = omega_scan_min,
            omega_scan_max = omega_scan_max,
        )

        modes = results[-1]   #CHANGE FOR ALL MV
        if len(modes) < 3:
            continue
        
        
        if True:  #FOR PERIOD SPACING
            oms  = np.array([md['omega'] for md in modes])
            Ps   = np.sort(2*np.pi / oms / 86400)
            dPs  = np.diff(Ps) * 24
            n   = len(dPs) 
    
            if n > MAX_MODES:
                print(f'Warning: n={n} > MAX_MODES={MAX_MODES}, truncating')
                n   = MAX_MODES
                Ps  = Ps[:n+1]
                dPs = dPs[:n]
    
            # Pad to MAX_MODES
            P_raw  = np.zeros(MAX_MODES)
            dP_raw = np.zeros(MAX_MODES)
            mask   = np.zeros(MAX_MODES)
    
            P_raw[:n]  = Ps[:n]    # periods at left edge of each spacing
            dP_raw[:n] = dPs[:n]
            mask[:n]   = 1.0
            
            all_rows.append({
                'Bc_G'       : Bc_G,
                'Omega_rot'  : Omega_rot,
                #'Prot_days'  : 2*np.pi/Omega_rot/86400 if Omega_rot > 0 else np.inf,
                #'delta_Pi0_s': scalars['delta_Pi0_s'],
                'P_crit_days': scalars.get('P_crit_days_m-1', np.nan),
                'n_modes'    : n,
                'P_raw'      : P_raw,
                'dP_raw'     : dP_raw,
                'mask'       : mask,
            })
        else:
            
            P_by_n = np.zeros(n_len)
            mask_n = np.zeros(n_len)
            n_modes = len(modes)
            
            for md in modes:
                n_g = md['n_g']
                if n_save_lo <= n_g <= n_save_hi:
                    idx = n_g - n_save_lo
                    P_by_n[idx] = 2*np.pi / md['omega'] / 86400   # days
                    mask_n[idx] = 1.0
            print(Bc_G, Omega_rot, P_by_n)
            all_rows.append({
                'Bc_G'       : Bc_G,
                'Omega_rot'  : Omega_rot,
                #'Prot_days'  : 2*np.pi/Omega_rot/86400 if Omega_rot > 0 else np.inf,
                #'delta_Pi0_s': scalars['delta_Pi0_s'],
                'P_crit_days': scalars.get('P_crit_days_m-1', np.nan),
                'n_modes'    : n_modes,
                'P_raw'      : P_by_n,
                
                'mask'       : mask_n,
            })

print(f'Generated {len(all_rows)} training examples')
print(f'Max modes seen: {max(r["n_modes"] for r in all_rows)}')
print(f'Min modes seen: {min(r["n_modes"] for r in all_rows)}')




#%%
# ── Save ──────────────────────────────────────────────────────
np.savez(os.path.join(_HERE, 'training_data.npz'),
         # parameters (labels)
         Bc          = np.array([r['Bc_G']        for r in all_rows]),
         Omega       = np.array([r['Omega_rot']    for r in all_rows]),
         #Prot        = np.array([r['Prot_days']    for r in all_rows]),
         #delta_Pi0   = np.array([r['delta_Pi0_s']  for r in all_rows]),
         P_crit      = np.array([r['P_crit_days']  for r in all_rows]),
         n_modes     = np.array([r['n_modes']       for r in all_rows]),
         # ML inputs
         P_raw       = np.array([r['P_raw']         for r in all_rows]),
         #dP_raw      = np.array([r['dP_raw']        for r in all_rows]), #FIXME
         mask        = np.array([r['mask']           for r in all_rows]),
         # metadata
         #MAX_MODES   = np.array([MAX_MODES]), #FIXME
         mv          = np.array(mv_arr),
)

print(f'Saved to training_data.npz')
print(f'Shapes: P_raw={np.array([r["P_raw"] for r in all_rows]).shape}')#', '
      #f'dP_raw={np.array([r["dP_raw"] for r in all_rows]).shape}')









