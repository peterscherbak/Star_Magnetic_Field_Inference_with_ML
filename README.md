# Stellar Magnetic Field Inference from G-mode Periods

A PyTorch neural network that infers stellar magnetic field strength (B_c) and rotation rate (Ω) from g-mode oscillation period sequences, using synthetic training data generated via the Traditional Approximation of Rotation with Magnetic fields (TARM).

## Physics background

γ Dor stars exhibit gravity (g) modes whose period spacings carry information about internal rotation and magnetic fields. In the asymptotic regime, the period spacing pattern is approximately uniform, but:

- **Rotation** imparts a slope onto the period spacing pattern, and breaks the degeneracy of modes of different azimuthal frequency m
- **Magnetic fields** suppress modes near a critical period P_crit where the Alfvén frequency equals the mode frequency, creating a characteristic curvature in the spacing pattern

This project trains an MLP to map a sequence of observed periods → (B_c, Ω), trained on a grid of synthetic stellar models.

## Repository structure

```
generate_data.py       — generates synthetic training data using TARM asymptotic solver
torch_mode_comb.py     — trains and evaluates the neural network
MS-1.5-young.data.GYRE — 1.5 M☉ main-sequence stellar model (GYRE format)
l1_m-1.txt             — TARM eigenvalue table, l=1, m=-1 (prograde)
l1_m0.txt              — TARM eigenvalue table, l=1, m=0  (zonal)
l1_m+1.txt             — TARM eigenvalue table, l=1, m=+1 (retrograde)
training_data_precomputed.npz   — training data for the current grid of B_c and Omega
```

## Dependencies

```
python >= 3.9
numpy
scipy
matplotlib
torch >= 2.0
```

## How to run

### 1. Generate training data

```bash
python generate_data.py
```

For a single MESA model, this sweeps over a grid of B_c and Ω values, solves for g-mode frequencies using the TARM asymptotic dispersion relation (Eq. 58 of Rui et al. 2023), and saves the period arrays to `training_data.npz`.

**Key parameters in `generate_data.py`:**
- `Bc_sweep`: grid of magnetic field strengths (Gauss)
- `Prot_sweep`: grid of rotation periods (days)
- `n_save_lo`, `n_save_hi`: radial order range to save (default n_g = 2–60)

### 2. Train the network

```bash
python torch_mode_comb.py
```

**Key flags at the top of `torch_mode_comb.py`:**

| Flag | Description |
|------|-------------|
| `do_P_train` | Train on raw periods (vs. period spacings) |
| `do_dP_train` | Train on period spacings ΔP (auto-set as `not do_P_train`) |
| `do_Pcut` / `Pcut_val_up` | Truncate modes above this period (days); simulates observational limit |
| `do_dropout` | Apply on-the-fly mode dropout during training (simulates missed detections) |
| `p_drop` | Dropout probability per mode (default 0.2) |
| `do_3_split` | Use separate val and test sets (80/10/10) |
| `do_unique` | Omega-exclusive split: hold out unseen rotation values for testing |

## Model

A 3-layer MLP (input → 512 → 512 → 2) trained with MSE loss and Adam optimizer. Early stopping with patience=50 over up to 1000 epochs.

**Inputs:** normalized period sequence (or period spacings), zero-padded to fixed length  
**Outputs:** log₁₀(B_c / G) and Ω (rad/s)

The `prepare_batch()` function applies mode dropout and optional period→spacing conversion on every batch, so the DataLoader always stores normalized periods and the conversion happens in the training loop.

## Key findings

- Ω is recovered well even for unseen rotation rates — the model learns the physical relationship between period scale and rotation rather than memorizing the training grid
- For high values of B_c, B_c recovery is very good.
- B_c recovery degrades for slow rotators at low field strengths: the critical period P_crit shifts outside the observable window (P_cut), making the magnetic signature invisible
- Fast rotators at intermediate B_c: P_crit falls within the observable window and B_c is recovered

## References

- Rui, Ong & Mathis (2023) — TARM asymptotic theory; dispersion relation used in `generate_data.py`
