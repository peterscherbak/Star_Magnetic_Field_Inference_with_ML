# Stellar Magnetic Field Inference from G-mode Periods

A PyTorch neural network that infers internal stellar magnetic field strength (B_c) and rotation rate (Ω) from g-mode oscillation period sequences, using synthetic training data generated via the Traditional Approximation of Rotation with Magnetic fields (TARM).

## Physics background

γ Doradus (γ Dor) stars exhibit gravity (g) modes whose period spacing pattern carries information about the stellar interior. In the asymptotic regime the spacing is approximately uniform, but two effects create characteristic deviations:

- **Rotation (Ω)** imparts a slope onto the period spacing pattern, and breaks the degeneracy of modes of different azimuthal frequency m.
- **Magnetic fields (B_c)** suppress modes near a critical period P_crit where the Alfvén frequency equals the mode frequency, creating a characteristic curvature in the spacing pattern.

The TARM dispersion relation (Eq. 58 of Rui & Fuller 2023) is solved numerically for each (B_c, Ω) pair using Brent's method to locate the mode frequencies. The resulting period sequences are used as training data for a neural network that learns the inverse mapping: observed periods → (B_c, Ω).

## Repository structure

```
generate_data.py              — generate synthetic training data using the TARM asymptotic solver
torch_mode_comb.py            — train and evaluate the CNN+MDN neural network
MS-1.5-young.data.GYRE        — 1.5 M☉ main-sequence stellar model (GYRE format)
l1_m-1.txt                    — TARM eigenvalue table, l=1, m=−1 (prograde)
l1_m0.txt                     — TARM eigenvalue table, l=1, m=0  (zonal)
l1_m+1.txt                    — TARM eigenvalue table, l=1, m=+1 (retrograde)
training_data_random.npz      — precomputed training data (20 000 random (B_c, Ω) examples)
```

## Dependencies

```
python >= 3.9
numpy >= 1.23
scipy >= 1.9
matplotlib >= 3.5
torch >= 2.0
```

Install with:
```bash
pip install -r requirements.txt
```

## How to run

### Option A — use the precomputed training data (recommended)

The file `training_data_random.npz` contains ~20 000 synthetic stars with (B_c, Ω) drawn randomly (log-uniform in B_c from 10–700 kG, log-uniform in P_rot from 0.5–8 d). Run training directly:

```bash
python torch_mode_comb.py
```

Diagnostic plots will pop up interactively during and after training. To save them to disk instead, replace `plt.show()` with `plt.savefig('filename.png', dpi=400)` in the relevant lines.

### Option B — regenerate training data

```bash
python generate_data.py
```

This sweeps over a 199 × 100 grid of (B_c, Ω) values (~19 900 models), solves for g-mode frequencies, and saves a new file `training_data.npz`. To load it instead of the precomputed data, change the filename at the top of `torch_mode_comb.py`:

```python
data = np.load(os.path.join(_HERE, 'training_data.npz'))
```

**Key parameters in `generate_data.py`:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Bc_sweep` | 199 log-spaced, 10–700 kG (two equal-density halves split at geometric midpoint ~83.7 kG) | Magnetic field strengths |
| `Prot_sweep` | 100 log-spaced, 0.5–8 d | Rotation periods |
| `Pmin_days / Pmax_days` | 0.1 / 2.0 d | Frequency scan range |
| `mv_arr` | `[-1]` | Azimuthal orders (prograde only by default) |
| `MAX_MODES` | 150 | Maximum modes stored per star |

## Architecture

The model is a **1D CNN with a Mixture Density Network (MDN) head** for B_c and a direct MSE head for Ω.

### CNN backbone

Input: two-channel sequence of shape `(batch, 2, n_P)`, where channel 0 is the normalized period sequence P and channel 1 is the normalized period spacing ΔP (zero-padded).

```
Conv1d(2 → 32, k=5) → ReLU
Conv1d(32 → 64, k=5) → ReLU
Conv1d(64 → 128, k=3) → ReLU
Global average pool → (batch, 128)
Linear(128 → 64) → ReLU
```

### Output heads

- **B_c (MDN, K=5 components):** outputs mixture weights π_k, means μ_k, and log-standard-deviations log σ_k. Trained with Gaussian mixture NLL. At inference the winning component (highest π_k) gives the point prediction; σ_k of that component gives per-star uncertainty.
- **Ω (MSE):** single linear output trained with mean squared error.

### Window augmentation

The full period sequence for each star can span 0.1–2.0 d, but real Kepler/TESS photometry typically covers a 0.05–0.3 d window. To match this, `prepare_batch()` applies a random window cut to each training batch:

- A random window width w ~ Uniform(0.05, 0.3) d is drawn **per star** each batch.
- The window start is drawn uniformly in [P_min, P_max − 0.05], allowing the window to extend beyond the last observed mode (**overhang**). This exposes the model to modes near P_crit and their characteristic ΔP signature.
- Surviving modes are left-packed (zeros at the right) and passed through the CNN.

This augmentation means the model sees the same star at many different window positions over training, preventing it from relying on absolute period values.

### Test inference

At evaluation time each test star is passed through `prepare_batch` with a single random window (fixed seed for reproducibility). The MDN selects the highest-weight component as the point prediction for B_c, with σ_k of that
 component as the per-star uncertainty.

## Key flags in `torch_mode_comb.py`

| Flag | Default | Description |
|------|---------|-------------|
| `train_mode` | `'both'` | Input type: `'P'` (periods), `'dP'` (spacings), or `'both'` |
| `use_cnn` | `True` | Use CNN backbone; `False` falls back to 3-layer MLP |
| `use_mdn` | `True` | Use MDN head for B_c; `False` uses MSE |
| `n_mdn_components` | `5` | Number of Gaussian components in the MDN |
| `window_width_lo / hi` | `0.05 / 0.3` d | Per-star window width range |
| `fixed_window` | `False` | If `True`, all stars get the same fixed window width |
| `no_overhang` | `False` | If `True`, window is clamped inside the star's mode range |
| `P_max_hard_cut` | `1.0` d | Hard upper period limit applied in `prepare_batch` |
| `modeuse` | `'random'` | Window mode: `'random'`, `'full'`, `'pmin'`, or `'fixed'` |
| `do_dropout` | `False` | Random mode dropout during training (simulates missed detections, experimental) |
| `p_drop` | `0.2` | Per-mode dropout probability |
| `do_noise` | `False` | Add noise to periods during training |
| `do_sin_noise` | `False` | Sinusoidal noise (physics-gap augmentation, experimental) |
| `do_3_split` | `True` | 80/10/10 train/val/test split |
| `do_unique_2d` | `False` | Stricter split: hold out unseen B_c × Ω grid rectangle for test |
| `n_modes_max` | `100` | Cap on number of shortest modes kept per star |
| `use_mixture_sigma` | `False` | If `True`, report law-of-total-variance uncertainty across all MDN components rather than the winning component's σ (experimental) |
| `debug_plot_batch` | `False` | Plot the first batch entering `prepare_batch`, then stop |
| `debug_plot_sin_noise` | `False` | Plot clean vs noisy ΔP–P for a batch, then stop |

## Key findings

**Ω recovery** is robust across all rotation rates. The slope of the ΔP–P pattern uniquely encodes Ω, and this signal is present in any 0.05–0.3 d window.

**B_c recovery depends on whether the observation window happens to contain modes near P_crit:**

- *Intermediate - High B_c:* P_crit falls within the observable period range. Whether recovery succeeds depends on whether the window catches it. When it does, the ΔP drop is visible and B_c is recovered accurately with low MDN uncertainty σ. When the window misses P_crit, recovery fails and σ is large. The MDN uncertainty therefore acts as a reliable quality flag for high-field stars.

- *Low B_c:* P_crit is pushed to very long periods, well outside the observable window. At shorter periods, the induced curvature in ΔP is very small and the magnetic signature is too weak to detect. Recovery is poor regardless of window position, and B_c is systematically underestimated.

The MDN's per-star uncertainty σ_k acts as a quality flag: stars where the window caught P_crit generally and recovery of B_c is good have low σ, while those where the window missed P_crit and B_c recovery is poor almost always have high σ. This allows downstream users to identify confident predictions, when the true value of B_c is not known.

## References

- Rui, N. Z. et al. (2023) — Asteroseismic g-mode period spacings in strongly magnetic rotating stars
