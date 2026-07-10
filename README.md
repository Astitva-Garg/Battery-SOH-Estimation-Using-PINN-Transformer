# PINNS — Physics-Informed Neural Networks for Battery SoH Estimation

## Structure

```
PINNS/
├── common/            # Shared model, dataset, and training code (used by both weeks)
│   ├── dataset.py     # BatteryPhysicsDataset — loads binned + high-res CSVs
│   ├── model.py        # PhysicsConstrainedTransformer (+ attention recording)
│   └── train.py        # train_model() / visualize_results()
├── week3/             # Week 3: SoH estimation on real NASA battery data
│   ├── gen.py           # Downloads NASA battery dataset (kagglehub), builds CSVs
│   ├── main.py          # Trains + visualizes the transformer
│   └── data/            # battery_main_binned20.csv, battery_phys_resampled200.csv (generated)
├── week4/             # Week 4: SoH estimation on PyBaMM SPM-simulated data
│   ├── gen_spm.py       # Simulates synthetic battery cycling with PyBaMM's SPM
│   ├── spm_ocv.py       # OCV-vs-SOC curve extraction from Chen2020 params (exploration)
│   ├── main.py          # Trains + visualizes the transformer on SPM data
│   └── data/            # spm_main_binned20.csv, spm_phys_resampled200.csv (generated)
└── legacy/             # Earlier toy PINN experiments, kept for reference
    ├── exp_decay_pinn.py
    ├── shm_pinn.py
    └── water_heater_sim.py
```

## Week 3 — SoH estimation from real cycling data

1. `week3/gen.py` downloads the NASA battery dataset from Kaggle, filters
   discharge cycles, truncates at the voltage cutoff, coulomb-counts each
   cycle's capacity, and resamples every cycle onto both a fine grid (200
   points, for physics ground truth) and a coarse 20-bin grid (for the model
   input sequence). Outputs go to `week3/data/`.
2. `week3/main.py` trains a `PhysicsConstrainedTransformer`: a transformer
   encoder over the 20-bin sequence (voltage, current, temperature, dt) with
   three heads — SoH (primary), Charge Q, and Net Energy E. Q and E are
   supervised against the area-under-the-curve ground truth computed from
   the fine-resolution physics CSV, acting as physics constraints
   (`lambda_q = lambda_e = 0.3`) that regularize the SoH prediction.
3. `visualize_results` produces:
   - Predicted vs. true SoH scatter + residual histogram
   - Input feature heatmap (verifies normalization keeps dt strictly positive)
   - Mean self-attention heatmaps per transformer layer (query x key positions)

Run:
```
uv run week3/gen.py     # one-time, downloads + builds datasets
uv run week3/main.py    # train + visualize
```

## Week 4 — SoH estimation from PyBaMM SPM-simulated data

Rather than relying on the SPM purely for an OCV curve (`spm_ocv.py`, kept
as the original exploration), Week 4 uses PyBaMM's Single Particle Model to
generate a full synthetic cycling dataset in the exact same schema as
Week 3, so the same transformer and training code can be reused unchanged.

1. `week4/gen_spm.py` simulates `N_BATTERIES` synthetic cells, each with its
   own capacity-fade rate and ambient temperature. Every cycle is solved with
   `pybamm.lithium_ion.SPM()` at a randomized C-rate, using the assigned SoH
   to set that cycle's nominal capacity and discharge current. Voltage,
   current, and temperature traces are resampled into the same
   binned/fine-resolution CSV pair as Week 3. Outputs go to `week4/data/`.
2. `week4/main.py` trains the identical `PhysicsConstrainedTransformer` via
   `common/train.py` on the SPM-generated data, with the same physics
   constraints and visualizations as Week 3.

Run:
```
uv run week4/gen_spm.py   # one-time, simulates + builds datasets
uv run week4/main.py      # train + visualize
```

## Legacy

Early PINN warm-up exercises kept for reference, not part of the battery SoH
pipeline:
- `exp_decay_pinn.py` — PINN for `dy/dt + ky = 0` (exponential decay)
- `shm_pinn.py` — PINN for `d^2y/dt^2 + omega^2 y = 0` (simple harmonic motion)
- `water_heater_sim.py` — lumped-parameter water heater controller simulation
