"""Week 4: generate a synthetic battery-cycling dataset using PyBaMM's Single
Particle Model (SPM), in the same schema as Week 3's NASA-derived dataset.

Instead of real measurements, each "cycle" is a physics simulation:
  - A synthetic battery has a nominal capacity that fades over cycles
    (linear fade + noise, following the SoH we assign as ground truth).
  - Each cycle discharges at a randomized C-rate and ambient temperature.
  - PyBaMM solves the SPM electrochemical equations and returns terminal
    voltage / current / temperature traces, which are then binned/resampled
    exactly like week3/gen.py so the same BatteryPhysicsDataset and
    PhysicsConstrainedTransformer can be reused unchanged.

This lets us train the same physics-constrained transformer on
physics-simulated ground truth instead of noisy real sensor data.
"""

import os

import numpy as np
import pandas as pd
import pybamm
from tqdm import tqdm

# --- CONFIGURATION ---
N_BATTERIES = 8
N_CYCLES = 40          # cycles per battery
BINS = 20              # matches week3 (SEQ_LEN)
M = 200                # physics resolution (fine resample)
NOMINAL_AH = 2.0        # reference capacity used to define SoH = 100 * cap / NOMINAL_AH
T_EVAL_S = 3600 * 6     # generous upper bound; SPM stops early at voltage cutoff

rng = np.random.default_rng(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

base_model = pybamm.lithium_ion.SPM()
base_param = pybamm.ParameterValues("Chen2020")

main_rows = []
phys_rows = []

for b in tqdm(range(N_BATTERIES), desc="Batteries"):
    battery_id = f"SPM{b:03d}"

    # Per-battery degradation profile: capacity fades roughly linearly with
    # cycle number, at a battery-specific rate, plus a little noise.
    fade_rate = rng.uniform(0.25, 0.65)   # % SoH lost per cycle
    noise_std = 0.4                        # % SoH noise
    ambient_c = rng.uniform(15.0, 35.0)   # this battery's operating temperature (deg C)

    for cyc in range(1, N_CYCLES + 1):
        soh_value = 100.0 - fade_rate * cyc + rng.normal(0, noise_std)
        soh_value = float(np.clip(soh_value, 50.0, 100.0))
        capacity = NOMINAL_AH * soh_value / 100.0

        # Randomize discharge C-rate and ambient temperature per cycle for
        # feature diversity (mirrors natural variability in real cycling data).
        c_rate = rng.uniform(0.6, 1.4)
        current = c_rate * capacity
        temp_k = 273.15 + ambient_c + rng.normal(0, 1.0)

        param = base_param.copy()
        param["Nominal cell capacity [A.h]"] = capacity
        param["Current function [A]"] = current
        param["Ambient temperature [K]"] = temp_k

        sim = pybamm.Simulation(base_model, parameter_values=param)
        try:
            sol = sim.solve([0, T_EVAL_S], initial_soc=1.0)
        except pybamm.SolverError:
            continue

        t = sol["Time [s]"].data
        if len(t) < 5 or t[-1] <= 0:
            continue

        V = sol["Terminal voltage [V]"].data
        I = sol["Current [A]"].data
        Tc = sol["X-averaged cell temperature [K]"].data - 273.15  # to Celsius

        # Resample onto a uniform fine grid of M points (like week3/gen.py)
        t0, t1 = float(t[0]), float(t[-1])
        t_grid = np.linspace(t0, t1, M)
        dt_hr = np.diff(t_grid, prepend=t_grid[0]) / 3600.0

        Vg = np.interp(t_grid, t, V)
        Ig = np.interp(t_grid, t, I)
        Tg = np.interp(t_grid, t, Tc)

        # PHYSICS DATASET (fine-resolution, not binned)
        for j in range(M):
            phys_rows.append({
                "battery_id": battery_id,
                "cycle_number": cyc,
                "t_idx": j,
                "Time_s": t_grid[j] - t0,
                "dt_hr": dt_hr[j],
                "Voltage_measured": Vg[j],
                "Current_measured": Ig[j],
                "Temperature_measured": Tg[j],
                "Capacity_Ah": capacity,
                "SoH": soh_value,
                "t_end_s": (t1 - t0),
            })

        # MAIN DATASET (BINS bins from the resampled series)
        rs = pd.DataFrame({
            "Time_s": t_grid - t0,
            "dt_hr": dt_hr,
            "Voltage_measured": Vg,
            "Current_measured": Ig,
            "Temperature_measured": Tg,
        })

        chunks = np.array_split(rs, BINS)
        if len(chunks) != BINS or any(c.empty for c in chunks):
            continue

        for b_idx, c in enumerate(chunks):
            main_rows.append({
                "battery_id": battery_id,
                "cycle_number": cyc,
                "bin_idx": b_idx,
                "Time_s": c["Time_s"].mean(),
                "dt_hr": c["dt_hr"].sum(),
                "Voltage_measured": c["Voltage_measured"].mean(),
                "Current_measured": c["Current_measured"].mean(),
                "Temperature_measured": c["Temperature_measured"].mean(),
                "Capacity_Ah": capacity,
                "SoH": soh_value,
                "t_end_s": (t1 - t0),
            })

main_df = pd.DataFrame(main_rows)
phys_df = pd.DataFrame(phys_rows)

main_df.to_csv(os.path.join(DATA_DIR, "spm_main_binned20.csv"), index=False)
phys_df.to_csv(os.path.join(DATA_DIR, "spm_phys_resampled200.csv"), index=False)

print("Saved:", main_df.shape, phys_df.shape)
