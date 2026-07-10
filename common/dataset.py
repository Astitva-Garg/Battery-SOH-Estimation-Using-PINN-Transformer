"""Shared dataset definition for physics-constrained battery SoH training.

Consumes two CSVs with a common schema:
  - main_file: binned sequence data (one row per bin, `SEQ_LEN` bins per cycle)
    with columns: battery_id, cycle_number, Voltage_measured, Current_measured,
    Temperature_measured, dt_hr
  - phys_file: high-resolution resampled data (one row per fine time step) with
    columns: battery_id, cycle_number, Voltage_measured, Current_measured,
    dt_hr, SoH

Both Week 3 (real NASA battery data) and Week 4 (synthetic PyBaMM SPM data)
produce CSVs in this schema, so this dataset class is reused unchanged.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURE_COLS = ["Voltage_measured", "Current_measured", "Temperature_measured", "dt_hr"]


class BatteryPhysicsDataset(Dataset):
    def __init__(self, main_file, phys_file, seq_len=20):
        self.seq_len = seq_len

        # Load datasets
        self.main_df = pd.read_csv(main_file)
        self.phys_df = pd.read_csv(phys_file)

        # --- Precompute Physics Constraints (Ground Truth) ---
        print("Computing physics ground truth from high-res data...")
        self.phys_df["power"] = self.phys_df["Voltage_measured"] * self.phys_df["Current_measured"]
        self.phys_df["dq"] = self.phys_df["Current_measured"] * self.phys_df["dt_hr"]
        self.phys_df["de"] = self.phys_df["power"] * self.phys_df["dt_hr"]

        # Group by cycle to get scalar totals
        stats = self.phys_df.groupby(["battery_id", "cycle_number"]).agg({
            "dq": "sum",   # Total Charge (Ah)
            "de": "sum",   # Total Energy (Wh)
            "SoH": "first",  # SoH is constant per cycle
        }).reset_index()

        # Rename for clarity
        stats.rename(columns={"dq": "true_q", "de": "true_e", "SoH": "true_soh"}, inplace=True)
        # Q and E should be positive magnitude for capacity/energy
        stats["true_q"] = stats["true_q"].abs()
        stats["true_e"] = stats["true_e"].abs()

        # --- PHYSICS-PRESERVING NORMALIZATION ---
        # We manually scale features to preserve the physical "Zero" point.
        # StandardScaler would shift 'dt' to negative values, breaking integration.

        # 1. Voltage: Normalize 0-5V range roughly to 0-1
        self.main_df["Voltage_measured"] = self.main_df["Voltage_measured"] / 5.0

        # 2. Current: Normalize by max absolute value.
        # Crucial: Preserves 0.0 Amps as 0.0 (StandardScaler would shift this).
        max_current = self.main_df["Current_measured"].abs().max()
        if max_current > 0:
            self.main_df["Current_measured"] = self.main_df["Current_measured"] / max_current

        # 3. Temperature: Center around 20C, scale by 40 (range approx -0.5 to 1.5)
        self.main_df["Temperature_measured"] = (self.main_df["Temperature_measured"] - 20.0) / 40.0

        # 4. Time steps (dt): MUST BE STRICTLY POSITIVE.
        max_dt = self.main_df["dt_hr"].max()
        if max_dt > 0:
            self.main_df["dt_hr"] = self.main_df["dt_hr"] / max_dt

        self.sequences = []
        self.labels = []

        # Efficient grouping to align sequences with physics targets
        grouped_main = self.main_df.groupby(["battery_id", "cycle_number"])

        for (bid, cyc), group in grouped_main:
            if len(group) != self.seq_len:
                continue

            # Get matching physics ground truth
            match = stats[(stats["battery_id"] == bid) & (stats["cycle_number"] == cyc)]
            if match.empty:
                continue

            seq = group[FEATURE_COLS].values.astype(np.float32)
            targets = match[["true_soh", "true_q", "true_e"]].values.astype(np.float32)[0]

            self.sequences.append(seq)
            self.labels.append(targets)

        self.sequences = torch.tensor(np.array(self.sequences))
        self.labels = torch.tensor(np.array(self.labels))

        print(f"Dataset ready. Sequences: {self.sequences.shape}, Targets: {self.labels.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
