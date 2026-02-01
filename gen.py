import kagglehub
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
path = kagglehub.dataset_download("patrickfleith/nasa-battery-dataset")

V_CUTOFF = 2.7
CAPACITY_MIN_AH = 1
BINS = 20
M = 200  # physics resolution 
NOMINAL_AH = 2.0

metadata = pd.read_csv(os.path.join(path, "cleaned_dataset", "metadata.csv"))
metadata["battery_id"] = metadata["battery_id"].astype(str)

excluded_batteries = ["B0049", "B0050", "B0051", "B0052"]

discharge_metadata = metadata[
    (metadata["type"] == "discharge") &
    (~metadata["battery_id"].isin(excluded_batteries))
].copy()

discharge_metadata["cycle_number"] = discharge_metadata.groupby("battery_id").cumcount() + 1

main_rows = []
phys_rows = []

for _, row in tqdm(discharge_metadata.iterrows(), total=len(discharge_metadata)):
    file_path = os.path.join(path, "cleaned_dataset", "data", row["filename"])
    df = pd.read_csv(file_path).copy()

   #sorting
    df = df.sort_values("Time").drop_duplicates(subset=["Time"], keep="first")
    if len(df) < 5:
        continue

    # truncate at cutoff voltage
    cutoff_idx = df[df["Voltage_measured"] < V_CUTOFF].index.min()
    if not pd.isna(cutoff_idx):
        df = df.loc[:cutoff_idx].copy()

    if len(df) < 5:
        continue

    # coulomb count on raw points for labels
    df["dt_hr_raw"] = df["Time"].diff().fillna(0) / 3600.0
    df["dQ"] = df["Current_measured"] * df["dt_hr_raw"]
    capacity = abs(df["dQ"].sum())

    if capacity <= CAPACITY_MIN_AH or capacity>=NOMINAL_AH:
        continue

    # SoC based on this cycle's capacity 
    df["CumQ"] = df["dQ"].cumsum()
    df["SoC"] = 100.0 * (1.0 + df["CumQ"] / capacity)

    soh_value = 100.0 * capacity / NOMINAL_AH

    # resample each cycle to M time points 
    t = df["Time"].to_numpy()
    t0, t1 = float(t[0]), float(t[-1])
    if t1 <= t0:
        continue

    t_grid = np.linspace(t0, t1, M)
    dt_hr = np.diff(t_grid, prepend=t_grid[0]) / 3600.0

    def interp(col):
        return np.interp(t_grid, t, df[col].to_numpy())

    Vg   = interp("Voltage_measured")
    Ig   = interp("Current_measured")
    Tg   = interp("Temperature_measured")
    SOCg = interp("SoC")

    # PHYSICS DATASET (resampled, not binned) 
    for j in range(M):
        phys_rows.append({
            "battery_id": row["battery_id"],
            "cycle_number": row["cycle_number"],
            "t_idx": j,
            "Time_s": t_grid[j] - t0,
            "dt_hr": dt_hr[j],
            "Voltage_measured": Vg[j],
            "Current_measured": Ig[j],
            "Temperature_measured": Tg[j],
            "SoC": SOCg[j],
            "Capacity_Ah": capacity,
            "SoH": soh_value,
            "t_end_s": (t1 - t0),
        })

    # MAIN DATASET (20 bins from the resampled series) 
    rs = pd.DataFrame({
        "Time_s": t_grid - t0,
        "dt_hr": dt_hr,
        "Voltage_measured": Vg,
        "Current_measured": Ig,
        "Temperature_measured": Tg,
        "SoC": SOCg,
    })

    chunks = np.array_split(rs, BINS)
    if len(chunks) != BINS or any(c.empty for c in chunks):
        continue

    for b_idx, c in enumerate(chunks):
        main_rows.append({
            "battery_id": row["battery_id"],
            "cycle_number": row["cycle_number"],
            "bin_idx": b_idx,

            "Time_s": c["Time_s"].mean(),
            "dt_hr": c["dt_hr"].sum(),  # total time represented by this bin

            "Voltage_measured": c["Voltage_measured"].mean(),
            "Current_measured": c["Current_measured"].mean(),
            "Temperature_measured": c["Temperature_measured"].mean(),
            "SoC": c["SoC"].mean(),

            "Capacity_Ah": capacity,
            "SoH": soh_value,
            "t_end_s": (t1 - t0),
        })

main_df = pd.DataFrame(main_rows)
phys_df = pd.DataFrame(phys_rows)

main_df.to_csv("battery_main_binned20.csv", index=False)
phys_df.to_csv("battery_phys_resampled200.csv", index=False)

print("Saved:", main_df.shape, phys_df.shape)
