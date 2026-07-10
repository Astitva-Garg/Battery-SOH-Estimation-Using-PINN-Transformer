"""Week 3: SoH estimation on real NASA battery data.

A transformer constrained by coulomb counting and energy conservation.
Predicts SoH, charge of the cycle, and net energy discharged; the latter two
are supervised against physics ground truth (area under the curve, computed
from the high-resolution resampled dataset) and used as auxiliary losses to
regularize the primary SoH prediction.

Run `gen.py` first to produce the CSVs in `data/`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.train import train_model, visualize_results  # noqa: E402

# --- CONFIGURATION ---
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
SEQ_LEN = 20         # Matches BINS from preprocessing
D_MODEL = 64         # Transformer embedding dimension
NHEAD = 4            # Number of attention heads
NUM_LAYERS = 2       # Number of transformer layers
LAMBDA_Q = 0.3       # Constraint weight for Charge
LAMBDA_E = 0.3       # Constraint weight for Energy

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAIN_CSV = os.path.join(DATA_DIR, "battery_main_binned20.csv")
PHYS_CSV = os.path.join(DATA_DIR, "battery_phys_resampled200.csv")


if __name__ == "__main__":
    if os.path.exists(MAIN_CSV) and os.path.exists(PHYS_CSV):
        model, test_loader = train_model(
            MAIN_CSV,
            PHYS_CSV,
            seq_len=SEQ_LEN,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            lambda_q=LAMBDA_Q,
            lambda_e=LAMBDA_E,
        )
        visualize_results(model, test_loader)
    else:
        print("Please run gen.py first to generate the CSV files in week3/data/.")
