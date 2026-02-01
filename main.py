import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# REMOVED: from sklearn.preprocessing import StandardScaler (Caused the physics bug)

# --- CONFIGURATION ---
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
SEQ_LEN = 20         # Matches BINS from your preprocessing
d_model = 64         # Transformer embedding dimension
nhead = 4            # Number of attention heads
num_layers = 2       # Number of transformer layers
LAMBDA_Q = 0.3       # Constraint weight for Charge
LAMBDA_E = 0.3       # Constraint weight for Energy

# --- 1. DATASET PREPARATION (FIXED) ---

class BatteryPhysicsDataset(Dataset):
    def __init__(self, main_file, phys_file):
        # Load datasets
        self.main_df = pd.read_csv(main_file)
        self.phys_df = pd.read_csv(phys_file)
        
        # --- Precompute Physics Constraints (Ground Truth) ---
        print("Computing physics ground truth from high-res data...")
        self.phys_df['power'] = self.phys_df['Voltage_measured'] * self.phys_df['Current_measured']
        self.phys_df['dq'] = self.phys_df['Current_measured'] * self.phys_df['dt_hr']
        self.phys_df['de'] = self.phys_df['power'] * self.phys_df['dt_hr']
        
        # Group by cycle to get scalar totals
        stats = self.phys_df.groupby(['battery_id', 'cycle_number']).agg({
            'dq': 'sum',   # Total Charge (Ah)
            'de': 'sum',   # Total Energy (Wh)
            'SoH': 'first' # SoH is constant per cycle
        }).reset_index()
        
        # Rename for clarity
        stats.rename(columns={'dq': 'true_q', 'de': 'true_e', 'SoH': 'true_soh'}, inplace=True)
        # Q and E should be positive magnitude for capacity/energy
        stats['true_q'] = stats['true_q'].abs()
        stats['true_e'] = stats['true_e'].abs()
        
        # --- PHYSICS-PRESERVING NORMALIZATION (THE FIX) ---
        # We manually scale features to preserve the physical "Zero" point.
        # StandardScaler was shifting 'dt' to negative values, breaking the integration.
        
        # 1. Voltage: Normalize 0-5V range roughly to 0-1
        self.main_df['Voltage_measured'] = self.main_df['Voltage_measured'] / 5.0
        
        # 2. Current: Normalize by max absolute value. 
        # Crucial: Preserves 0.0 Amps as 0.0 (StandardScaler would shift this).
        max_current = self.main_df['Current_measured'].abs().max()
        if max_current > 0:
            self.main_df['Current_measured'] = self.main_df['Current_measured'] / max_current
            
        # 3. Temperature: Center around 20C, scale by 40 (range approx -0.5 to 1.5)
        self.main_df['Temperature_measured'] = (self.main_df['Temperature_measured'] - 20.0) / 40.0
        
        # 4. Time steps (dt): MUST BE STRICTLY POSITIVE. 
        # StandardScaler made these negative, which caused the +2.5 offset bias.
        max_dt = self.main_df['dt_hr'].max()
        if max_dt > 0:
            self.main_df['dt_hr'] = self.main_df['dt_hr'] / max_dt

        # Define feature columns after scaling
        feature_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'dt_hr']
        
        self.sequences = []
        self.labels = []
        
        # Efficient grouping to align sequences with physics targets
        grouped_main = self.main_df.groupby(['battery_id', 'cycle_number'])
        
        for (bid, cyc), group in grouped_main:
            if len(group) != SEQ_LEN: continue
            
            # Get matching physics ground truth
            match = stats[(stats['battery_id'] == bid) & (stats['cycle_number'] == cyc)]
            if match.empty: continue
            
            # Convert to float32
            seq = group[feature_cols].values.astype(np.float32)
            targets = match[['true_soh', 'true_q', 'true_e']].values.astype(np.float32)[0]
            
            self.sequences.append(seq)
            self.labels.append(targets)
            
        self.sequences = torch.tensor(np.array(self.sequences))
        self.labels = torch.tensor(np.array(self.labels))
        
        print(f"Dataset ready. Sequences: {self.sequences.shape}, Targets: {self.labels.shape}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# --- 2. PHYSICS-CONSTRAINED TRANSFORMER MODEL ---

class PhysicsConstrainedTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        # Input Embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, SEQ_LEN, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- Prediction Heads ---
        # 1. SoH Prediction (Primary Task)
        self.head_soh = nn.Sequential(
            nn.Linear(d_model * SEQ_LEN, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 2. Physics Constraint Head: Charge (Q)
        self.head_q = nn.Sequential(
            nn.Linear(d_model * SEQ_LEN, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 3. Physics Constraint Head: Energy (E)
        self.head_e = nn.Sequential(
            nn.Linear(d_model * SEQ_LEN, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]
        
        # Embedding + Positional Encoding
        x = self.input_proj(x) + self.pos_encoder
        
        # Pass through Transformer
        x = self.transformer_encoder(x)
        
        # Flatten for dense heads
        flat = x.reshape(x.size(0), -1)
        
        # Multi-Task Outputs
        soh = self.head_soh(flat)
        q = self.head_q(flat)
        e = self.head_e(flat)
        
        return soh, q, e

# --- 3. TRAINING LOOP ---

def train_model():
    # Load Data
    dataset = BatteryPhysicsDataset("battery_main_binned20.csv", "battery_phys_resampled200.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model
    model = PhysicsConstrainedTransformer(input_dim=4, d_model=d_model, nhead=nhead, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print("\nStarting Training with Physics Constraints...")
    print(f"Weights -> SoH: 1.0, Charge(Q): {LAMBDA_Q}, Energy(E): {LAMBDA_E}")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        l_soh_total = 0
        
        for x, y in train_loader:
            # y contains [soh, q, e]
            true_soh = y[:, 0].unsqueeze(1)
            true_q = y[:, 1].unsqueeze(1)
            true_e = y[:, 2].unsqueeze(1)
            
            optimizer.zero_grad()
            
            pred_soh, pred_q, pred_e = model(x)
            
            # --- Physics Constrained Loss Calculation ---
            l_soh = criterion(pred_soh, true_soh)
            l_q = criterion(pred_q, true_q)
            l_e = criterion(pred_e, true_e)
            
            loss = l_soh + (LAMBDA_Q * l_q) + (LAMBDA_E * l_e)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            l_soh_total += l_soh.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Total Loss: {total_loss/len(train_loader):.4f} | SoH MSE: {l_soh_total/len(train_loader):.4f}")
            
    return model, test_loader

# --- 4. VISUALIZATION ---

def visualize_results(model, loader):
    model.eval()
    preds_soh, trues_soh = [], []
    
    # 1. Collect Predictions
    with torch.no_grad():
        for x, y in loader:
            p_soh, p_q, p_e = model(x)
            preds_soh.extend(p_soh.flatten().numpy())
            trues_soh.extend(y[:, 0].flatten().numpy())
            
    preds_soh = np.array(preds_soh)
    trues_soh = np.array(trues_soh)
    
    # 2. Plotting
    plt.figure(figsize=(14, 6))
    
    # Plot A: True vs Pred (Scatter)
    plt.subplot(1, 2, 1)
    plt.scatter(preds_soh, trues_soh, s=10, alpha=0.7)
    # Identity line
    lims = [min(min(preds_soh), min(trues_soh)), max(max(preds_soh), max(trues_soh))]
    plt.plot(lims, lims, 'orange', alpha=0.75, zorder=0)
    plt.xlabel("Predicted SoH")
    plt.ylabel("True SoH")
    plt.title("SoH Prediction Accuracy")
    plt.grid(True, alpha=0.3)
    
    # Plot B: Residual Histogram
    plt.subplot(1, 2, 2)
    residuals = preds_soh - trues_soh
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.8)
    plt.axvline(0, color='r', linestyle='--', linewidth=1) # Zero line for reference
    plt.xlabel("Residual (Pred - True)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution (Should be centered at 0)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # 3. Visualization Check for Inputs
    # We verify that dt is now positive
    print("\nVerifying Input Features (Check dt is positive)...")
    x_sample, _ = next(iter(loader))
    sample_seq = x_sample[0].numpy()
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(sample_seq.T, cmap='viridis', cbar=True)
    plt.title("Input Sequence Features (Corrected Normalization)")
    plt.xlabel("Time Step (Bin)")
    plt.yticks([0.5, 1.5, 2.5, 3.5], ['V', 'I', 'T', 'dt'], rotation=0)
    plt.show()

if __name__ == "__main__":
    import os
    if os.path.exists("battery_main_binned20.csv") and os.path.exists("battery_phys_resampled200.csv"):
        model, test_loader = train_model()
        visualize_results(model, test_loader)
    else:
        print("Please run the preprocessing script first to generate CSV files.")