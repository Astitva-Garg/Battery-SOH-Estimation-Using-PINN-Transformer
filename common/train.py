"""Shared training loop and visualization helpers for the physics-constrained
transformer, used by both Week 3 (real NASA data) and Week 4 (SPM-simulated
data).
"""

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from common.dataset import BatteryPhysicsDataset
from common.model import PhysicsConstrainedTransformer


def train_model(
    main_csv,
    phys_csv,
    seq_len=20,
    d_model=64,
    nhead=4,
    num_layers=2,
    batch_size=32,
    epochs=100,
    lr=1e-4,
    lambda_q=0.3,
    lambda_e=0.3,
):
    dataset = BatteryPhysicsDataset(main_csv, phys_csv, seq_len=seq_len)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PhysicsConstrainedTransformer(
        input_dim=4, seq_len=seq_len, d_model=d_model, nhead=nhead, num_layers=num_layers
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("\nStarting Training with Physics Constraints...")
    print(f"Weights -> SoH: 1.0, Charge(Q): {lambda_q}, Energy(E): {lambda_e}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        l_soh_total = 0

        for x, y in train_loader:
            true_soh = y[:, 0].unsqueeze(1)
            true_q = y[:, 1].unsqueeze(1)
            true_e = y[:, 2].unsqueeze(1)

            optimizer.zero_grad()

            pred_soh, pred_q, pred_e = model(x)

            l_soh = criterion(pred_soh, true_soh)
            l_q = criterion(pred_q, true_q)
            l_e = criterion(pred_e, true_e)

            loss = l_soh + (lambda_q * l_q) + (lambda_e * l_e)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            l_soh_total += l_soh.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss/len(train_loader):.4f} "
                f"| SoH MSE: {l_soh_total/len(train_loader):.4f}"
            )

    return model, test_loader


def visualize_results(model, loader, feature_labels=("V", "I", "T", "dt")):
    model.eval()
    preds_soh, trues_soh = [], []

    with torch.no_grad():
        for x, y in loader:
            p_soh, p_q, p_e = model(x)
            preds_soh.extend(p_soh.flatten().numpy())
            trues_soh.extend(y[:, 0].flatten().numpy())

    preds_soh = np.array(preds_soh)
    trues_soh = np.array(trues_soh)

    # Plot A/B: True vs Pred scatter + residual histogram
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(preds_soh, trues_soh, s=10, alpha=0.7)
    lims = [min(min(preds_soh), min(trues_soh)), max(max(preds_soh), max(trues_soh))]
    plt.plot(lims, lims, "orange", alpha=0.75, zorder=0)
    plt.xlabel("Predicted SoH")
    plt.ylabel("True SoH")
    plt.title("SoH Prediction Accuracy")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    residuals = preds_soh - trues_soh
    plt.hist(residuals, bins=30, edgecolor="k", alpha=0.8)
    plt.axvline(0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("Residual (Pred - True)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution (Should be centered at 0)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Input feature heatmap for one sample sequence
    print("\nVerifying Input Features (Check dt is positive)...")
    x_sample, _ = next(iter(loader))
    sample_seq = x_sample[0].numpy()

    plt.figure(figsize=(10, 5))
    sns.heatmap(sample_seq.T, cmap="viridis", cbar=True)
    plt.title("Input Sequence Features (Corrected Normalization)")
    plt.xlabel("Time Step (Bin)")
    n_features = sample_seq.shape[1]
    plt.yticks([i + 0.5 for i in range(n_features)], list(feature_labels)[:n_features], rotation=0)
    plt.show()

    # Attention head visualization (mean over heads, mean over batch)
    print("\nExtracting mean self-attention maps per layer...")
    with torch.no_grad():
        model(x_sample)  # forward pass to populate last_attn_weights

    attn_maps = model.get_attention_maps()
    num_layers_ = len(attn_maps)

    plt.figure(figsize=(6 * num_layers_, 5))
    for i, attn in enumerate(attn_maps):
        mean_attn = attn.mean(dim=0).numpy()
        plt.subplot(1, num_layers_, i + 1)
        sns.heatmap(mean_attn, cmap="viridis", cbar=True)
        plt.title(f"Layer {i} mean attention")
        plt.xlabel("Key positions")
        plt.ylabel("Query positions")
    plt.tight_layout()
    plt.show()
