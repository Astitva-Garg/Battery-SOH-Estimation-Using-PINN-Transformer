"""Physics-constrained transformer shared by Week 3 and Week 4.

Predicts State of Health (SoH) as the primary task, with two auxiliary heads
(Charge Q and Energy E) that are supervised against physics ground truth
(area under the current/power curve) computed from high-resolution data.
The auxiliary losses act as physics constraints that regularize the SoH head.
"""

import torch
import torch.nn as nn


class AttnRecordingEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer that stashes its self-attention weights
    (averaged over heads) so we can plot them later for interpretability."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x, weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
            is_causal=is_causal,
        )
        self.last_attn_weights = weights.detach()
        return self.dropout1(x)


class PhysicsConstrainedTransformer(nn.Module):
    def __init__(self, input_dim=4, seq_len=20, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len

        # Input Embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))

        # Transformer Encoder (layers record their attention weights)
        encoder_layer = AttnRecordingEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Prediction Heads ---
        # 1. SoH Prediction (Primary Task)
        self.head_soh = nn.Sequential(
            nn.Linear(d_model * seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 2. Physics Constraint Head: Charge (Q)
        self.head_q = nn.Sequential(
            nn.Linear(d_model * seq_len, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # 3. Physics Constraint Head: Energy (E)
        self.head_e = nn.Sequential(
            nn.Linear(d_model * seq_len, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]
        x = self.input_proj(x) + self.pos_encoder
        x = self.transformer_encoder(x)

        flat = x.reshape(x.size(0), -1)

        soh = self.head_soh(flat)
        q = self.head_q(flat)
        e = self.head_e(flat)

        return soh, q, e

    def get_attention_maps(self):
        """Returns a list of [Batch, Seq_Len, Seq_Len] attention maps,
        one per transformer encoder layer, from the most recent forward pass."""
        return [layer.last_attn_weights for layer in self.transformer_encoder.layers]
