import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.02):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return F.relu(out + x)


class DeepSetsNet(nn.Module):
    """
    Column-permutation-invariant network for estimating log2(m-height).

    Each column of P (a k-vector representing one parity position) is encoded
    independently by a shared MLP. The resulting embeddings are aggregated with
    sum + mean + max pooling, which is invariant to the order of the columns.
    Three scalar context values (n, k, m) are appended before the head.

    This architecture directly respects the mathematical symmetry that permuting
    the parity columns of G leaves the m-height of the code unchanged.
    """

    def __init__(self, k, n_minus_k, col_embed_dim=128, head_width=256, head_depth=4, dropout=0.02):
        super().__init__()

        # Shared encoder: maps each k-dimensional column → col_embed_dim embedding.
        # BatchNorm is applied over (batch * n_minus_k) samples, which is valid
        # because every column embedding is an independent forward pass through the same weights.
        self.col_encoder = nn.Sequential(
            nn.Linear(k, col_embed_dim),
            nn.BatchNorm1d(col_embed_dim),
            nn.ReLU(),
            nn.Linear(col_embed_dim, col_embed_dim),
            nn.BatchNorm1d(col_embed_dim),
            nn.ReLU(),
        )

        # sum + mean + max → 3 * col_embed_dim, then 3 context scalars (n, k, m)
        agg_dim = 3 * col_embed_dim + 3

        head_layers = [
            nn.Linear(agg_dim, head_width),
            nn.BatchNorm1d(head_width),
            nn.ReLU(),
        ]
        for _ in range(head_depth):
            head_layers.append(ResBlock(head_width, dropout))
        head_layers += [
            nn.Linear(head_width, head_width // 2),
            nn.ReLU(),
            nn.Linear(head_width // 2, 1),
        ]
        self.head = nn.Sequential(*head_layers)

    def forward(self, P, context):
        # P:       (batch, k, n_minus_k)
        # context: (batch, 3)  — [n, k, m] as floats
        batch, k, cols = P.shape

        # Flatten columns across the batch for the shared encoder
        P_cols = P.permute(0, 2, 1).reshape(batch * cols, k)  # (batch*cols, k)
        embeddings = self.col_encoder(P_cols)                  # (batch*cols, embed_dim)
        embeddings = embeddings.view(batch, cols, -1)          # (batch, cols, embed_dim)

        agg_sum  = embeddings.sum(dim=1)            # (batch, embed_dim)
        agg_mean = embeddings.mean(dim=1)           # (batch, embed_dim)
        agg_max  = embeddings.max(dim=1).values     # (batch, embed_dim)

        agg = torch.cat([agg_sum, agg_mean, agg_max, context], dim=1)
        return self.head(agg)
