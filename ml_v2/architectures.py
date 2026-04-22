import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.02):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return F.relu(out + residual)

class AdvancedResMLP(nn.Module):
    """
    The main architecture based on Phase 4 recommendations.
    Accepts dynamic feature dimensions and builds a deep residual network.
    """
    def __init__(self, input_dim, width=256, depth=4, dropout=0.02):
        super(AdvancedResMLP, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU()
        )
        
        self.res_blocks = nn.ModuleList([
            ResBlock(width, dropout) for _ in range(depth)
        ])
        
        self.fc_out = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, 1)
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.fc_out(x)