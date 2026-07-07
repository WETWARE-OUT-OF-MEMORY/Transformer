import torch
import torch.nn as nn
from torch import Tensor


class FFN(nn.Module):
    """Transformer 内置双层全连接前向层"""

    def __init__(self, dim: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = dim * 4

        self.layer1 = nn.Linear(dim, d_ff)
        self.layer2 = nn.Linear(d_ff, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.relu(self.layer1(x))
        return self.layer2(self.dropout(x))


class LN(nn.Module):
    """自定义 Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor):
        """x: [batch, n, dim]"""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta


class LinearBlock(nn.Module):
    """线性投影层"""

    def __init__(self, dim: int, emb_dim: int):
        super().__init__()
        self.layer = nn.Linear(dim, emb_dim)

    def forward(self, x: Tensor):
        return self.layer(x)
