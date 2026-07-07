import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention
from .layers import FFN, LN


class EncoderBlock(nn.Module):
    """post_layer_norm 的 transformer encoder 块"""

    def __init__(self, dim: int, head: int, eps: float = 1e-5,
                 d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.h = head

        self.attention = MultiHeadAttention(dim, self.h)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = LN(dim, eps)

        self.ffn = FFN(dim, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = LN(dim, eps)

    def forward(self, x: Tensor, pad_mask: Tensor):
        x = x + self.dropout1(self.attention(x, x, x, pad_mask))
        x = self.ln1(x)

        x = x + self.dropout2(self.ffn(x))
        x = self.ln2(x)
        return x


class Encoder(nn.Module):
    """堆叠 N 个 EncoderBlock"""

    def __init__(self, n: int, dim: int, head: int, eps: float = 1e-5,
                 d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(dim, head, eps, d_ff, dropout) for _ in range(n)]
        )

    def forward(self, x: Tensor, pad_mask: Tensor):
        for encoder in self.layers:
            x = encoder(x, pad_mask)
        return x
