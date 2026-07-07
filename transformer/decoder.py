import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention
from .layers import FFN, LN


class DecoderBlock(nn.Module):
    """post_layer_norm 的 transformer decoder 块"""

    def __init__(self, dim: int, head: int, eps: float = 1e-5,
                 d_ff: int = None, dropout: float = 0.1):
        super().__init__()

        self.h = head

        self.mask_attention = MultiHeadAttention(dim, self.h, need_mask=True)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = LN(dim, eps)

        self.attention = MultiHeadAttention(dim, self.h)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = LN(dim, eps)

        self.ffn = FFN(dim, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ln3 = LN(dim, eps)

    def forward(self, x: Tensor, encoder_output: Tensor,
                src_pad_mask: Tensor, tgt_pad_mask: Tensor):
        x = x + self.dropout1(self.mask_attention(x, x, x, tgt_pad_mask))
        x = self.ln1(x)

        x = x + self.dropout2(
            self.attention(x, encoder_output, encoder_output, src_pad_mask))
        x = self.ln2(x)

        x = x + self.dropout3(self.ffn(x))
        x = self.ln3(x)
        return x


class Decoder(nn.Module):
    """堆叠 N 个 DecoderBlock"""

    def __init__(self, n: int, dim: int, head: int, eps: float = 1e-5,
                 d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderBlock(dim, head, eps, d_ff, dropout) for _ in range(n)]
        )

    def forward(self, x: Tensor, encoder_output: Tensor,
                src_pad_mask: Tensor, tgt_pad_mask: Tensor):
        for decoder in self.layers:
            x = decoder(x, encoder_output, src_pad_mask, tgt_pad_mask)
        return x
