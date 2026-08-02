import torch.nn as nn
from torch import Tensor

from typing import List

from .attention import MultiHeadAttention
from .layers import FFN, LN


class DecoderBlock(nn.Module):
    """post_layer_norm 的 transformer decoder 块"""

    def __init__(self, dim: int, head: int, eps: float = 1e-5,
                 d_ff: int = None, dropout: float = 0.1):
        super().__init__()

        self.h = head

        self.mask_attention = MultiHeadAttention(dim, self.h, need_mask=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = LN(dim, eps)

        self.attention = MultiHeadAttention(dim, self.h, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = LN(dim, eps)

        self.ffn = FFN(dim, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ln3 = LN(dim, eps)

    def forward(self, x: Tensor, encoder_output: Tensor, src_pad_mask: Tensor, tgt_pad_mask: Tensor,
                past_kv_self: tuple=None, past_kv_cross: tuple=None, use_cache: bool=False):
        if use_cache:
            att, present_kv_self = self.mask_attention(x, x, x, tgt_pad_mask, past_kv_self, use_cache)
        else:
            att = self.mask_attention(x, x, x, tgt_pad_mask)
        x = x + self.dropout1(att)
        x = self.ln1(x)

        if use_cache:
            att, present_kv_cross = self.attention(x, encoder_output, encoder_output, src_pad_mask,
                                                   past_kv=past_kv_cross, use_cache=use_cache, is_cross=True)
        else:
            att = self.attention(x, encoder_output, encoder_output, src_pad_mask)
        x = x + self.dropout2(att)
        x = self.ln2(x)

        x = x + self.dropout3(self.ffn(x))
        x = self.ln3(x)

        if use_cache:
            return x, (present_kv_self, present_kv_cross)
        return x


class Decoder(nn.Module):
    """堆叠 N 个 DecoderBlock"""

    def __init__(self, n: int, dim: int, head: int, eps: float = 1e-5,
                 d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderBlock(dim, head, eps, d_ff, dropout) for _ in range(n)]
        )

    def forward(self, x: Tensor, encoder_output: Tensor, src_pad_mask: Tensor, tgt_pad_mask: Tensor,
                past_kv_list: List=None, use_cache: bool=False):
        new_kv_list = [] if use_cache else None
        for i, decoder in enumerate(self.layers):
            if use_cache:
                past_kv_self, past_kv_cross = past_kv_list[i] if past_kv_list is not None else (None, None)
                x, new_kv = decoder(x, encoder_output, src_pad_mask,tgt_pad_mask, past_kv_self, past_kv_cross, use_cache)
                new_kv_list.append(new_kv)
            else:
                x = decoder(x, encoder_output, src_pad_mask, tgt_pad_mask)
        if use_cache:
            return x, new_kv_list
        return x
