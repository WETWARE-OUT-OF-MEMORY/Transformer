import torch
import torch.nn as nn
from torch.nn import init
from torch import Tensor

class MultiHeadAttention(nn.Module):
    """
    多头注意力模块
    含KV Cache
    """

    def __init__(self, dim: int, head: int, dropout: float=0.1, need_mask: bool=False):
        super().__init__()

        self.w_q = nn.Parameter(init.xavier_uniform_(torch.empty(dim, dim)), requires_grad=True)
        self.w_k = nn.Parameter(init.xavier_uniform_(torch.empty(dim, dim)), requires_grad=True)
        self.w_v = nn.Parameter(init.xavier_uniform_(torch.empty(dim, dim)), requires_grad=True)
        self.w_o = nn.Parameter(init.xavier_uniform_(torch.empty(dim, dim)), requires_grad=True)

        self.h = head
        self.mask = need_mask

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q: Tensor, x_k: Tensor, x_v: Tensor, pad_mask: Tensor = None,
                past_kv: tuple=None, use_cache: bool=False, is_cross: bool=False):
        dim = x_q.shape[-1]

        assert dim % self.h == 0

        q = x_q @ self.w_q
        q = q.reshape(q.shape[0], q.shape[1], self.h, -1) \
            .transpose(1, 2)

        if is_cross and past_kv is not None:
            k, v = past_kv
        else:
            k = x_k @ self.w_k
            v = x_v @ self.w_v

            # 不使用kv_cache- k/v: [batch, head, n, dim//head]
            # 使用kv_cache-  k/v: [batch, head, 1, dim//head]
            # 训练- q: [batch, head, n, dim//head]
            # 生成- q: [batch, head, 1, dim//head]

            k = k.reshape(k.shape[0], k.shape[1], self.h, -1) \
                .transpose(1, 2)
            v = v.reshape(v.shape[0], v.shape[1], self.h, -1) \
                .transpose(1, 2)

            if past_kv is not None:
                # past_k/past_v: [batch, head, n-1, dim//head]
                past_k, past_v = past_kv
                k = torch.concat((past_k, k), dim=2)
                v = torch.concat((past_v, v), dim=2)
        # k/v: [batch, head, n, dim//head]

        # pad_mask: [batch, n] ---.unsqueeze(1).unsqueeze(2)---> [batch, 1, 1, n]
        # 训练- relevance: [batch, head, n, n]
        # 生成- relevance: [batch, head, 1, n]
        k_t = k.transpose(-2, -1)
        relevance = torch.matmul(q, k_t) / (dim // self.h) ** 0.5

        # pad_mask: [batch, n] -unsqueeze-> [batch, 1, 1, n]
        if pad_mask is not None:
            relevance = relevance.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        if self.mask and past_kv is None:
            # torch.triu(input, diagonal=0, *, out=None)
            # 使diagonal对应的对角线及以上保持，以下归零
            mask = torch.triu(torch.ones(relevance.shape[-1], relevance.shape[-1],
                                         dtype=torch.bool), diagonal=1).to(relevance.device)
            relevance = relevance.masked_fill(mask, float('-inf'))
        relevance = torch.softmax(relevance, dim=-1)

        relevance = self.dropout(relevance)

        # att: [batch, head, n, dim//head]
        att = relevance @ v
        # att: [batch, n, dim]
        att = att.transpose(1, 2).contiguous()
        att = att.reshape(att.shape[0], att.shape[1], -1)
        att = att @ self.w_o

        if use_cache:
            return att, (k, v)
        return att
