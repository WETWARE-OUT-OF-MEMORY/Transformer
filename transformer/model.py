import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init

from typing import List

from .encoder import Encoder
from .decoder import Decoder
from .layers import LinearBlock


class Transformer(nn.Module):
    """完整的 Transformer 模型"""

    def __init__(self, layer_num: int, d_model: int, src_emb_sz: int, tgt_emb_sz: int, head: int,
                 eps: float = 1e-5, d_ff: int = None, dropout: float = 0.1, max_length: int = 500):
        super().__init__()
        self.d_model = d_model
        # 普通变量不会随着model.to(device)移动到device
        # self.register_buffer()将非参数的变量注册为buffer, 可以多次使用注册多个buffer
        # self.pe = self.positional_encoding(max_length)
        self.register_buffer('pe', self.positional_encoding(max_length), persistent=False)
        self.src_emb = nn.Embedding(src_emb_sz, self.d_model)
        # init.xavier_uniform_(self.src_emb.weight)
        self.tgt_emb = nn.Embedding(tgt_emb_sz, self.d_model)
        # init.xavier_uniform_(self.tgt_emb.weight)

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = Encoder(layer_num, self.d_model, head, eps, d_ff,
                               dropout)
        self.decoder = Decoder(layer_num, self.d_model, head, eps, d_ff,
                               dropout)
        self.linear = LinearBlock(self.d_model, tgt_emb_sz)
        init.xavier_uniform_(self.linear.layer.weight)
        # 最终输出与两个embedding共享参数
        self.src_emb.weight = self.linear.layer.weight
        self.tgt_emb.weight = self.linear.layer.weight

    def positional_encoding(self, n: int):
        """生成位置编码矩阵 [n, d_model]"""
        pe = torch.zeros(n, self.d_model)
        meta = torch.exp(
            torch.arange(0, self.d_model, 2) * -math.log(10000) / self.d_model
        )
        position = torch.arange(0, n).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * meta)
        pe[:, 1::2] = torch.cos(position * meta)
        return pe

    def encode(self, src_ids: Tensor, src_pad_mask: Tensor):
        src = self.src_emb(src_ids) * math.sqrt(self.d_model)
        # pe_src = self.positional_encoding(src.shape[-2])

        # 确保预生成的pe满足最大长度要求
        assert src.shape[-2] <= self.pe.shape[0]

        pe_src = self.pe[:src.shape[-2]]
        src = src + pe_src.to(src.device)
        return self.encoder(self.dropout(src), src_pad_mask)

    def decode(self, tgt_ids: Tensor, tgt_pad_mask: Tensor, encoder_output: Tensor, src_pad_mask: Tensor,
               past_kv_list: List=None, use_cache: bool=False, start_pos:int=0):
        # tgt: [batch, n, dim]
        tgt = self.tgt_emb(tgt_ids) * math.sqrt(self.d_model)
        # pe_tgt = self.positional_encoding(tgt.shape[-2])

        # 确保预生成的pe满足最大长度要求
        assert start_pos + tgt.shape[-2] <= self.pe.shape[0]

        pe_tgt = self.pe[start_pos:start_pos + tgt.shape[-2]]
        tgt = tgt + pe_tgt.to(tgt.device)
        return self.decoder(self.dropout(tgt), encoder_output, src_pad_mask, tgt_pad_mask,
                            past_kv_list, use_cache)

    def forward(self, src_ids: Tensor, tgt_ids: Tensor,
                src_pad_mask: Tensor, tgt_pad_mask: Tensor):
        encoder_output = self.encode(src_ids, src_pad_mask)
        tgt = self.decode(tgt_ids, tgt_pad_mask, encoder_output, src_pad_mask)
        tgt = self.linear(tgt)
        return tgt

def sample_token(logits: Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> int:
    """
    logits: [VOCAB_SZ] 单个位置的原始 logits
    返回采样的 token id。
    temperature=1, top_k=0, top_p=0 时退化为贪心。
    """
    if temperature != 1.0:
        logits = logits / temperature

    # Top-K：保留前 K 个，其余置 -inf
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        topk_vals, _ = torch.topk(logits, k)
        logits[logits < topk_vals[-1]] = float("-inf")

    # Top-P：按累积概率截断
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        # cumsum: 含当前项的累加和
        cumsum = torch.cumsum(probs, dim=-1)
        # 找到累积概率首次超过 top_p 的位置，其后的全部剔除
        # 剔除了当前项，使截断后至少含有一项
        mask = cumsum - probs > top_p
        sorted_logits[mask] = float("-inf")
        # tensor.argsort(): 返回排序后元素的原索引，即tensor.sort()索引部分的逆操作
        # tensor.gather(dim, idx): 在指定tensor的dim维度按照idx查表
        logits = sorted_logits.gather(0, sorted_idx.argsort())
    # 如选择截断前softmax还需要一次归一化，引入浮点误差
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

@torch.no_grad()
def generate(model, tokenizer, input_text:str, max_length: int, device='cuda'):
    """
    输入文本，模型复制输出
    返回: (输入字符串, 输出字符串, token级准确率)
    """
    model.eval()
    # 1. Tokenize 输入
    input_ids = tokenizer.encode(input_text)[:max_length - 2]
    input_ids = [tokenizer.bos_id()] + input_ids + [tokenizer.eos_id()]
    src = torch.tensor([input_ids]).to(device)
    src_pad_mask = (src == tokenizer.pad_id()).to(torch.bool)
    # 2. Encoder — 只跑一次
    encoder_output = model.encode(src, src_pad_mask)
    # 3. Decoder — 自回归生成
    generated = [tokenizer.bos_id()]
    past_kv_list = None
    for step in range(max_length - 1):
        # tgt = torch.tensor(generated).unsqueeze(0).to(device)
        #  只传最新token
        tgt = torch.tensor(generated[-1]).unsqueeze(0).unsqueeze(0).to(device)
        # tgt_pad_mask = torch.zeros_like(tgt, dtype=torch.bool)

        hidden_state, past_kv_list = model.decode(tgt, None, encoder_output, src_pad_mask,
                                    past_kv_list=past_kv_list, use_cache=True, start_pos=step)
        # ToDo: 采样封装
        # 取最后一个位置的 logits
        next_token_logits = model.linear(hidden_state)[:, -1, :]  # [1, vocab_sz]
        # # 贪心解码
        # next_token = next_token_logits.argmax(dim=-1).item()
        # generated.append(next_token)

        # 暂时保持贪心设置
        next_token = sample_token(next_token_logits)

        if next_token == tokenizer.eos_id():
            break
    # 4. Detokenize：
    # 跳过起始 bos
    # output_ids = generated[1:]

    # 过滤pad_id、bos_id、eos_id
    output_ids = [i for i in generated if i not in {tokenizer.pad_id(), tokenizer.bos_id(), tokenizer.eos_id()}]
    output_text = tokenizer.decode(output_ids)
    return output_text
