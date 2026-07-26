import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init

from .encoder import Encoder
from .decoder import Decoder
from .layers import LinearBlock


class Transformer(nn.Module):
    """完整的 Transformer 模型"""

    def __init__(self, layer_num: int, d_model: int, src_emb_sz: int,
                 tgt_emb_sz: int, head: int, eps: float = 1e-5,
                 d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

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
        pe_src = self.positional_encoding(src.shape[-2])
        src = src + pe_src.to(src.device)
        return self.encoder(self.dropout(src), src_pad_mask)

    def decode(self, tgt_ids: Tensor, tgt_pad_mask: Tensor,
               encoder_output: Tensor, src_pad_mask: Tensor):
        tgt = self.tgt_emb(tgt_ids) * math.sqrt(self.d_model)

        pe_tgt = self.positional_encoding(tgt.shape[-2])
        tgt = tgt + pe_tgt.to(tgt.device)
        return self.decoder(self.dropout(tgt), encoder_output, src_pad_mask, tgt_pad_mask)

    def forward(self, src_ids: Tensor, tgt_ids: Tensor,
                src_pad_mask: Tensor, tgt_pad_mask: Tensor):
        encoder_output = self.encode(src_ids, src_pad_mask)
        tgt = self.decode(tgt_ids, tgt_pad_mask, encoder_output, src_pad_mask)
        tgt = self.linear(tgt)
        return tgt

@torch.no_grad()
def generate(model, tokenizer, input_text:str, max_length:int, device='cuda'):
    """
    输入文本，模型复制输出
    返回: (输入字符串, 输出字符串, token级准确率)
    """
    model.eval()
    # 1. Tokenize 输入
    input_ids = tokenizer.encode(input_text)[:max_length]
    input_len = len(input_ids)
    src = torch.tensor([input_ids]).to(device)
    src_pad_mask = (src == tokenizer.pad_token_id).to(torch.bool)
    # 2. Encoder — 只跑一次
    encoder_output = model.encode(src, src_pad_mask)
    # 3. Decoder — 自回归生成
    # 起始 token: eos_token（在复制任务中充当 BOS）
    generated = [tokenizer.eos_token_id]
    for step in range(max_length - 1):
        tgt = torch.tensor(generated).unsqueeze(0).to(device)
        tgt_pad_mask = torch.zeros_like(tgt, dtype=torch.bool)
        # 取最后一个位置的 logits
        logits = model.decode(tgt, tgt_pad_mask, encoder_output, src_pad_mask)
        next_token_logits = logits[:, -1, :]  # [1, vocab_sz]
        # 贪心解码
        next_token = next_token_logits.argmax(dim=-1).item()
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    # 4. Detokenize：跳过起始 eos
    output_ids = generated[1:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    # 5. 评估复制准确率
    min_len = min(input_len, len(output_ids))
    correct = sum(1 for i in range(min_len) if input_ids[i] == output_ids[i])
    accuracy = correct / max(input_len, len(output_ids))
    return input_text, output_text, accuracy