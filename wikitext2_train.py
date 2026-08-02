import os
from typing import Dict, List
import yaml

import torch
from torch import Tensor

from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_from_disk
from torch.nn import functional as F

from transformer import Transformer
from transformer.scheduler import TransformerLRScheduler
from transformer.auto_clip_grad import AutoClipGradNorm
from transformer.agc_clip_grad import AGCGradClip

# ============================================================
# 超参数配置
# ============================================================
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
with open('configs.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
LAYER_NUM = config['MODEL']['LAYER_NUM']
D_MODEL = config['MODEL']['D_MODEL']
HEAD = config['MODEL']['HEAD']
EPS = config['MODEL']['EPS']
D_FF = config['MODEL']['D_FF']
DROPOUT = config['MODEL']['DROPOUT']
MAX_LENGTH = config['MODEL']['MAX_LENGTH']
BETA1 = config['LR']['BETA1']
BETA2 = config['LR']['BETA2']
L_EPS = config['LR']['L_EPS']
WARMUP = config['LR']['WARMUP']
EPOCHS = config['TRAIN']['EPOCHS']
BATCH_SIZE = config['TRAIN']['BATCH_SIZE']

# ============================================================
# Tokenizer & 词表大小
# ============================================================
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
VOCAB_SZ = tokenizer.vocab_size

# ============================================================
# 设备
# ============================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device: ",device)


def collate_fn(batch):
    input_ids = [torch.tensor(x['input_ids']) for x in batch]
    labels = [torch.tensor(x['labels']) for x in batch]

    src = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True,
        padding_value=tokenizer.pad_token_id)
    tgt = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True,
        padding_value=tokenizer.pad_token_id)

    src_pad_mask = (src == tokenizer.pad_token_id)
    tgt_pad_mask = src_pad_mask

    return src, tgt, src_pad_mask, tgt_pad_mask

def tokenize(tokenizer, batch) -> Dict[str, List]:
    return {'input_ids': [tokenizer.encode(x)[:MAX_LENGTH]
                          for x in batch]}

def add_labels(batch):
    return {'labels': batch['input_ids']}

def smoothed_loss(logits: Tensor, targets: Tensor,
                  ignore_index: int, smooth_indices: Tensor, epsilon: float = 0.1):
    """
    logits: [N, V]  模型输出
    targets: [N]    整数 token ID
    ignore_index:   pad 位置不计算 loss
    smooth_indices: 参与平滑均匀分布的 token ID 列表（不含 PAD、EOS）
    epsilon:        平滑强度
    """
    mask = (targets != ignore_index)
    logits = logits[mask]
    targets = targets[mask]
    # 正确标签的 one-hot
    true_dist = torch.zeros_like(logits)
    true_dist.scatter_(1, targets.unsqueeze(1), 1.0)
    # 平滑背景：只在 smooth_indices 上均匀分布
    smooth_dist = torch.zeros_like(logits)
    smooth_dist[:, smooth_indices] = 1.0 / len(smooth_indices)
    # 混合
    target_dist = true_dist * (1 - epsilon) + smooth_dist * epsilon
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1).mean()

if __name__ == "__main__":
    # ============================================================
    # 模型、优化器、调度器、损失函数
    # ============================================================
    tf = Transformer(
        layer_num=LAYER_NUM,
        d_model=D_MODEL,
        src_emb_sz=VOCAB_SZ,
        tgt_emb_sz=VOCAB_SZ,
        head=HEAD,
        eps=EPS,
        d_ff=D_FF,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        params=tf.parameters(), betas=(BETA1, BETA2), eps=L_EPS)
    lr_scheduler = TransformerLRScheduler(
        optimizer=optimizer, d_model=D_MODEL, warmup_steps=WARMUP)
    # criterion = torch.nn.CrossEntropyLoss(
    #     ignore_index=tokenizer.pad_token_id)

    # auto_clipper = AutoClipGradNorm(percentile=90, buffer_size=100)
    # agc_grad_clipper = AGCGradClip()
    # scaler = GradScaler()

    # 平滑分布只覆盖非 PAD、非 EOS 的 token
    smooth_indices = [i for i in range(VOCAB_SZ)
                      if i != tokenizer.pad_token_id and i != tokenizer.eos_token_id]
    smooth_indices = torch.tensor(smooth_indices, device=device)

    # ============================================================
    # 数据加载
    # ============================================================
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "wikitext2")

    train_data = load_from_disk(os.path.join(data_path, "train"))
    train_data = train_data.filter(lambda x: len(x['text'].strip()) > 10)

    train_data = train_data.map(
        lambda x: tokenize(tokenizer, x['text']),
        batched=True, remove_columns=['text'])

    train_data = train_data.map(lambda x: add_labels(x), batched=True)

    train_loader = DataLoader(
        train_data, collate_fn=collate_fn,
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True)

    # ============================================================
    # 训练循环
    # ============================================================

    start_epoch = 0
    if os.path.exists('transformer.pt'):
        print("Resuming from checkpoint...")
        checkpoint = torch.load('transformer.pt', map_location=device)
        tf.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('current_epoch', 0)
        print(f"Resumed at epoch {start_epoch}, step_num={lr_scheduler.step_num}")

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        epoch_loss = 0
        cnt = 0
        for batch in train_loader:
            src, tgt, src_pad_mask, tgt_pad_mask = batch
            src = src.to(device)
            tgt = tgt.to(device)
            src_pad_mask = src_pad_mask.to(device)
            tgt_pad_mask = tgt_pad_mask.to(device)

            optimizer.zero_grad()

            y = tf(src, tgt[:, :-1], src_pad_mask, tgt_pad_mask[:, :-1])

            # loss = criterion(y.reshape(-1, VOCAB_SZ), tgt[:, 1:].reshape(-1))
            loss = smoothed_loss(
                y.reshape(-1, VOCAB_SZ),
                tgt[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_token_id,
                smooth_indices=smooth_indices,
                epsilon=0.1
            )
            numerical_loss = loss.item()
            if torch.isnan(numerical_loss):
                for x in batch:
                    print(tokenizer.decode(x, skip_special_tokens=True))
                    print(y.reshape(-1, VOCAB_SZ))
                    print(tgt[:, 1:].reshape(-1))
                    input()
            if (cnt + 1) % 50 == 0:
                print(f"Batch No: {cnt + 1}, loss: {numerical_loss}")

            epoch_loss += numerical_loss
            cnt += 1

            loss.backward()

            # 梯度裁剪：将梯度的全局 L2 范数限制在 max_norm 以内
            # torch.nn.utils.clip_grad_norm_(tf.parameters(), max_norm=1.0)

            # 基于历史百分位的动态阈值（AutoClip）
            # Seethrough 等人提出的方法，每 N 步统计历史梯度范数的 p90 分位数作为阈值：
            # auto_clipper.clip(tf.parameters())

            # AGC（Adaptive Gradient Clipping，自适应梯度裁剪）
            # 不要用固定的梯度阈值裁剪，而是根据参数自身的大小动态决定梯度允许的最大范围。
            # agc_grad_clipper.clip(tf.parameters())

            optimizer.step()
            lr_scheduler.step()
        print(f"Epoch: {epoch}; Average loss: {epoch_loss / cnt}")

    # ============================================================
    # 保存模型
    # ============================================================
    checkpoint = {
        'model_state_dict': tf.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'current_epoch': start_epoch + EPOCHS,
    }
    torch.save(checkpoint, "transformer.pt")
    print("Model saved to transformer.pt")
    # 已连续运行的epoch: