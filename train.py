import os
from typing import Dict, List

import torch
from torch.amp import autocast, GradScaler
import yaml
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_from_disk

from transformer import Transformer, TransformerLRScheduler

# ============================================================
# 超参数配置
# ============================================================
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
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id)
# scaler = GradScaler()

# ============================================================
# 数据加载
# ============================================================
data_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "wikitext2")

train_data = load_from_disk(os.path.join(data_path, "train"))
train_data = train_data.filter(lambda x: len(x['text'].strip()) > 10)

def tokenize(tokenizer, batch) -> Dict[str, List]:
    return {'input_ids': [tokenizer.encode(x)[:MAX_LENGTH]
                          for x in batch]}

train_data = train_data.map(
    lambda x: tokenize(tokenizer, x['text']),
    batched=True, remove_columns=['text'])

def add_labels(batch):
    return {'labels': batch['input_ids']}

train_data = train_data.map(lambda x: add_labels(x), batched=True)

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


train_loader = DataLoader(
    train_data, collate_fn=collate_fn,
    batch_size=BATCH_SIZE, shuffle=True)

if os.path.exists('transformer.pt'):
    tf.load_state_dict(torch.load('transformer.pt', map_location=device))

# ============================================================
# 训练循环
# ============================================================
torch.autograd.set_detect_anomaly(True)
for epoch in range(EPOCHS):
    epoch_loss = 0
    cnt = 0
    for batch in train_loader:
        if (cnt + 1) % 50 == 0:
            print(f"Batch No: {cnt + 1}")
        src, tgt, src_pad_mask, tgt_pad_mask = batch
        src = src.to(device)
        tgt = tgt.to(device)
        src_pad_mask = src_pad_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)

        optimizer.zero_grad()
        # with autocast(device):
        #     y = tf(src, tgt[:, :-1], src_pad_mask, tgt_pad_mask[:, :-1])
        #     loss = criterion(y.reshape(-1, VOCAB_SZ), tgt[:, 1:].reshape(-1))

        y = tf(src, tgt[:, :-1], src_pad_mask, tgt_pad_mask[:, :-1])
        loss = criterion(y.reshape(-1, VOCAB_SZ), tgt[:, 1:].reshape(-1))

        if torch.isnan(loss):
            for x in batch:
                print(tokenizer.decode(x, skip_special_tokens=True))
                print(y.reshape(-1, VOCAB_SZ))
                print(tgt[:, 1:].reshape(-1))
                input()
        print(loss.item())

        epoch_loss += loss.item()
        cnt += 1

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        # lr_scheduler.step()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print(f"Epoch: {epoch}; Average loss: {epoch_loss / cnt}")

# ============================================================
# 保存模型
# ============================================================
torch.save(tf.state_dict(), "transformer.pt")
print("Model saved to transformer.pt")
