import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial

import os
from typing import List
import sentencepiece as spm
import yaml

from transformer.model import Transformer
from transformer.scheduler import TransformerLRScheduler


# ============================================================
# 工具函数
# ============================================================
def load_data(file_path: str) -> List[str]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence[0] == "<":
                continue
            data.append(sentence)
    return data


def bpe_tokenize(sp_model, texts_en, texts_de):
    emb_en, emb_de = [], []
    for en, de in zip(texts_en, texts_de):
        emb_en.append(sp_model.encode(en, out_type=int))
        emb_de.append(sp_model.encode(de, out_type=int))
    return emb_en, emb_de


def merge_short_sentences(emb_en: List[List], emb_de: List[List], merge_sz: int):
    combine_en, combine_de = [], []
    for start in range(0, len(emb_en), merge_sz):
        end = min(start + merge_sz, len(emb_en))
        combine_en.append(
            [token for group in emb_en[start:end] for token in group])
        combine_de.append(
            [token for group in emb_de[start:end] for token in group])
    return combine_en, combine_de

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_data[idx], dtype=torch.long),
            torch.tensor(self.tgt_data[idx], dtype=torch.long),
        )

def smoothed_loss(logits: Tensor, targets: Tensor,
                  ignore_index: int, smooth_indices: Tensor,
                  epsilon: float = 0.1):
    mask = (targets != ignore_index)
    logits = logits[mask]
    targets = targets[mask]

    true_dist = torch.zeros_like(logits)
    true_dist.scatter_(1, targets.unsqueeze(1), 1.0)

    smooth_dist = torch.zeros_like(logits)
    smooth_dist[:, smooth_indices] = 1.0 / len(smooth_indices)

    target_dist = true_dist * (1.0 - epsilon) + smooth_dist * epsilon
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1).mean()

def collate_fn(batch, pad_id):
    src_batch, tgt_batch = [], []
    for src, tgt in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

    src_pad_mask = (src_batch == pad_id)
    tgt_pad_mask = (tgt_batch == pad_id)

    return src_batch, tgt_batch, src_pad_mask, tgt_pad_mask

# class Collator:
#     def __init__(self, pad_id):
#         self.pad_id = pad_id
#
#     def collate_fn(self, batch):
#         src_batch, tgt_batch = [], []
#         for src, tgt in batch:
#             src_batch.append(src)
#             tgt_batch.append(tgt)
#         src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.pad_id)
#         tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=self.pad_id)
#
#         src_pad_mask = (src_batch == self.pad_id)
#         tgt_pad_mask = (tgt_batch == self.pad_id)
#
#         return src_batch, tgt_batch, src_pad_mask, tgt_pad_mask
#
#     def __call__(self, batch):
#         return self.collate_fn(batch)

# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # 配置 & 数据加载
    # ============================================================
    DATA_PATH = "D:/Learn/machine_learning/data/iwslt2017-en-de/en-de"
    train_en = load_data(DATA_PATH + "/train.tags.en-de.en")
    train_de = load_data(DATA_PATH + "/train.tags.en-de.de")

    with open("configs.yaml", "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    # ============================================================
    # BPE 模型训练（仅首次）
    # ============================================================
    bpe_cfg = configs["BPE"]
    if not bpe_cfg.get("IS_TRAINED", False):
        with open("bpe_corpus.txt", "w", encoding="utf-8") as f:
            for en_sent, de_sent in zip(train_en, train_de):
                f.write(en_sent + "\n")
                f.write(de_sent + "\n")

        spm.SentencePieceTrainer.train(
            input="bpe_corpus.txt",
            model_prefix="bpe_shared",
            vocab_size=bpe_cfg["VOCAB_SIZE"],
            character_coverage=bpe_cfg["CHARACTER_COVERAGE"],
            model_type=bpe_cfg["MODEL_TYPE"],
            num_threads=bpe_cfg["NUM_THREADS"],
            pad_id=bpe_cfg["PAD_ID"],
            unk_id=bpe_cfg["UNK_ID"],
            bos_id=bpe_cfg["BOS_ID"],
            eos_id=bpe_cfg["EOS_ID"],
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
        )
        configs["BPE"]["IS_TRAINED"] = True
        with open("configs.yaml", "w", encoding="utf-8") as f:
            yaml.dump(configs, f, allow_unicode=True, sort_keys=False)

    # ============================================================
    # 加载 BPE 模型 & 分词
    # ============================================================
    sp = spm.SentencePieceProcessor()
    sp.load("bpe_shared.model")

    emb_en, emb_de = bpe_tokenize(sp, train_en, train_de)
    combine_en, combine_de = merge_short_sentences(
        emb_en, emb_de, bpe_cfg["MERGE_SIZE"])

    # ============================================================
    # 超参数
    # ============================================================
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    LAYER_NUM = configs["MODEL"]["LAYER_NUM"]
    D_MODEL = configs["MODEL"]["D_MODEL"]
    HEAD = configs["MODEL"]["HEAD"]
    EPS = configs["MODEL"]["EPS"]
    D_FF = configs["MODEL"]["D_FF"]
    DROPOUT = configs["MODEL"]["DROPOUT"]
    beta1 = configs["LR"]["BETA1"]
    beta2 = configs["LR"]["BETA2"]
    l_eps = configs["LR"]["L_EPS"]
    WARMUP = configs["LR"]["WARMUP"]
    EPOCHS = configs["TRAIN"]["EPOCHS"]
    BATCH_SIZE = configs["TRAIN"]["BATCH_SIZE"]
    VOCAB_SZ = bpe_cfg["VOCAB_SIZE"]
    PAD_ID = bpe_cfg["PAD_ID"]
    EOS_ID = bpe_cfg["EOS_ID"]

    # ============================================================
    # 设备 & 模型
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Current device:", device)

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
        params=tf.parameters(), betas=(beta1, beta2), eps=l_eps)
    lr_scheduler = TransformerLRScheduler(
        optimizer=optimizer, d_model=D_MODEL, warmup_steps=WARMUP)

    smooth_indices = [i for i in range(VOCAB_SZ)
                      if i != PAD_ID and i != EOS_ID]
    smooth_indices = torch.tensor(smooth_indices, device=device)

    # ============================================================
    # DataLoader
    # ============================================================
    dataset = TranslationDataset(combine_en, combine_de)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=PAD_ID),
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
    )

    # ============================================================
    # 训练循环
    # ============================================================
    start_epoch = 0
    if os.path.exists("transformer.pt"):
        print("Resuming from checkpoint...")
        checkpoint = torch.load("transformer.pt", map_location=device)
        tf.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("current_epoch", 0)
        print(f"Resumed at epoch {start_epoch}, "
              f"step_num={lr_scheduler.step_num}")

    # batch size等效扩大倍数
    accumulate_factor = 5
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        epoch_loss = 0.0
        cnt = 0
        for batch in loader:
            src, tgt, src_pad_mask, tgt_pad_mask = batch
            src = src.to(device)
            tgt = tgt.to(device)
            src_pad_mask = src_pad_mask.to(device)
            tgt_pad_mask = tgt_pad_mask.to(device)

            y = tf(src, tgt[:, :-1], src_pad_mask, tgt_pad_mask[:, :-1])

            loss = smoothed_loss(
                y.reshape(-1, VOCAB_SZ),
                tgt[:, 1:].reshape(-1),
                ignore_index=PAD_ID,
                smooth_indices=smooth_indices,
                epsilon=0.1,
            )

            numerical_loss = loss.item()
            loss /= accumulate_factor
            loss.backward()

            if (cnt + 1) % 50 == 0:
                print(f"Epoch No: {epoch}, Batch No: {cnt + 1}, loss: {numerical_loss:.4f}")

            epoch_loss += numerical_loss

            if (cnt + 1) % accumulate_factor == 0 or (cnt + 1) == len(loader):
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
            cnt += 1

        print(f"Epoch: {epoch}; Average loss: {epoch_loss / cnt:.4f}")

        # 每个 epoch 结束后保存 checkpoint
        checkpoint = {
            "model_state_dict": tf.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "current_epoch": epoch + 1,
        }
        torch.save(checkpoint, "transformer.pt")
        print(f"Checkpoint saved at epoch {epoch + 1}")
