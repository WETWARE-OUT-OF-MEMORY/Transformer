import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from functools import partial

import os
import re
from typing import List
import sentencepiece as spm
import sacrebleu
import yaml
import datetime

from transformer.token_bucket_batch_sampler import TokenBucketBatchSampler as Sampler
from transformer.model import Transformer, generate
from transformer.scheduler import TransformerLRScheduler
from transformer.paths import run_dir, find_latest_ckpt


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


def load_xml_data(file_path: str) -> List[str]:
    """从 IWSLT XML 文件提取 <seg> 文本（用于验证/测试集）"""
    seg_title = r'<seg id="\d+">.*?</seg>'
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if re.match(seg_title, line.strip()):
                data.append(re.sub(r"<.*?>", '', line).strip())
    return data


def bpe_tokenize(sp_model, texts_en, texts_de):
    # emb_en, emb_de = [], []
    # for en, de in zip(texts_en, texts_de):
    #     emb_en.append(sp_model.encode(en, out_type=int))
    #     emb_de.append(sp_model.encode(de, out_type=int))
    emb_en = sp_model.encode(texts_en)
    emb_de = sp_model.encode(texts_de)
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

def smoothed_loss(logits: Tensor, targets: Tensor, pad_index: int, exclude_ids: Tensor,
                  epsilon: float = 0.1):
    # logits: [batch * n, VOCAB_SZ]
    # target: tgt[batch * n]
    # pad_index: PAD_ID, 用于忽略batch * n中补齐的pad
    # exclude_ids: [PAD_ID, BOS_ID, EOS_ID, UNK_ID], 排除VOCAB_SZ中的特殊token id

    # 标记非pad位，丢弃pad位
    # mask = (targets != ignore_index)
    # logits = logits[mask]
    # targets = targets[mask]
    #
    # # 构造one-hot，正确token处为1，其他为0
    # true_dist = torch.zeros_like(logits)
    # true_dist.scatter_(1, targets.unsqueeze(1), 1.0)
    #
    # # 构造均匀分布部分
    # smooth_dist = torch.zeros_like(logits)
    # smooth_dist[:, smooth_indices] = 1.0 / len(smooth_indices)
    # # 混合得到目标分布
    # target_dist = true_dist * (1.0 - epsilon) + smooth_dist * epsilon
    # # 取对数概率，返回交叉熵
    # log_probs = F.log_softmax(logits, dim=-1)
    # return -(target_dist * log_probs).sum(dim=-1).sum()

    log_prob = F.log_softmax(logits, dim=-1)
    # nll: [batch * n], 每个token取到正确token id的对数概率取负
    nll = -log_prob.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    # excl: [batch * n], 每个token取到exclude_ids的对数概率和
    excl = log_prob[:, exclude_ids].sum(-1)
    # smooth_num: VOCAB_SZ - 特殊token id数 - 正确token id数(1)
    smooth_num = logits.shape[-1] - exclude_ids.shape[-1] - 1
    # smooth: [batch * n], 每个token取到其他token id的对数概率和取负
    smooth = -(log_prob.sum(-1) - excl + nll) / smooth_num
    per_token = (1 - epsilon) * nll + epsilon * smooth
    return (per_token * (targets != pad_index)).sum()


@torch.no_grad()
def validate(model, sp, device, dev_ids_en, dev_ids_de, dev_texts_en, dev_texts_de,
             vocab_sz, pad_id, exclude_ids, max_length, run_bleu=True):
    """每个 epoch 末在 dev2010 上验证。

    返回 (per-token 验证 loss, dev2010 BLEU 或 None)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    bs = 64
    for i in range(0, len(dev_ids_en), bs):
        src = pad_sequence(
            [torch.tensor(e, dtype=torch.long) for e in dev_ids_en[i:i + bs]],
            batch_first=True, padding_value=pad_id).to(device)
        tgt = pad_sequence(
            [torch.tensor(d, dtype=torch.long) for d in dev_ids_de[i:i + bs]],
            batch_first=True, padding_value=pad_id).to(device)
        src_pad = (src == pad_id)
        tgt_pad = (tgt == pad_id)

        y = model(src, tgt[:, :-1], src_pad, tgt_pad[:, :-1])
        loss = smoothed_loss(y.reshape(-1, vocab_sz), tgt[:, 1:].reshape(-1),
                             pad_index=pad_id, exclude_ids=exclude_ids, epsilon=0.1)
        total_loss += loss.item()
        total_tokens += int((tgt[:, 1:] != pad_id).sum())
    val_loss = total_loss / total_tokens

    bleu = None
    if run_bleu:
        hypo = [generate(model, sp, s, max_length, device) for s in dev_texts_en]
        bleu = sacrebleu.corpus_bleu(hypo, [dev_texts_de]).score

    model.train()
    return val_loss, bleu

# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # 配置 & 数据加载
    # ============================================================
    with open("configs.yaml", "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    DATA_PATH = configs["DATA"]["ROOT"]
    train_en = load_data(DATA_PATH + "/train.tags.en-de.en")
    train_de = load_data(DATA_PATH + "/train.tags.en-de.de")

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
    MAX_LENGTH = configs["MODEL"]["MAX_LENGTH"]

    beta1 = configs["LR"]["BETA1"]
    beta2 = configs["LR"]["BETA2"]
    l_eps = configs["LR"]["L_EPS"]
    WARMUP = configs["LR"]["WARMUP"]

    EPOCHS = configs["TRAIN"]["EPOCHS"]
    BATCH_SIZE = configs["TRAIN"]["BATCH_SIZE"]
    MAX_BATCH_LENGTH = configs["TRAIN"]["MAX_BATCH_LENGTH"]
    MAX_TOKENS = configs["TRAIN"]["MAX_TOKENS"]

    RUN_BLEU = configs["VAL"].get("RUN_BLEU", True)
    MAX_GENERATE_LENGTH = configs["TEST"]["MAX_GENERATE_LENGTH"]

    MODEL_TYPE = configs["BPE"]["MODEL_TYPE"]
    NUM_THREADS = configs["BPE"]["NUM_THREADS"]
    VOCAB_SZ = configs["BPE"]["VOCAB_SIZE"]
    CHARACTER_COVERAGE = configs["BPE"]["CHARACTER_COVERAGE"]
    MERGE_SZ = configs["BPE"]["MERGE_SIZE"]
    BOS_ID = configs["BPE"]["BOS_ID"]
    PAD_ID = configs["BPE"]["PAD_ID"]
    EOS_ID = configs["BPE"]["EOS_ID"]
    UNK_ID = configs["BPE"]["UNK_ID"]

    # ============================================================
    # BPE 模型训练（仅首次）
    # ============================================================
    if not configs["BPE"].get("IS_TRAINED", False):
        with open("bpe_corpus.txt", "w", encoding="utf-8") as f:
            for en_sent, de_sent in zip(train_en, train_de):
                f.write(en_sent + "\n")
                f.write(de_sent + "\n")

        spm.SentencePieceTrainer.train(
            input="bpe_corpus.txt",
            model_prefix="bpe_shared",
            vocab_size=VOCAB_SZ,
            character_coverage=CHARACTER_COVERAGE,
            model_type=MODEL_TYPE,
            num_threads=NUM_THREADS,
            pad_id=PAD_ID,
            unk_id=UNK_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
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

    # ============================================================
    # 验证集 dev2010（每个 epoch 末监控过拟合与翻译质量）
    # ============================================================
    dev_en_texts = load_xml_data(DATA_PATH + "/IWSLT17.TED.dev2010.en-de.en.xml")
    dev_de_texts = load_xml_data(DATA_PATH + "/IWSLT17.TED.dev2010.en-de.de.xml")
    dev_en_ids = [[BOS_ID] + t + [EOS_ID] for t in sp.encode(dev_en_texts, out_type=int)]
    dev_de_ids = [[BOS_ID] + t + [EOS_ID] for t in sp.encode(dev_de_texts, out_type=int)]
    print(f"dev2010 validation set: {len(dev_en_texts)} sentences")

    # single_emb_*: 单句训练对
    # medium_emb_*: 5句合并为中句训练对
    single_emb_en, single_emb_de = bpe_tokenize(sp, train_en, train_de)
    medium_emb_en, medium_emb_de = merge_short_sentences(
        single_emb_en, single_emb_de, MERGE_SZ)
    # 所有单句和所有中句并入训练数据
    # 中句为按步长为5划分，不含起始位移位情况
    emb_en = single_emb_en + medium_emb_en
    emb_de = single_emb_de + medium_emb_de

    # 添加BOS、EOS
    for i in range(len(emb_en)):
        emb_en[i] = [BOS_ID] + emb_en[i] + [EOS_ID]
    for i in range(len(emb_de)):
        emb_de[i] = [BOS_ID] + emb_de[i] + [EOS_ID]

    idx = list(range(len(emb_en)))

    # def create_key(emb_en, emb_de):
    #     def key_func(i):
    #         return max(len(emb_en[i]), len(emb_de[i]))
    #     return key_func
    # idx.sort(key=create_key(emb_en, emb_de))

    lengths = [max(len(s), len(t)) for s, t in zip(emb_en, emb_de)]

    sampler = Sampler(
        lengths,
        max_tokens=MAX_TOKENS,  # 单批补齐后 token 上限，由显存决定
        max_sentences=256,  # 防止全是短句时一批塞进上千条
        pool_factor=100,
        seed=42,
    )

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
        max_length=MAX_LENGTH,
    ).to(device)

    optimizer = torch.optim.Adam(
        params=tf.parameters(), betas=(beta1, beta2), eps=l_eps)

    scaler = GradScaler()

    lr_scheduler = TransformerLRScheduler(
        optimizer=optimizer, d_model=D_MODEL, warmup_steps=WARMUP)

    # smooth_indices = [i for i in range(VOCAB_SZ)
    #                   if i != PAD_ID and i != BOS_ID and i != EOS_ID]
    # smooth_indices = torch.tensor(smooth_indices, device=device)
    exclude_ids = torch.tensor([PAD_ID, BOS_ID, EOS_ID, UNK_ID], device=device)

    # ============================================================
    # DataLoader
    # ============================================================
    dataset = TranslationDataset(emb_en, emb_de)
    # 注意：已启用Sampler，不能再传 batch_size / shuffle / drop_last
    loader = DataLoader(
        dataset,
        collate_fn=partial(collate_fn, pad_id=PAD_ID),
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        batch_sampler=sampler,
    )

    # ============================================================
    # 训练循环
    # ============================================================
    OUT_DIR = run_dir()

    start_epoch = 0
    ckpt_path, _ = find_latest_ckpt()
    if ckpt_path is not None:
        print(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        tf.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("current_epoch", 0)
        print(f"Resumed at epoch {start_epoch}, "
              f"step_num={lr_scheduler.step_num}")
    else:
        print("No checkpoint found, starting from scratch")

    # 训练循环开始前（checkpoint 恢复之后）
    log_path = os.path.join(
        OUT_DIR, f"train_log_{datetime.now().strftime('%H%M%S')}.txt")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)   # 行缓冲，崩溃不丢已写行

    for epoch in range(start_epoch, start_epoch + EPOCHS):
        # 每个epoch重建划分
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        cnt = 0
        epoch_tokens = 0
        cur_batch_sz = 0
        for batch in loader:
            src, tgt, src_pad_mask, tgt_pad_mask = batch

            src = src.to(device)
            tgt = tgt.to(device)
            src_pad_mask = src_pad_mask.to(device)
            tgt_pad_mask = tgt_pad_mask.to(device)

            with autocast(dtype=torch.bfloat16):
                # y: [batch, n, VOCAB_SZ]
                y = tf(src, tgt[:, :-1], src_pad_mask, tgt_pad_mask[:, :-1])

            loss = smoothed_loss(
                y.reshape(-1, VOCAB_SZ),
                tgt[:, 1:].reshape(-1),
                pad_index=PAD_ID,
                exclude_ids=exclude_ids,
                epsilon=0.1,
            )

            # 只计入去除BOS的decoder输入的token数
            token_num = (tgt[:, 1:] != PAD_ID).sum().item()
            numerical_loss = loss.item()

            # loss.backward()
            scaler.scale(loss).backward()

            cnt += 1
            cur_batch_sz += token_num
            epoch_tokens += token_num
            if cnt % 50 == 0:
                print(f"Epoch No: {epoch}, Batch No: {cnt}, loss: {numerical_loss / token_num:.4f}")

                log_file.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"epoch={epoch} batch={cnt} "
                    f"loss_per_token={numerical_loss / token_num:.4f} "
                    f"tokens={token_num}\n"
                )

            epoch_loss += numerical_loss

            if cur_batch_sz > MAX_BATCH_LENGTH or cnt == len(loader):

                # 累计等价扩大batch的总token数，在反向传播前更新梯度
                # 除以cur_batch_sz，使不同batch的每个token等权重
                for p in tf.parameters():
                    if p.grad is not None:
                        p.grad /= cur_batch_sz

                lr_scheduler.step()

                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()
                cur_batch_sz = 0

        print(f"Epoch: {epoch}; Average loss per token: {epoch_loss / epoch_tokens:.4f}")

        # ============================================================
        # 每个 epoch 末验证 dev2010：teacher-forcing 前向 loss + 生成 BLEU
        # ============================================================
        val_loss, bleu = validate(
            tf, sp, device, dev_en_ids, dev_de_ids, dev_en_texts, dev_de_texts,
            VOCAB_SZ, PAD_ID, exclude_ids, MAX_GENERATE_LENGTH, RUN_BLEU)
        if bleu is not None:
            print(f"Epoch: {epoch}; val loss: {val_loss:.4f}; dev2010 BLEU: {bleu:.2f}")
            log_file.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"epoch={epoch} val_loss={val_loss:.4f} dev2010_bleu={bleu:.2f}\n"
            )
        else:
            print(f"Epoch: {epoch}; val loss: {val_loss:.4f}")
            log_file.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"epoch={epoch} val_loss={val_loss:.4f}\n"
            )

        # 每个 epoch 结束后保存 checkpoint
        checkpoint = {
            "model_state_dict": tf.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "current_epoch": epoch + 1,
        }
        torch.save(checkpoint, os.path.join(OUT_DIR, "transformer.pt"))
        print(f"Checkpoint saved at {os.path.join(OUT_DIR, 'transformer.pt')}")


