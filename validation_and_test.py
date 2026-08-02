import torch
import os
import re
import sacrebleu
import yaml
import sentencepiece as spm

from transformer.model import Transformer, generate

# ============================================================
# 工具函数
# ============================================================
DATA_ROOT = "D:/Learn/machine_learning/data/iwslt2017-en-de/en-de"

VALID_FILE_EN = [f"IWSLT17.TED.tst{i}.en-de.en.xml" for i in range(2010, 2013)]
VALID_FILE_DE = [f"IWSLT17.TED.tst{i}.en-de.de.xml" for i in range(2010, 2013)]


def load_data(file_name: str):
    data_path = os.path.join(DATA_ROOT, file_name)
    seg_title = r'<seg id="\d+">.*?</seg>'
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if re.match(seg_title, line.strip()):
                data.append(re.sub(r"<.*?>", '', line).strip())
    return data


def validation(model, tokenizer, max_length: int, device="cuda"):
    """
    在验证集上评估模型 BLEU 分数
    :param model:       Transformer 模型实例
    :param tokenizer:   SentencePieceProcessor 实例
    :param max_length:  生成的最大 token 数
    :param device:      设备字符串
    """
    # 短文本翻译（2010）
    valid_en_short = load_data(VALID_FILE_EN[0])
    valid_de_short = load_data(VALID_FILE_DE[0])
    # 长文本翻译（2011-2013，5 句合并）
    valid_en, valid_de = [], []
    for i in range(1, 3):
        en = load_data(VALID_FILE_EN[i])
        de = load_data(VALID_FILE_DE[i])
        step = 5
        for start in range(0, len(en), step):
            end = min(start + step, len(en))
            valid_en.append(" ".join(en[start:end]))
            valid_de.append(" ".join(de[start:end]))

    model.eval()

    # 短文本
    hypo = [generate(model, tokenizer, en, max_length, device)
            for en in valid_en_short]
    short_bleu = sacrebleu.corpus_bleu(hypo, [valid_de_short])
    print(f"Short sentences' average bleu: {short_bleu}")

    # 长文本
    hypo = [generate(model, tokenizer, en, max_length, device)
            for en in valid_en]
    bleu = sacrebleu.corpus_bleu(hypo, [valid_de])
    print(f"Sentences' average bleu: {bleu}")

    return short_bleu, bleu


if __name__ == "__main__":
    # ============================================================
    # 超参数
    # ============================================================
    with open("configs.yaml", 'r', encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    LAYER_NUM = configs["MODEL"]["LAYER_NUM"]
    D_MODEL = configs["MODEL"]["D_MODEL"]
    HEAD = configs["MODEL"]["HEAD"]
    EPS = configs["MODEL"]["EPS"]
    D_FF = configs["MODEL"]["D_FF"]
    MAX_LENGTH = configs["MODEL"]["MAX_LENGTH"]
    DROPOUT = configs["MODEL"]["DROPOUT"]
    VOCAB_SZ = configs["BPE"]["VOCAB_SIZE"]
    MAX_GENERATE_LENGTH = configs["TEST"]["MAX_GENERATE_LENGTH"]

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

    if os.path.exists("transformer.pt"):
        checkpoint = torch.load("transformer.pt", map_location=device)
        tf.load_state_dict(checkpoint["model_state_dict"])

    # ============================================================
    # 加载 BPE 模型
    # ============================================================
    sp = spm.SentencePieceProcessor()
    sp.load("bpe_shared.model")
    # ============================================================
    # 验证
    # ============================================================
    validation(tf, sp, MAX_GENERATE_LENGTH, device)
