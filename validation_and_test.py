import os
import re

import sacrebleu
import sentencepiece as spm
import torch
import yaml

from transformer.model import Transformer, generate

DATA_ROOT = None


def load_data(file_name: str):
    """从 IWSLT XML 文件提取 <seg> 文本"""
    data_path = os.path.join(DATA_ROOT, file_name)
    seg_title = r'<seg id="\d+">.*?</seg>'
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if re.match(seg_title, line.strip()):
                data.append(re.sub(r"<.*?>", '', line).strip())
    return data


def evaluate(model, tokenizer, files_en, files_de, max_length, device,
             merge=1, tokenize="intl"):
    """对给定文件列表逐句（或 merge 句合并）翻译，计算语料级 BLEU。

    返回 (CorpusBLEU 对象, 假设译文列表, 参考译文列表)
    """
    en_all, de_all = [], []
    for fe, fd in zip(files_en, files_de):
        en_all += load_data(fe)
        de_all += load_data(fd)

    if merge > 1:
        en, de = [], []
        for start in range(0, len(en_all), merge):
            end = min(start + merge, len(en_all))
            en.append(" ".join(en_all[start:end]))
            de.append(" ".join(de_all[start:end]))
    else:
        en, de = en_all, de_all

    hypo = [generate(model, tokenizer, s, max_length, device) for s in en]
    bleu = sacrebleu.corpus_bleu(hypo, [de], tokenize=tokenize)
    print(f"evaluated {len(en)} sentences (merge={merge}): {bleu}")
    return bleu, hypo, de


if __name__ == "__main__":
    with open("configs.yaml", "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    DATA_ROOT = configs["DATA"]["ROOT"]
    MODE = configs["TEST"]["MODE"]
    BLEU_TOKENIZE = configs["TEST"].get("BLEU_TOKENIZE", "intl")
    MAX_GENERATE_LENGTH = configs["TEST"]["MAX_GENERATE_LENGTH"]

    LAYER_NUM = configs["MODEL"]["LAYER_NUM"]
    D_MODEL = configs["MODEL"]["D_MODEL"]
    HEAD = configs["MODEL"]["HEAD"]
    EPS = configs["MODEL"]["EPS"]
    D_FF = configs["MODEL"]["D_FF"]
    MAX_LENGTH = configs["MODEL"]["MAX_LENGTH"]
    DROPOUT = configs["MODEL"]["DROPOUT"]
    VOCAB_SZ = configs["BPE"]["VOCAB_SIZE"]

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

    sp = spm.SentencePieceProcessor()
    sp.load("bpe_shared.model")

    if MODE == "test":
        # 最终上报：tst2010-2012 逐句语料级 BLEU，附分年子分数
        files_en = [f"IWSLT17.TED.tst{i}.en-de.en.xml" for i in range(2010, 2013)]
        files_de = [f"IWSLT17.TED.tst{i}.en-de.de.xml" for i in range(2010, 2013)]
        overall, _, _ = evaluate(tf, sp, files_en, files_de,
                                 MAX_GENERATE_LENGTH, device, tokenize=BLEU_TOKENIZE)
        print(f"tst2010-2012 overall BLEU: {overall.score:.2f}")
        for i in range(3):
            b, _, _ = evaluate(tf, sp, [files_en[i]], [files_de[i]],
                               MAX_GENERATE_LENGTH, device, tokenize=BLEU_TOKENIZE)
            print(f"tst201{0 + i} BLEU: {b.score:.2f}")
    else:
        # 验证：dev2010（训练/调参期间反复使用）
        bleu, _, _ = evaluate(tf, sp,
                              ["IWSLT17.TED.dev2010.en-de.en.xml"],
                              ["IWSLT17.TED.dev2010.en-de.de.xml"],
                              MAX_GENERATE_LENGTH, device, tokenize=BLEU_TOKENIZE)
        print(f"dev2010 BLEU: {bleu.score:.2f}")
