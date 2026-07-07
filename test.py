import torch
import yaml
from transformer.model import *
from transformers import GPT2Tokenizer

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current device: ", device)
tf = Transformer(layer_num=LAYER_NUM, d_model=D_MODEL, src_emb_sz=VOCAB_SZ, tgt_emb_sz=VOCAB_SZ,
                 head=HEAD, eps=EPS, d_ff=D_FF, dropout=DROPOUT).to(device)
tf.load_state_dict(torch.load("transformer.pt", map_location=device))
input = "Chance fights ever on the side of the prudent."
print(generate(tf, tokenizer, input, MAX_LENGTH, device))