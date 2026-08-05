# Transformer

从零实现 *Attention Is All You Need*（Vaswani et al., 2017），在 IWSLT2017 英德数据集上
完成机器翻译的训练与评估。注意力、LayerNorm、位置编码、学习率调度、标签平滑损失
均为手写实现，未使用 `nn.Transformer` 或 `nn.MultiheadAttention`。

## 分支说明

本仓库维护两个长期分支，对应本地开发与云端训练两种运行环境。

**main（本地实现版本）**

- 本地开发与调试用的实现分支
- 面向本地开发机（RTX 2060 6GB）的全 FP32 训练
- 显存配置较为保守：`MAX_TOKENS: 4096`、`MAX_BATCH_LENGTH: 4096`
- 每次功能开发先在本地验证，通过后再同步到云端分支

**Online-Server-Version（云服务器运行版本）**

- 在云端 GPU 实例（RTX 4090 24GB）上训练运行的分支
- 针对大显存放宽 batch 配置：`MAX_TOKENS: 12000`
- 启用 BF16 混合精度（AMP）加速训练
- 手写 LayerNorm 增加 FP32 保精度处理
- 训练 loss 追加写入 `train_log.txt`，支持事后分析
- 2026/08/04 云端完成 40 epoch 训练，tst2010 短句集 BLEU = 20.35

## 模型规模

| 项 | 值 |
|----|-----|
| 参数量 | 60.5M |
| d_model | 512 |
| 层数 | Encoder 6 + Decoder 6 |
| 注意力头数 | 8 |
| d_ff | 2048 |
| 词表 | 32,000（英德共享 BPE） |

## 目录结构

```
transformer/
├── attention.py                  多头注意力，含 KV Cache
├── layers.py                     FFN / LayerNorm / 输出投影层
├── encoder.py                    EncoderBlock / Encoder
├── decoder.py                    DecoderBlock / Decoder
├── model.py                      Transformer 主模型 + 自回归生成
├── scheduler.py                  Noam 学习率调度
├── token_bucket_batch_sampler.py 分桶 + token 预算的动态组批
├── paths.py                      output 日期目录与 checkpoint 查找
├── auto_clip_grad.py             AutoClip 梯度裁剪（可选，当前停用）
└── agc_clip_grad.py              AGC 自适应梯度裁剪（可选，当前停用）

IWSLT_train.py                    训练入口
validation_and_test.py            BLEU 评估
configs.yaml                      超参数配置
output/                           训练输出（按日期分目录）
```

## 对齐论文的部分

- Post-LN 残差结构
- 缩放点积注意力 `QK^T / √d_k` 与多头拼接
- 正弦位置编码，按 `MAX_LENGTH` 预生成并注册为 buffer
- Embedding 输出乘 `√d_model`
- 三向权重共享：`src_emb` / `tgt_emb` / pre-softmax 线性层
- Adam `β=(0.9, 0.98)`，`ε=1e-9`
- Noam 调度：`d_model^-0.5 · min(step^-0.5, step · warmup^-1.5)`
- Label Smoothing `ε=0.1`
- Dropout 作用于注意力权重、子层残差、Embedding 与位置编码之和

## 论文之外的工程实现

**KV Cache** — 生成时 self-attention 缓存历史 K/V 并增量拼接，cross-attention 的 K/V
在首步计算后全程复用，避免每步重算 encoder 侧投影。

**动态组批** — `TokenBucketBatchSampler` 按「条数 × 批内最长长度 ≤ MAX_TOKENS」切分批次。
每个 epoch 先全局打乱决定 pool 成员，再在 pool 内稳定排序使等长样本保留乱序，
最后打乱 batch 顺序。相比定长 batch 显著减少 padding 占比。

**梯度累积** — 累计到 `MAX_BATCH_LENGTH` 个 token 后，梯度统一除以实际累积的
token 数再更新，使每个 token 等权重，在有限显存下逼近论文的大 batch 设定。

**显存优化的标签平滑** — 利用目标分布「一个尖峰加一片均匀」的结构，用 `gather`
替代 one-hot 矩阵、用「全词表求和减去被排除项」替代均匀分布矩阵，把同时存活的
`[N, V]` 张量从 7 份降到 2 份。PAD 行的屏蔽下沉到 `[N]` 空间完成，不引入额外拷贝。

**断点续训** — checkpoint 保存模型权重、优化器状态、调度器 `step_num` 与 epoch 计数。

## 数据处理

IWSLT2017 En-De。SentencePiece 训练 32k 英德共享 BPE 词表，`character_coverage=0.9995`
以覆盖德语变音字符。训练数据混合两种粒度：原始单句，以及每 5 句合并的长文本，
使模型同时见到短句和长上下文。

评估分两档：`tst2010` 作为短句测试，`tst2011`/`tst2012` 每 5 句合并作为长文本测试，
用 sacrebleu 计算 BLEU。

## 使用

```bash
# 训练。首次运行会自动训练 BPE 模型并写回 configs.yaml
python IWSLT_train.py

# BLEU 评估
python validation_and_test.py
```

超参数集中在 `configs.yaml`。`MAX_TOKENS` 需按显存调整，本地 6GB 显存下建议 2048，
云端 24GB 显存下建议 12000。

```bash
# 云端训练切换到 Online-Server-Version 分支
git checkout Online-Server-Version
```

## 依赖

torch, sentencepiece, sacrebleu, pyyaml

---

# 更新与调整

## 2026/08/05

**输出管理与评估**

- 新增 `transformer/paths.py`：训练输出统一归档到 `output/YYYYMMDD/`，
  `find_latest_ckpt()` 按日期目录自动定位最新 checkpoint，支持多轮训练留档
- `IWSLT_train.py` checkpoint 与训练日志迁入 `output/日期/`，日志文件名带
  时间戳，避免覆盖历史记录
- `validation_and_test.py` 新增 `save_predictions()`，将源句/译文/参考及
  逐句 BLEU 写入 TSV（按句 BLEU 升序排列，最差排最前），便于对照分析

**解码采样**

- `transformer/model.py` 新增 `sample_token()` 并接入 `generate()`，实现
  temperature、top-k、top-p 三种采样策略；当前参数 `temperature=1, top_k=0,
  top_p=0` 等效贪心解码，采样开关参数待接入配置

## 2026/08/04

**云端训练（Online-Server-Version）**

- 在 RTX 4090 24GB 云实例上完成首轮 40 epoch 训练
- `MAX_TOKENS` 提升至 12000，充分利用大显存
- 启用 BF16 混合精度（AMP）+ GradScaler，等效算力约 50 TFLOPS
- 手写 LayerNorm 进出转 FP32 保精度
- 训练 loss 追加写入 `train_log.txt`
- tst2010（单句）BLEU = 20.35，loss 与 BLEU 自洽

## 2026/08/02

**任务与数据**

- 训练任务从 WikiText-2 复制切换为 IWSLT2017 英德翻译
- 引入 SentencePiece BPE，32k 英德共享词表
- 训练数据改为单句与 5 句合并文本的双粒度混合
- 新增 `TokenBucketBatchSampler`，按 token 预算动态组批
- 新增 `MAX_TOKENS` 配置项，由显存预算决定单批 token 上限

**模型**

- 位置编码改为按 `MAX_LENGTH` 预生成并 `register_buffer`，不再每次前向重算
- 补全三向权重共享，`src_emb` 与 `tgt_emb` 一并绑定到输出层
- 实现 KV Cache：self-attention 增量拼接，cross-attention 首步缓存后复用
- FFN 内部 dropout 移除，仅保留论文明确列出的三处 dropout
- FFN 第一层初始化改为 `xavier_uniform_(gain=√2)`，补偿 ReLU 的方差衰减

**损失函数**

- 标签平滑改为 `gather` 加求和的等价形式，`[N, V]` 张量峰值从 7 份降至 2 份
- 修正 PAD 行未被排除的问题，屏蔽下沉到 `[N]` 空间实现，零显存代价
- 均匀项统一为排除正确 token 的约定，分母与求和范围自洽
- `exclude_ids` 加入 `UNK_ID`，并修正构造时的 dtype 与 device

**训练与评估**

- checkpoint 扩充为模型、优化器、调度器状态与 epoch 计数，支持断点续训
- 新增 `validation_and_test.py`，基于 sacrebleu 的 BLEU 评估

## 2026/07/27

- 调整模型训练策略为等效 5 倍 batch size 的 token 数（约 4.5k）
- `DataLoader` 初始化的 `collate_fn` 采用了自实现类与 `functools.partial` 两种实现方式

## 2026/07/26

- 落地 `.ipynb` 为 `.py` 格式
- 优化部分细节以提高训练速度

## 2026/07/07

- Transformer 复现
- 复制任务训练、测试
