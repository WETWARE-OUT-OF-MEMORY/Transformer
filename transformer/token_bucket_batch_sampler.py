import random
from torch.utils.data import Sampler


class TokenBucketBatchSampler(Sampler):
    """
    分桶 + token 预算的 batch 采样器。

    每个 epoch 的流程：
      1. 全局打乱          -> 决定 pool 成员，这是随机性的来源
      2. pool 内按长度排序  -> 稳定排序，等长样本保留上一步的乱序
      3. 按 token 预算切 batch
      4. 打乱 batch 顺序
    """

    def __init__(self, lengths, max_tokens, max_sentences=None,
                 pool_factor=100, seed=0):
        self.lengths = lengths                    # 每条样本的 max(src_len, tgt_len)
        self.max_tokens = max_tokens              # 单个 batch 补齐后的 token 上限
        self.max_sentences = max_sentences or 10 ** 9
        # avg_len: 样本平均长度
        avg_len = sum(lengths) / len(lengths)
        # est_batch: 1batch内预计的样本数
        est_batch = max(1, int(max_tokens / avg_len))
        # pool_factor: 1pool预计能装多少batch
        # pool_size: 1pool预计有多少条样本
        self.pool_size = est_batch * pool_factor
        self.seed = seed
        self.epoch = 0
        self._batches = None

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._batches = None                      # 失效缓存，下次迭代重新划分

    def _build(self):
        # 打乱全局每条样本的 max(src_len, tgt_len) 列表
        g = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        g.shuffle(idx)
        # pool内排序、划分batch
        batches = []
        for i in range(0, len(idx), self.pool_size):
            # sorted 是稳定排序，保留g.shuffle()的随机性
            pool = sorted(idx[i:i + self.pool_size], key=lambda j: self.lengths[j])
            cur, cur_max = [], 0
            for j in pool:
                # nxt_max: 当前最长长度
                nxt_max = max(cur_max, self.lengths[j])
                # 条数 * 本批最长长度 = 补齐pad后的总token数
                # if - 当前批次新增后超出单batch最大token限制或者是最后一条
                if cur and ((len(cur) + 1) * nxt_max > self.max_tokens
                            or len(cur) + 1 > self.max_sentences):
                    # 截断batch，导致超限的句子顺延到下一batch
                    batches.append(cur)
                    cur, cur_max = [j], self.lengths[j]
                else:
                    cur.append(j)
                    cur_max = nxt_max
            if cur:
                batches.append(cur)

        # 打乱batch顺序
        g.shuffle(batches)
        self._batches = batches

    def __iter__(self):
        if self._batches is None:
            self._build()
        return iter(self._batches)

    def __len__(self):
        if self._batches is None:
            self._build()
        return len(self._batches)