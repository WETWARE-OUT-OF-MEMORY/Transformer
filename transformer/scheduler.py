class TransformerLRScheduler:
    """Transformer 学习率调度器：warmup + 衰减策略"""

    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),                 # 衰减阶段
            self.step_num * self.warmup ** (-1.5)    # 预热阶段
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
