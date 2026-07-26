import torch

class AutoClipGradNorm:
    def __init__(self, percentile=90, buffer_size=100, min_clip=0.1):
        self.percentile = percentile
        self.buffer = []
        self.buffer_size = buffer_size
        self.min_clip = min_clip

    def clip(self, parameters):
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=float('inf'))
        self.buffer.append(total_norm.item())
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        if len(self.buffer) >= 10:
            threshold = max(
                self.min_clip,
                sorted(self.buffer)[int(len(self.buffer) * self.percentile / 100)]
            )
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=threshold)
        return total_norm