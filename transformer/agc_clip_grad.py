

class AGCGradClip:
    def __init__(self, clip_factor=0.1, eps=1e-3):
        self.clip_factor = clip_factor
        self.eps = eps

    def clip(self, parameters):
        for p in parameters:
            if p.grad is None:
                continue
            p_norm = p.norm(2)
            g_norm = p.grad.norm(2)
            if p_norm > self.eps and g_norm > self.eps:
                max_g_norm = self.clip_factor * p_norm
                if g_norm > max_g_norm:
                    p.grad.mul_(max_g_norm / (g_norm + self.eps))