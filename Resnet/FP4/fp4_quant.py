import torch
import torch.nn as nn
import torch.nn.functional as F

def fp4_e2m1_levels(device=None, dtype=torch.float32):
    exponents = [-2, -1, 0, 1]
    mantissas = [1.0, 1.5]
    pos = [0.0]
    for e in exponents:
        for m in mantissas:
            # Drop the largest value (1.5 * 2**1 = 3.0) to keep total at 16
            if not (e == 1 and m == 1.5):
                pos.append(m * (2.0 ** e))
    neg = [-v for v in pos[1:]]
    neg.reverse()
    vals = neg + pos  # 8 neg + (0 + 7 pos) = 16
    return torch.tensor(vals, device=device, dtype=dtype)

class STE(nn.Module):
    def forward(self, x):
        return x + (x.detach() - x).detach()

class FP4Quantizer(nn.Module):
    # FP4 (E2M1) fake-quantizer. See comments inside for details.
    def __init__(self, mode: str, per_channel: bool = False, ch_axis: int = 0,
                 ema_decay: float = 0.99, enabled: bool = True):
        super().__init__()
        assert mode in ("weight", "activation")
        self.mode = mode
        self.per_channel = per_channel if mode == "weight" else False
        self.ch_axis = ch_axis
        self.ema_decay = ema_decay
        self.enabled = enabled

        self.register_buffer("ema_maxabs", torch.tensor(0.0))
        self.register_buffer("ema_maxabs_w", torch.empty(0))
        self.register_buffer("calibrated", torch.tensor(False))
        self.register_buffer("levels", fp4_e2m1_levels())
        self.ste = STE()

    @torch.no_grad()
    def _update_ema_(self, x):
        if self.mode == "activation":
            cur = x.detach().abs().max()
            if not bool(self.calibrated):
                self.ema_maxabs.copy_(cur)
                self.calibrated.fill_(True)
            else:
                self.ema_maxabs.mul_(self.ema_decay).add_(cur * (1 - self.ema_decay))
        else:
            w = x.detach()
            if self.per_channel:
                co = w.shape[self.ch_axis]
                w_perm = w.transpose(0, self.ch_axis).contiguous().view(co, -1)
                cur = w_perm.abs().amax(dim=1)
                if self.ema_maxabs_w.numel() != co:
                    self.ema_maxabs_w = self.ema_maxabs_w.new_zeros(co)
                    self.calibrated.fill_(False)
                if not bool(self.calibrated):
                    self.ema_maxabs_w.copy_(cur)
                    self.calibrated.fill_(True)
                else:
                    self.ema_maxabs_w.mul_(self.ema_decay).add_(cur * (1 - self.ema_decay))
            else:
                cur = w.abs().max()
                if self.ema_maxabs_w.numel() != 1:
                    self.ema_maxabs_w = self.ema_maxabs_w.new_zeros(1)
                    self.calibrated.fill_(False)
                if not bool(self.calibrated):
                    self.ema_maxabs_w[0].copy_(cur)
                    self.calibrated.fill_(True)
                else:
                    self.ema_maxabs_w[0].mul_(self.ema_decay).add_(cur * (1 - self.ema_decay))

    def enable(self, flag: bool = True):
        self.enabled = flag

    def forward(self, x):
        if not self.enabled or (not self.training and not bool(self.calibrated)):
            return x
        if self.levels.device != x.device:
            self.to(x.device)
        self._update_ema_(x)

        levels = self.levels.to(x.device, x.dtype)
        L = levels.numel()
        max_level = levels.abs().max()



        if self.mode == "activation":
            s = (self.ema_maxabs / (max_level + 1e-12)).clamp(min=1e-8)
            x_scaled = x / s
            dist = (x_scaled.unsqueeze(-1) - levels.view([1] * x_scaled.dim() + [L])).abs()
            idx = dist.argmin(dim=-1)
            xq = levels[idx] * s
            return self.ste(xq)
        else:
            if self.per_channel:
                maxabs = self.ema_maxabs_w.clamp(min=1e-12)
                s = (maxabs / (max_level + 1e-12)).clamp(min=1e-8)
                shape = [1] * x.dim()
                shape[self.ch_axis] = -1
                s = s.view(*shape)
                x_scaled = x / s
                dist = (x_scaled.unsqueeze(-1) - levels.view([1] * x_scaled.dim() + [L])).abs()
                idx = dist.argmin(dim=-1)
                xq = levels[idx] * s
                return self.ste(xq)
            else:
                s = (self.ema_maxabs_w[0] / (max_level + 1e-12)).clamp(min=1e-8)
                x_scaled = x / s
                dist = (x_scaled.unsqueeze(-1) - levels.view([1] * x_scaled.dim() + [L])).abs()
                idx = dist.argmin(dim=-1)
                xq = levels[idx] * s
                return self.ste(xq)
