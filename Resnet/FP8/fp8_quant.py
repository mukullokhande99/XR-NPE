import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# FP8 level generator (IEEE-like)
# -----------------------------
def _fp_levels(ebits: int, mbits: int, device=None, dtype=torch.float32):
    """
    Build all finite representable values (exclude +/-inf, NaN) for a floating format with:
      - ebits: number of exponent bits
      - mbits: number of mantissa (fraction) bits
    We assume IEEE-like bias = 2^(ebits-1)-1, include subnormals and a single zero.
    """
    assert 2 <= ebits <= 6, "ebits out of supported range"
    assert 1 <= mbits <= 6, "mbits out of supported range"
    bias = (1 << (ebits - 1)) - 1
    maxE = (1 << ebits) - 1  # all ones

    vals = set()
    # Zero (single 0.0)
    vals.add(0.0)

    # Subnormals: E == 0, F in [1 .. (2^m - 1)]
    for sgn in (-1.0, 1.0):
        for F in range(1, (1 << mbits)):
            frac = F / float(1 << mbits)
            val = sgn * (2.0 ** (1 - bias)) * frac
            vals.add(val)

    # Normalized: E in [1 .. maxE-1], F in [0 .. 2^m - 1]
    for sgn in (-1.0, 1.0):
        for E in range(1, maxE):
            for F in range(0, (1 << mbits)):
                frac = 1.0 + F / float(1 << mbits)
                exp = E - bias
                val = sgn * (2.0 ** exp) * frac
                vals.add(val)

    # Sort and return
    vals = sorted(vals)
    return torch.tensor(vals, device=device, dtype=dtype)


def fp8_e4m3_levels(device=None, dtype=torch.float32):
    # E4M3: 4 exponent bits, 3 mantissa bits (bias = 7)
    return _fp_levels(4, 3, device=device, dtype=dtype)


def fp8_e5m2_levels(device=None, dtype=torch.float32):
    # E5M2: 5 exponent bits, 2 mantissa bits (bias = 15)
    return _fp_levels(5, 2, device=device, dtype=dtype)


class STE(nn.Module):
    def forward(self, x):
        return x + (x.detach() - x).detach()


class FP8Quantizer(nn.Module):
    """
    FP8 fake-quantizer with EMA scaling.
    - mode: 'weight' or 'activation'
    - fmt: 'e4m3' (good for activations) or 'e5m2' (often used for weights)
    - weights: per-output-channel scaling
    - activations: per-tensor scaling
    """
    def __init__(self, mode: str, fmt: str = "e4m3",
                 per_channel: bool = False, ch_axis: int = 0,
                 ema_decay: float = 0.99, enabled: bool = True):
        super().__init__()
        assert mode in ("weight", "activation")
        assert fmt in ("e4m3", "e5m2")
        self.mode = mode
        self.fmt = fmt
        self.per_channel = per_channel if mode == "weight" else False
        self.ch_axis = ch_axis
        self.ema_decay = ema_decay
        self.enabled = enabled

        self.register_buffer("ema_maxabs", torch.tensor(0.0))  # activations per-tensor
        self.register_buffer("ema_maxabs_w", torch.empty(0))   # weights per-channel
        self.register_buffer("calibrated", torch.tensor(False))

        if fmt == "e4m3":
            self.register_buffer("levels", fp8_e4m3_levels())
        else:
            self.register_buffer("levels", fp8_e5m2_levels())

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

    def enable(self, flag=True):
        self.enabled = flag

    def forward(self, x):
        if not self.enabled or (not self.training and not bool(self.calibrated)):
            return x

        # device guard
        if self.levels.device != x.device:
            self.to(x.device)

        self._update_ema_(x)

        levels = self.levels.to(x.device, x.dtype)
        L = int(levels.numel())
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

