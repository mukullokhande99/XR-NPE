# posit_quant.py
import torch
import torch.nn as nn

try:
    from qtorch_plus import Posit
    from qtorch_plus.quant import quantizer, posit_quantize
except Exception as e:
    raise ImportError(
        "qtorch-plus is required. Install with `pip install qtorch-plus` "
        "and set TORCH_EXTENSIONS_DIR/CUDA_HOME if needed."
    ) from e

def _normalize_es(es: int) -> int:
    # Qtorch+ requires 1 <= es <= 3
    if es is None or es <= 0:
        return 1
    return min(int(es), 3)

def posit_number(bits: int, es: int):
    return Posit(nsize=bits, es=_normalize_es(es))

class PositActQuant(nn.Module):
    def __init__(self, bits=8, es=1, enabled=True, rounding='nearest'):
        super().__init__()
        self.enabled = enabled
        self.bits = bits
        self.es = _normalize_es(es)
        self.rounding = rounding
        self.q = quantizer(
            forward_number = posit_number(self.bits, self.es),
            forward_rounding = rounding,
            backward_number = None,
            backward_rounding = 'nearest'
        )
    def enable(self, flag=True):
        self.enabled = flag
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if hasattr(self.q, "to") and x.device != next(self.q.parameters(), torch.empty(0)).device:
            self.q.to(x.device)
        return self.q(x)

class PositWeightQuantFn(nn.Module):
    def __init__(self, bits=8, es=1, enabled=True, rounding='nearest'):
        super().__init__()
        self.enabled = enabled
        self.bits = bits
        self.es = _normalize_es(es)
        self.rounding = rounding
        self.q = quantizer(
            forward_number = posit_number(self.bits, self.es),
            forward_rounding = rounding,
            backward_number = None,
            backward_rounding = 'nearest'
        )
    def enable(self, flag=True):
        self.enabled = flag
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return w
        if hasattr(self.q, "to") and w.device != next(self.q.parameters(), torch.empty(0)).device:
            self.q.to(w.device)
        return self.q(w)

# Helpers for OptimLP
def make_weight_quant_fn(bits=8, es=1, rounding='nearest'):
    es = _normalize_es(es)
    return lambda x: posit_quantize(x, nsize=bits, es=es, rounding=rounding)
def make_grad_quant_fn(bits=8, es=1, rounding='nearest'):
    es = _normalize_es(es)
    return lambda x: posit_quantize(x, nsize=bits, es=es, rounding=rounding)
def make_momentum_quant_fn(bits=16, es=1, rounding='nearest'):
    es = _normalize_es(es)
    return lambda x: posit_quantize(x, nsize=bits, es=es, rounding=rounding)
