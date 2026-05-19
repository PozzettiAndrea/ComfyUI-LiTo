import torch
import torch.nn as nn
from . import SparseTensor

_OPS = None
def _ops():
    global _OPS
    if _OPS is None:
        import comfy.ops
        _OPS = comfy.ops.manual_cast
    return _OPS


__all__ = [
    'SparseLinear'
]


class SparseLinear(_ops().Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))
