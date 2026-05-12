# SPDX-License-Identifier: Apache-2.0

"""Drop-in replacement for timm.models.vision_transformer.Mlp using
comfy.ops.manual_cast.Linear so weight/input dtype mismatches are
handled the comfy-native way.

State-dict keys (fc1.weight, fc1.bias, fc2.weight, fc2.bias) match
timm's exactly, so existing checkpoints load without remapping.
"""

import torch.nn as nn

_OPS = None
def _ops():
    global _OPS
    if _OPS is None:
        import comfy.ops
        _OPS = comfy.ops.manual_cast
    return _OPS


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias1, bias2 = (bias, bias) if isinstance(bias, bool) else bias
        drop1, drop2 = (drop, drop) if isinstance(drop, (int, float)) else drop

        self.fc1 = _ops().Linear(in_features, hidden_features, bias=bias1)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop1)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = _ops().Linear(hidden_features, out_features, bias=bias2)
        self.drop2 = nn.Dropout(drop2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
