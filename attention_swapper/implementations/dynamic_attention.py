import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_swapper.registry import register_attention


@register_attention("dynamic_attention")
class DynamicAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable scaling parameter
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output 