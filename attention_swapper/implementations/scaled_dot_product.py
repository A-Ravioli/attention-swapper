import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_swapper.registry import register_attention


@register_attention("scaled_dot_product")
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        # q, k, v are expected to be of shape [batch, num_heads, seq_len, dim]
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output 