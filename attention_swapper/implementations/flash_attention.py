import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_swapper.registry import register_attention


@register_attention("flash_attention")
class FlashAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, q, k, v, mask=None):
        # Simulated flash attention using a fused operation placeholder
        attn = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output 