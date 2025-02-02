import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_swapper.registry import register_attention


@register_attention("efficient_attention")
class EfficientAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        # q, k, v are expected to be of shape [batch, num_heads, seq_len, dim]
        scale = q.size(-1) ** 0.5
        # Use einsum for batch matrix multiplication
        attn = torch.einsum('bhlk,bhmk->bhlm', q, k) / scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        output = torch.einsum('bhlm,bhmk->bhlk', attn, v)
        return output 