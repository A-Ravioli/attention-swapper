import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_swapper.registry import register_attention


@register_attention("causal_attention")
class CausalAttention(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        # q, k, v are expected to be of shape [B, H, L, D]
        B, H, L, D = q.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        # Create a causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output 