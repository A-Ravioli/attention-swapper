import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_swapper.registry import register_attention


@register_attention("local_attention")
class LocalAttention(nn.Module):
    def __init__(self, window_size=3, temperature=1.0):
        super().__init__()
        self.window_size = window_size
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        # q, k, v: shape [B, H, L, D]
        B, H, L, D = q.shape
        # Compute full scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        # Create local mask: allowed if |i-j| <= window_size
        idx = torch.arange(L, device=q.device)
        diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))  # shape [L, L]
        local_mask = (diff <= self.window_size).to(q.dtype)
        
        # Apply the local mask by setting disallowed positions to -infinity
        scores = scores + (1 - local_mask) * (-1e9)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output 