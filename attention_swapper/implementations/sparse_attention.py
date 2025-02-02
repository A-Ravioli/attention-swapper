import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_swapper.registry import register_attention


@register_attention("sparse_attention")
class SparseAttention(nn.Module):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def forward(self, q, k, v, mask=None):
        # Compute raw attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        # Zero out attention values below threshold
        attn = torch.where(attn < self.threshold, torch.tensor(0.0, device=attn.device), attn)
        # Renormalize attention weights
        attn_sum = attn.sum(dim=-1, keepdim=True) + 1e-6
        attn = attn / attn_sum
        output = torch.matmul(attn, v)
        return output 