import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_swapper.registry import register_attention


@register_attention("adaptive_attention")
class AdaptiveAttention(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        # Learnable parameter to adaptively combine scores
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, q, k, v, mask=None):
        # Compute standard dot-product scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # Adaptive combination: mix with the global average of scores
        adaptive_scores = self.alpha * scores + (1 - self.alpha) * scores.mean(dim=-1, keepdim=True)
        attn = F.softmax(adaptive_scores, dim=-1)
        output = torch.matmul(attn, v)
        return output 