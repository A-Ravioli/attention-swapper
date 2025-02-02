import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_swapper.registry import register_attention


@register_attention("gated_attention")
class GatedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable gate parameter as a scalar
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, q, k, v, mask=None):
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, v)
        # Gated combination: mix attention output and original v using a sigmoid gate
        gate = torch.sigmoid(self.gate)
        return gate * attn_output + (1 - gate) * v 