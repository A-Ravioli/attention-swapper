import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention_swapper.registry import register_attention


@register_attention("relative_attention")
class RelativeAttention(nn.Module):
    def __init__(self, max_relative_position=8):
        super().__init__()
        self.max_relative_position = max_relative_position
        # Create a learnable relative bias of shape [2*max_relative_position + 1]
        self.relative_bias = nn.Parameter(torch.zeros(2 * max_relative_position + 1))

    def forward(self, q, k, v, mask=None):
        # q, k, v are expected to be of shape [B, H, L, D]
        B, H, L, D = q.shape
        scale = math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Compute relative position biases
        positions = torch.arange(L, device=q.device)
        rel_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # [L, L]
        # Clamp the relative positions to the max range
        rel_positions_clamped = rel_positions.clamp(-self.max_relative_position, self.max_relative_position)
        # Shift to index range: [0, 2*max_relative_position]
        bias_indices = rel_positions_clamped + self.max_relative_position
        relative_bias = self.relative_bias[bias_indices]  # [L, L]

        # Add the relative bias to scores
        scores = scores + relative_bias.unsqueeze(0).unsqueeze(0)  # [B, H, L, L]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output 