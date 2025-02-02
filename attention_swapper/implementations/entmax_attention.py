import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention_swapper.registry import register_attention


def sparsemax(input, dim=-1):
    # This is a simple implementation of sparsemax (a special case of entmax with alpha=2)
    # Based on: From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification
    # input: tensor of shape [*, n]
    # Sort input in descending order along dim
    input_sorted, _ = torch.sort(input, descending=True, dim=dim)
    input_sorted_cumsum = input_sorted.cumsum(dim=dim)

    rhos = torch.arange(1, input.size(dim)+1, device=input.device, dtype=input.dtype)
    # Reshape rhos for proper broadcasting
    view_shape = [1] * input.dim()
    view_shape[dim] = -1
    rhos = rhos.view(view_shape)

    support = (input_sorted * rhos) > (input_sorted_cumsum - 1);
    support_size = support.sum(dim=dim, keepdim=True)

    # Gather cumulative sum at the last index of support
    input_sorted_support = input_sorted_cumsum.gather(dim, support_size.long()-1)
    tau = (input_sorted_support - 1) / support_size.to(input.dtype)
    output = torch.clamp(input - tau, min=0)
    return output


@register_attention("entmax_attention")
class EntmaxAttention(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        # q, k, v are expected to be of shape [B, H, L, D]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # Instead of softmax, use sparsemax (a case of entmax with alpha=2)
        attn = sparsemax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output 