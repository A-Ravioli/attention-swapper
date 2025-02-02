import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_swapper.registry import register_attention


@register_attention("linear_attention")
class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, q, k, v, mask=None):
        # q, k, v are expected to be of shape [B, H, L, D]
        # Apply a simple kernel function: ELU + 1 to ensure non-negativity
        q_prime = F.elu(q) + 1
        k_prime = F.elu(k) + 1
        
        # Compute numerator: Q' dot (K'^T dot V)
        num = torch.matmul(q_prime, torch.matmul(k_prime.transpose(-2, -1), v))
        
        # Compute denominator: Q' dot (sum of K' across sequence dimension)
        k_sum = k_prime.transpose(-2, -1).sum(dim=-1, keepdim=True)  # [B, H, D, 1]
        denom = torch.matmul(q_prime, k_sum) + self.eps  # [B, H, L, 1]
        
        output = num / denom
        return output 