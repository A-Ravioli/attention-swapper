import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_swapper.registry import register_attention


@register_attention("performer_attention")
class PerformerAttention(nn.Module):
    def __init__(self, projection_dim=10):
        super().__init__()
        self.projection_dim = projection_dim
        self.query_proj = nn.Linear(10, projection_dim)
        self.key_proj = nn.Linear(10, projection_dim)

    def forward(self, q, k, v, mask=None):
        # q, k, v are expected to be of shape [batch, num_heads, seq_len, dim]
        B, H, L, D = q.shape
        q_proj = self.query_proj(q.view(-1, D)).view(B, H, L, self.projection_dim)
        k_proj = self.key_proj(k.view(-1, D)).view(B, H, L, self.projection_dim)
        attn = torch.matmul(q_proj, k_proj.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output 