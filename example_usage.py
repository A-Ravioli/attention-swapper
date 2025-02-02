import torch
import torch.nn as nn
from attention_swapper.core import swap_attention
from attention_swapper.implementations.scaled_dot_product import ScaledDotProductAttention


class DummyAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        # Dummy implementation: simply return v
        return v


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DummyAttention()

    def forward(self, x):
        # For demonstration, just call the attention module
        return self.attn(x, x, x)


if __name__ == '__main__':
    model = DummyModel()
    print("Before swap:", model.attn.__class__.__name__)
    
    # Swap out DummyAttention with the registered 'scaled_dot_product' attention
    model = swap_attention(model, DummyAttention, 'scaled_dot_product', temperature=2.0)
    print("After swap:", model.attn.__class__.__name__)
    
    # Generate dummy data with shape (batch, num_heads, seq_len, dim)
    q = k = v = torch.randn(1, 1, 5, 10)
    output = model.attn(q, k, v)
    print("Attention output shape:", output.shape) 