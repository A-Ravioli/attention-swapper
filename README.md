# attention-swapper

An easy way to swap out all the newest attention blocks in PyTorch.

## Overview

The `attention_swapper` package provides a simple API to swap out different implementations of attention mechanisms in any PyTorch model. A global registry holds a variety of attention implementations such as Scaled Dot Product, Flash, Linear, Performer, Efficient, Sparse, Gated, Relative, Dynamic, Entmax, Adaptive, Local, and more. You can also register your own custom attention modules using the provided decorator.

## Installation

To install the package locally in editable mode, run:

```bash
pip install -e .
```

## Usage

You can use the `swap_attention` function from the package to replace attention blocks in your model. For example:

```python
import torch
import torch.nn as nn
from attention_swapper.core import swap_attention

# Define a dummy attention block
class DummyAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        return v

# Define a dummy model that uses the attention block
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = DummyAttention()

    def forward(self, x):
        return self.attn(x, x, x)

if __name__ == '__main__':
    model = DummyModel()
    print("Before swap:", model.attn.__class__.__name__)

    # Swap out DummyAttention with a registered attention implementation (e.g., 'flash_attention')
    model = swap_attention(model, DummyAttention, 'flash_attention', dropout=0.1)
    print("After swap:", model.attn.__class__.__name__)

    # Generate dummy data with shape (batch, num_heads, seq_len, dim)
    q = k = v = torch.randn(1, 1, 5, 10)
    output = model.attn(q, k, v)
    print("Output shape:", output.shape)
```

## Available Implementations

The package comes with several pre-registered attention implementations:

- `scaled_dot_product` (default) via `ScaledDotProductAttention`
- `flash_attention` via `FlashAttention`
- `linear_attention` via `LinearAttention`
- `performer_attention` via `PerformerAttention`
- `efficient_attention` via `EfficientAttention`
- `sparse_attention` via `SparseAttention`
- `gated_attention` via `GatedAttention`
- `relative_attention` via `RelativeAttention`
- `dynamic_attention` via `DynamicAttention`
- `entmax_attention` via `EntmaxAttention`
- `adaptive_attention` via `AdaptiveAttention`
- `local_attention` via `LocalAttention`

You can swap your model's attention block by passing the target attention block class and the desired implementation name along with any additional constructor arguments.

## Extensibility

To add your own attention mechanisms, simply create a new class that implements your attention logic and register it using the `@register_attention("your_attention_name")` decorator. The new implementation will automatically be available in the global registry.

## License

This project is licensed under the MIT License.
