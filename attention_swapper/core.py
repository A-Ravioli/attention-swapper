import torch.nn as nn
from .registry import ATTENTION_REGISTRY


def swap_attention(model: nn.Module, target_class: type, new_attention_name: str, **kwargs):
    """Recursively traverse the model and replace instances of target_class with a new attention implementation.
    
    Args:
        model (nn.Module): The PyTorch model to modify.
        target_class (type): The class of the attention blocks to replace.
        new_attention_name (str): The name of the new attention implementation to use.
        **kwargs: Additional parameters to pass to the new attention implementation constructor.
    
    Returns:
        nn.Module: The modified model.
    """
    if new_attention_name not in ATTENTION_REGISTRY:
        raise ValueError(f"Attention implementation '{new_attention_name}' not registered!")
    new_attention_cls = ATTENTION_REGISTRY[new_attention_name]

    def _swap(module):
        for name, child in module.named_children():
            if isinstance(child, target_class):
                setattr(module, name, new_attention_cls(**kwargs))
            else:
                _swap(child)
    _swap(model)
    return model 