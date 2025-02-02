ATTENTION_REGISTRY = {}


def register_attention(name):
    """Decorator to register a new attention implementation with a given name."""
    def decorator(cls):
        ATTENTION_REGISTRY[name] = cls
        return cls
    return decorator 