import torch


class KeyValueCaching(torch.nn.Module):
    def __init__(self, caching_tensor_names: list[str] | None = None):
        super().__init__()
        for tensor_name in caching_tensor_names:
            setattr(self, tensor_name, None)

    def update(self, **kwargs):
        for tensor_name, tensor in kwargs.items():
            cached_tensor = getattr(self, tensor_name, None)
            if cached_tensor is None:
                # prefill stage
                setattr(self, tensor_name, tensor)
            else:
                # decoding stage
                new_cached_tensor = torch.cat([cached_tensor, tensor], dim=1)
                setattr(self, tensor_name, new_cached_tensor)

        return tuple(
            getattr(self, tensor_name)
            for tensor_name in kwargs.keys())