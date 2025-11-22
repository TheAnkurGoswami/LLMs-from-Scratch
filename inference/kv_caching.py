import torch


class KeyValueCaching(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj_cache = None
        self.v_proj_cache = None

    def prefill(self, k_proj, v_proj):
        self.k_proj_cache = k_proj
        self.v_proj_cache = v_proj

    def update(self, k_proj, v_proj):
        if self.k_proj_cache is None or self.v_proj_cache is None:
            self.prefill(k_proj, v_proj)
        else:
            self.k_proj_cache = torch.cat([self.k_proj_cache, k_proj], dim=1)
            self.v_proj_cache = torch.cat([self.v_proj_cache, v_proj], dim=1)

        return self.k_proj_cache, self.v_proj_cache
