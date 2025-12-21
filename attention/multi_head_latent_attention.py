from attention.projection import Projection
import torch
from torch import Tensor

class MultiHeadLatentAttention(torch.nn.module):
    def __init__(
        self,
        d_model: int,
        d_kv_compression: int,
        num_heads: int,
        dim_q: int | None = None,
        dim_k: int | None = None,
        dim_v: int | None = None,
        add_bias: bool = True,
        causal_mask: bool = False,
    ):
        super().__init__()
        self.d_model: int = d_model  # Store as int, not tensor
        self.d_kv_compression = d_kv_compression
        self.num_heads: int = num_heads
        self.add_bias: bool = add_bias
        self.causal_mask: bool = causal_mask

        # If specific dimensions for Q, K, V are not provided,
        # default them to d_model
        self.dim_q: int = dim_q if dim_q is not None else d_model
        self.dim_k: int = dim_k if dim_k is not None else d_model
        self.dim_v: int = dim_v if dim_v is not None else d_model

        if not (
            self.dim_q % num_heads == 0
            and self.dim_k % num_heads == 0
            and self.dim_v % num_heads == 0
        ):
            raise ValueError(
                "Total dimensions for Q, K, V must be divisible by num_heads."
            )
        

        self.q_down_proj_layer = Projection(
            in_features=self.d_model,
            out_features=self.d_kv_compression,
            add_bias=False,
        )

        self.kv_down_proj_layer = Projection(
            in_features=self.d_model,
            out_features=self.d_kv_compression,
            add_bias=False,
        )

        self.q_up_proj_layer = Projection(
            in_features=self.d_kv_compression,
            out_features=dim_q,
            add_bias=False,
        )

        self.k_up_proj_layer = Projection(
            in_features=self.d_kv_compression,
            out_features=dim_k,
            add_bias=False,
        )

        self.v_up_proj_layer = Projection(
            in_features=self.d_kv_compression,
            out_features=dim_v,
            add_bias=False,
        )

        
    def forward(
        self, inputs_q: Tensor, inputs_k: Tensor, inputs_v: Tensor
    ) -> Tensor:
        # inputs shape (batch_size, seq_len, model_dims)
        q_latent = self.q_down_proj_layer(inputs_q)
        kv_latent = self.kv_down_proj_layer(inputs_k)
        # Latents shape
        # q: (batch_size, seq_len, d_kv_compression)
        # kv: (batch_size, seq_len, d_kv_compression)

        projected_q = self.q_up_proj_layer(q_latent)
        projected_k = self.k_up_proj_layer(kv_latent)
        projected_v = self.v_up_proj_layer(kv_latent)

        # Projections Shape
        # q: (batch_size, seq_len, dim_q)
        # k: (batch_size, seq_len, dim_k)
        # v: (batch_size, seq_len, dim_v)
