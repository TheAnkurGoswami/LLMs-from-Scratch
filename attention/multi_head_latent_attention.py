import torch
from torch import Tensor

from attention.projection import Projection
from attention.scaled_dot_product_attention import ScaledDotProductAttention
from positional_encoding.rotary import RotaryPositionalEncoding


class MultiHeadLatentAttention(torch.nn.Module):
    """
    Implements the Multi-Head Latent Attention mechanism.
    Reference: DeepSeek-V2: A Strong, Economical, and
        Efficient Mixture-of-Experts Language Model
        (https://arxiv.org/pdf/2405.04434)
    """

    def __init__(
        self,
        d_model: int,
        q_latent_dim: int,
        kv_latent_dim: int,
        num_heads: int,
        dim_q: int | None = None,
        dim_k: int | None = None,
        dim_v: int | None = None,
        add_bias: bool = True,
        causal_mask: bool = False,
    ):
        super().__init__()
        self.d_model: int = d_model  # Store as int, not tensor
        self.q_latent_dim = q_latent_dim
        self.kv_latent_dim = kv_latent_dim
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
        self.head_dim = self.dim_q // num_heads

        self.q_down_proj_layer = Projection(
            in_features=self.d_model,
            out_features=self.q_latent_dim,
            add_bias=False,
        )

        self.kv_down_proj_layer = Projection(
            in_features=self.d_model,
            out_features=self.kv_latent_dim,
            add_bias=False,
        )

        self.q_up_proj_layer = Projection(
            in_features=self.q_latent_dim,
            out_features=self.dim_q,
            add_bias=False,
        )

        self.k_up_proj_layer = Projection(
            in_features=self.kv_latent_dim,
            out_features=self.dim_k,
            add_bias=False,
        )

        self.v_up_proj_layer = Projection(
            in_features=self.kv_latent_dim,
            out_features=self.dim_v,
            add_bias=False,
        )

        self.q_rope_proj_layer = Projection(
            in_features=self.q_latent_dim,
            out_features=self.dim_q,
            add_bias=False,
        )
        self.k_rope_proj_layer = Projection(
            in_features=self.d_model,
            out_features=self.head_dim,
            add_bias=False,
        )

        self.out_proj_layer = Projection(
            in_features=self.dim_v,
            out_features=self.d_model,
            add_bias=add_bias,
        )

    def decoupled_rope(self, q_latent, inputs_k):
        q_rope_proj = self.q_rope_proj_layer(q_latent)
        k_rope_proj = self.k_rope_proj_layer(inputs_k)
        # q_rope_proj: (batch_size, seq_len, dim_q)
        # k_rope_proj: (batch_size, seq_len, head_dim)

        q_rope = RotaryPositionalEncoding(d_model=self.d_model)(q_rope_proj)
        k_rope = RotaryPositionalEncoding(d_model=self.head_dim)(k_rope_proj)
        # q_rope shape: (batch_size, seq_len, dim_q)
        # k_rope shape: (batch_size, seq_len, head_dim)
        return q_rope, k_rope

    def forward(
        self, inputs_q: Tensor, inputs_k: Tensor, inputs_v: Tensor
    ) -> Tensor:
        # Low-rank key-value joint compression
        # inputs shape (batch_size, seq_len, model_dims)
        q_latent = self.q_down_proj_layer(inputs_q)
        # NOTE: Query compression does not affect KV cache performance.
        # It is used solely during training to reduce activation memory.
        kv_latent = self.kv_down_proj_layer(inputs_k)
        # q_latent: (batch_size, seq_len, q_latent_dim)
        # kv_latent: (batch_size, seq_len, kv_latent_dim)

        q_proj = self.q_up_proj_layer(q_latent)
        k_proj = self.k_up_proj_layer(kv_latent)
        v_proj = self.v_up_proj_layer(kv_latent)

        # Projections Shape
        # q_proj: (batch_size, seq_len, dim_q)
        # k_proj: (batch_size, seq_len, dim_k)
        # v_proj: (batch_size, seq_len, dim_v)

        q_rope, k_rope = self.decoupled_rope(q_latent, inputs_k)

        # Apply Rotary Positional Encoding
        batch, seq_len, _ = q_proj.shape
        q_proj = q_proj.view(batch, seq_len, self.num_heads, self.head_dim)
        k_proj = k_proj.view(batch, seq_len, self.num_heads, self.head_dim)
        v_proj = v_proj.view(batch, seq_len, self.num_heads, self.head_dim)

        q_rope = q_rope.view(batch, seq_len, self.num_heads, self.head_dim)
        k_rope = k_rope.view(batch, seq_len, 1, self.head_dim)

        q_new = torch.concat((q_proj, q_rope), dim=-1)
        k_rope_shared = k_rope.expand(-1, -1, self.num_heads, -1)
        k_new = torch.concat((k_proj, k_rope_shared), dim=-1)

        q_new = q_new.transpose(1, 2).contiguous()
        k_new = k_new.transpose(1, 2).contiguous()
        v_proj = v_proj.transpose(1, 2).contiguous()
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        q_new = q_new.view(batch * self.num_heads, seq_len, -1)
        k_new = k_new.view(batch * self.num_heads, seq_len, -1)
        v_proj = v_proj.view(batch * self.num_heads, seq_len, self.head_dim)

        outputs = ScaledDotProductAttention().forward(
            q_proj=q_new,
            k_proj=k_new,
            v_proj=v_proj,
            causal_mask=self.causal_mask,
        )  # Shape: (batch * num_heads, seq_len, head_dim)

        outputs = outputs.view(batch, self.num_heads, seq_len, self.head_dim)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs.view(
            batch, seq_len, self.d_model  # OR head_dim * self.num_heads
        )

        # Apply the final linear projection
        out = self.out_proj_layer.forward(outputs)
        return out
