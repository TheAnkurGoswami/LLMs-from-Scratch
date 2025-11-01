import torch
from torch import Tensor

from attention.projection import Projection


class GroupedQueryAttention(torch.nn.Module):
    """
    Implements the Grouped Query Attention mechanism.
    Reference: GQA: Training Generalized Multi-Query Transformer Models from
    Multi-Head Checkpoints (https://arxiv.org/abs/2305.13245)

    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_q: int | None = None,
        n_kv_heads: int = 1,
        add_bias: bool = True,
        causal_mask: bool = False,
    ):
        super().__init__()
        self.d_model: int = d_model  # Store as int, not tensor
        self.num_heads: int = num_heads
        self.n_kv_heads: int = n_kv_heads
        self.add_bias: bool = add_bias
        self.causal_mask: bool = causal_mask

        # If specific dimensions for Q, K, V are not provided,
        # default them to d_model
        self.dim_q: int = dim_q if dim_q is not None else d_model
        
        if self.dim_q % num_heads:
            raise ValueError(
                "Total dimensions for Q must be divisible by num_heads."
            )
        
        self.head_dim = self.dim_q // num_heads
        self.dim_k: int = self.head_dim * n_kv_heads
        self.dim_v: int = self.dim_k

        if num_heads % n_kv_heads:
            raise ValueError("num_heads must be divisible by n_kv_heads.")
        self.n_heads_per_group = num_heads // n_kv_heads
        # Mapping from 'h' query heads to 'g' key/value heads.

        self.q_proj_layer = Projection(
            in_features=self.d_model,
            out_features=self.dim_q,
            add_bias=add_bias,
        )
        self.k_proj_layer = Projection(
            in_features=self.d_model,
            out_features=self.dim_k,
            add_bias=add_bias,
        )
        self.v_proj_layer = Projection(
            in_features=self.d_model,
            out_features=self.dim_v,
            add_bias=add_bias,
        )

        self.out_proj_layer = Projection(
            in_features=self.dim_k * self.n_heads_per_group,
            out_features=self.d_model,
            add_bias=add_bias,
        )

    def forward(
        self, inputs_q: Tensor, inputs_k: Tensor, inputs_v: Tensor
    ) -> Tensor:
        # Project the input into queries, keys, and values
        q_proj = self.q_proj_layer(inputs_q)
        # Shape: (batch, seq_len, d_model)
        k_proj = self.k_proj_layer(inputs_k)
        # Shape: (batch, seq_len, dim_k)
        v_proj = self.v_proj_layer(inputs_v)
        # Shape: (batch, seq_len, dim_k)

        batch_size, seq_len, _ = q_proj.shape
        q_proj = q_proj.view(
            batch_size,
            seq_len,
            self.n_kv_heads,
            self.n_heads_per_group,
            self.head_dim,
        ).permute(0, 2, 3, 1, 4)
        # Shape: (batch, seq_len, d_model) ->
        # (batch, num_heads, seq_len, head_dim) ->
        # (batch, n_kv_heads, n_heads_per_group, seq_len, head_size)

        k_proj = k_proj.view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        # Shape: (batch, seq_len, dim_k) ->
        # (batch, n_kv_heads, seq_len, head_dim)

        logits = torch.einsum("bgpmd, bgnd -> bgpmn", q_proj, k_proj)
        logits /= self.dim_k**0.5

        # Apply causal mask if required
        mask = torch.zeros_like(logits)
        if self.causal_mask:
            mask = torch.masked_fill(
                mask,
                mask=torch.ones_like(mask).tril().logical_not(),
                value=-torch.inf,
            )

        logits += mask

        # Apply softmax to get attention weights
        attention = torch.softmax(logits, dim=-1)

        v_proj = v_proj.view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        # Shape: (batch, seq_len, dim_v) ->
        # (batch, num_groups, seq_len, head_dim)

        # Multiply attention weights by values
        output = torch.einsum("bgpmn, bgnd -> bgpmd", attention, v_proj)
        # Shape: (batch, n_kv_heads, n_heads_per_group, seq_len, head_dim)
        output = output.permute(0, 3, 1, 2, 4).contiguous()
        # Shape: (batch, seq_len, n_kv_heads, n_heads_per_group, head_dim)
        # Convert gpd (n_kv_heads * n_heads_per_group * head_dim) back to
        # original dim_q.
        output = output.view(
            batch_size,
            seq_len,
            self.head_dim * self.n_kv_heads * self.n_heads_per_group,
        )
        # Final output projection
        output = self.out_proj_layer(output)
        return output
