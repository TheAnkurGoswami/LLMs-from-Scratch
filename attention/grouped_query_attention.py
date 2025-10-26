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
        dim_k: int | None = None,
        dim_v: int | None = None,
        n_groups: int = 1,
        add_bias: bool = True,
        causal_mask: bool = False,
    ):
        super().__init__()
        self.d_model: int = d_model  # Store as int, not tensor
        self.num_heads: int = num_heads
        self.n_groups: int = n_groups
        self.add_bias: bool = add_bias
        self.causal_mask: bool = causal_mask

        if self.dim_q % num_heads:
            raise ValueError(
                "Total dimensions for Q must be divisible by num_heads."
            )
        # If specific dimensions for Q, K, V are not provided,
        # default them to d_model
        self.dim_q: int = dim_q if dim_q is not None else d_model
        self.head_dim = self.dim_q // num_heads
        self.dim_k: int = self.head_dim * n_groups
        self.dim_v: int = self.dim_k

        if num_heads % n_groups:
            raise ValueError("num_heads must be divisible by n_groups.")
        self.n_heads_per_group = num_heads // n_groups
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
            self.n_groups,
            self.n_heads_per_group,
            seq_len,
            self.head_dim,
        )
        # Shape: (batch, seq_len, d_model) ->
        # (batch, num_heads, seq_len, head_dim) ->
        # (batch, n_groups, n_heads_per_group, seq_len, head_size)

        k_proj = k_proj.view(batch_size, self.n_groups, seq_len, self.head_dim)
        # Shape: (batch, seq_len, dim_k) ->
        # (batch, num_groups, seq_len, head_dim)

        logits = torch.einsum("bgpmd, bgnd -> bgpmn", q_proj, k_proj)

        # Scale
        # Mask

        # Apply softmax to get attention weights
        attention = torch.softmax(logits, dim=-1)

        v_proj = v_proj.view(batch_size, self.n_groups, seq_len, self.head_dim)
        # Shape: (batch, seq_len, dim_v) ->
        # (batch, num_groups, seq_len, head_dim)

        # Multiply attention weights by values
        output = torch.einsum("bgpmn, bgnd -> bgpmd", attention, v_proj)

        #  Convert gpd (n_groups * n_heads_per_group * head_dim) back to original dim.
        output = output.view(
            batch_size,
            seq_len,
            self.head_dim * self.n_groups * self.n_heads_per_group,
        )
        # Final output projection
        output = self.out_proj_layer(output)
        return output
