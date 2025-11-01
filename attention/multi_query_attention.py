import torch
from torch import Tensor

from attention.projection import Projection


class MultiQueryAttention(torch.nn.Module):
    """
    Implements the Multi-Query Attention mechanism.
    Reference: "Fast Transformers with Multi-Query Attention" (https://arxiv.org/abs/1911.02150)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_q: int | None = None,
        add_bias: bool = True,
        causal_mask: bool = False,
    ):
        super().__init__()
        self.d_model: int = d_model  # Store as int, not tensor
        self.num_heads: int = num_heads
        self.add_bias: bool = add_bias
        self.causal_mask: bool = causal_mask

        self.dim_q: int = dim_q if dim_q is not None else d_model
        self.dim_k: int = self.dim_q // num_heads
        self.dim_v: int = self.dim_q // num_heads

        if not (self.dim_q % num_heads == 0):
            raise ValueError(
                "Total dimensions for Q must be divisible by num_heads."
            )

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
            in_features=self.dim_v * self.num_heads,
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
        # Shape: (batch, seq_len, d_model/num_heads)
        v_proj = self.v_proj_layer(inputs_v)
        # Shape: (batch, seq_len, d_model/num_heads)
        batch_size, seq_len, _ = q_proj.shape

        # Reshape queries for multi-head attention
        q_proj = q_proj.view(
            batch_size, seq_len, self.num_heads, self.dim_k
        ).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len, d_model/num_heads)

        logits = torch.einsum("bhnk, bmk -> bhnm", q_proj, k_proj)
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
        # Multiply attention weights by values
        output = torch.einsum("bhnm, bmv -> bhnv", attention, v_proj)
        # Shape: (batch, num_heads, seq_len, d_model/num_heads)

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.dim_v * self.num_heads)
        )

        # Final output projection
        output = self.out_proj_layer(output)
        return output
