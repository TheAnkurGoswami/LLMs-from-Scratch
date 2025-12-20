import torch
from torch import Tensor

from attention.projection import Projection
from attention.scaled_dot_product_attention import ScaledDotProductAttention
from inference.kv_caching import KeyValueCaching


class MultiHeadAttentionNaive(torch.nn.Module):
    """
    Implements the Multi-Head Attention mechanism.

    Multi-Head Attention allows the model to jointly attend to information from
    different representation subspaces at different positions. It consists of
    several parallel attention layers, or "heads".

    The mechanism involves three main steps:
    1.  **Linear Projections**: The input queries, keys, and values are
        linearly projected `num_heads` times with different, learned linear
        projections.
    2.  **Scaled Dot-Product Attention**: For each head, Scaled Dot-Product
        Attention is applied in parallel to the projected queries, keys, and
        values.
    3.  **Concatenation and Final Projection**: The outputs of the attention
        heads are concatenated and then passed through a final linear
        projection to produce the final output.

    This implementation supports different dimensions for queries, keys, and
    values, and allows for causal masking.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_q: int | None = None,
        dim_k: int | None = None,
        dim_v: int | None = None,
        add_bias: bool = True,
        causal_mask: bool = False,
    ):
        """
        Initializes the MultiHeadAttention module.

        Args:
            d_model (int): The dimensionality of the input and output.
            num_heads (int): The number of parallel attention heads.
            dim_q (int, optional): The dimensionality of the queries. If None,
                defaults to `d_model`.
            dim_k (int, optional): The dimensionality of the keys. If None,
                defaults to `d_model`.
            dim_v (int, optional): The dimensionality of the values. If None,
                defaults to `d_model`.
            add_bias (bool, optional): Whether to include a bias term in the
                linear projections. Defaults to True.
            causal_mask (bool, optional): If True, applies a causal mask to
                prevent attention to future positions. Defaults to False.
        """
        super().__init__()
        self.d_model: int = d_model  # Store as int, not tensor
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
            in_features=self.dim_v,
            out_features=self.d_model,
            add_bias=add_bias,
        )

    def get_head_size(self, param: str) -> int:
        """
        Calculates the size of a single attention head for a given parameter.

        Args:
            param (str): The parameter for which to calculate the head size.
                         Can be "query", "key", or "value".

        Returns:
            int: The size of a single attention head.
        """
        match param:
            case "query":
                return self.dim_q // self.num_heads
            case "key":
                return self.dim_k // self.num_heads
            case "value":
                return self.dim_v // self.num_heads

    def forward(
        self, inputs_q: Tensor, inputs_k: Tensor, inputs_v: Tensor
    ) -> Tensor:
        """
        Forward pass for the Multi-Head Attention mechanism.

        Args:
            inputs (Tensor): The input tensor of shape
                (batch_size, seq_len, d_model).

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """
        # Project the input into queries, keys, and values
        q_proj = self.q_proj_layer(inputs_q)
        k_proj = self.k_proj_layer(inputs_k)
        v_proj = self.v_proj_layer(inputs_v)

        head_outputs = []

        # Process each attention head in parallel
        for head_ix in range(self.num_heads):
            sliced_projs = []
            for param, proj_mat in zip(
                ["query", "key", "value"],
                [q_proj, k_proj, v_proj],
                strict=False,
            ):
                head_size = self.get_head_size(param)
                start, end = (head_size * head_ix, head_size * (head_ix + 1))
                sliced_projs.append(proj_mat[:, :, start:end])

            # Apply scaled dot-product attention for the current head
            head_output = ScaledDotProductAttention().forward(
                *sliced_projs, causal_mask=self.causal_mask
            )
            head_outputs.append(head_output)

        # Concatenate the outputs of all heads
        concat_heads = torch.concat(head_outputs, dim=-1)

        # Apply the final linear projection
        out = self.out_proj_layer.forward(concat_heads)
        return out


class MultiHeadAttention(MultiHeadAttentionNaive):
    """
    Implements Multi-Head Attention mechanism.
    Reference: "Attention is All You Need" (Vaswani et al., 2017)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_q: int | None = None,
        dim_k: int | None = None,
        dim_v: int | None = None,
        add_bias: bool = True,
        causal_mask: bool = False,
        allow_kv_caching: bool = False,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            dim_q=dim_q,
            dim_k=dim_k,
            dim_v=dim_v,
            add_bias=add_bias,
            causal_mask=causal_mask,
        )
        self.allow_kv_caching = allow_kv_caching
        
        if allow_kv_caching:
            self.kv_cache = KeyValueCaching()

    def forward(
        self, inputs_q: Tensor, inputs_k: Tensor, inputs_v: Tensor
    ) -> Tensor:
        # Project the input into queries, keys, and values
        q_proj = self.q_proj_layer(inputs_q)
        k_proj = self.k_proj_layer(inputs_k)
        v_proj = self.v_proj_layer(inputs_v)

        batch, seq_len, model_dims = q_proj.shape
        head_dim = model_dims // self.num_heads

        """
        Idea is to convert (batch, seq_len, model_dim) to 
        (batch, seq_len, num_heads, head_dim) which can abbreviated as 
        bsd -> bnhd.
        A fine trick is to reshape to (batch * num_heads, seq_len, head_dim)
        so that we can perform attention in parallel for all heads. 
        Since, softmax is applied along seq_len dimension, moving num_heads to 
        batch dimension makes sense.
        """
        q_proj = q_proj.view(batch, seq_len, self.num_heads, head_dim)
        k_proj = k_proj.view(batch, seq_len, self.num_heads, head_dim)
        v_proj = v_proj.view(batch, seq_len, self.num_heads, head_dim)
        # Shape: (batch_size, seq_len, num_heads, head_dim)

        if self.allow_kv_caching:
            k_proj, v_proj = self.kv_cache.update(k_proj, v_proj)

        q_proj = q_proj.transpose(1, 2).contiguous()
        k_proj = k_proj.transpose(1, 2).contiguous()
        v_proj = v_proj.transpose(1, 2).contiguous()
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        q_proj = q_proj.view(batch * self.num_heads, seq_len, head_dim)
        k_proj = k_proj.view(batch * self.num_heads, seq_len, head_dim)
        v_proj = v_proj.view(batch * self.num_heads, seq_len, head_dim)
        # Shape: (batch_size * num_heads, seq_len, head_dim)

        outputs = ScaledDotProductAttention().forward(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            causal_mask=self.causal_mask,
        )  # Shape: (batch * num_heads, seq_len, head_dim)

        # Reshape back to (batch, seq_len, model_dim)
        outputs = outputs.view(batch, self.num_heads, seq_len, head_dim)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs.view(
            batch, seq_len, model_dims  # OR head_dim * self.num_heads
        )

        # Apply the final linear projection
        out = self.out_proj_layer.forward(outputs)
        return out
