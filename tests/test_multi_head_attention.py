import pytest
import torch

from attention.multi_head_attention import (
    MultiHeadAttention,
    MultiHeadAttentionNaive,
)
from utils import check_closeness

torch.manual_seed(42)


def sync_weights(custom_mha, torch_mha):
    """Copy weights from the custom MHA to the PyTorch MHA."""
    # PyTorch's MHA combines Q, K, V projections into one matrix
    q_w = custom_mha.q_proj_layer._weights.T
    k_w = custom_mha.k_proj_layer._weights.T
    v_w = custom_mha.v_proj_layer._weights.T
    in_proj_weight = torch.cat([q_w, k_w, v_w], dim=0)

    q_b = custom_mha.q_proj_layer._bias
    k_b = custom_mha.k_proj_layer._bias
    v_b = custom_mha.v_proj_layer._bias
    in_proj_bias = torch.cat(
        [q_b.squeeze(), k_b.squeeze(), v_b.squeeze()], dim=0
    )

    torch_mha.in_proj_weight = torch.nn.Parameter(in_proj_weight)
    torch_mha.in_proj_bias = torch.nn.Parameter(in_proj_bias)
    torch_mha.out_proj.weight = torch.nn.Parameter(
        custom_mha.out_proj_layer._weights.T
    )
    torch_mha.out_proj.bias = torch.nn.Parameter(
        custom_mha.out_proj_layer._bias.squeeze()
    )


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, num_heads",
    [
        (2, 10, 12, 4),
        (4, 5, 16, 8),
    ],
)
def test_mha_naive(batch_size, seq_len, d_model, num_heads):
    """
    Test MultiHeadAttentionNaive against torch.nn.MultiheadAttention for 
    self-attention.
    """
    inputs = torch.randn(batch_size, seq_len, d_model)

    # Custom Naive MHA
    custom_mha_naive = MultiHeadAttentionNaive(
        d_model=d_model,
        num_heads=num_heads,
    )

    # PyTorch MHA
    torch_mha = torch.nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        batch_first=True,
    )

    sync_weights(custom_mha_naive, torch_mha)

    # Forward pass
    output_custom = custom_mha_naive(inputs, inputs, inputs)
    output_torch, _ = torch_mha(inputs, inputs, inputs)

    assert check_closeness(
        output_custom.detach(), output_torch.detach()
    ), "Self-attention outputs do not match for Naive MHA"


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, num_heads",
    [
        (2, 10, 12, 4),
        (4, 5, 16, 8),
    ],
)
def test_mha(batch_size, seq_len, d_model, num_heads):
    """
    Test the optimized MultiHeadAttention against torch.nn.MultiheadAttention 
    for self-attention.
    """
    inputs = torch.randn(batch_size, seq_len, d_model)

    # Custom Optimized MHA
    custom_mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    # PyTorch MHA
    torch_mha = torch.nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        batch_first=True,
    )

    sync_weights(custom_mha, torch_mha)

    # Forward pass
    output_custom = custom_mha(inputs, inputs, inputs)
    output_torch, _ = torch_mha(inputs, inputs, inputs)

    assert check_closeness(
        output_custom.detach(), output_torch.detach()
    ), "Self-attention outputs do not match for Optimized MHA"
