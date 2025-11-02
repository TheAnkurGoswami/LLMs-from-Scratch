import numpy as np
import pytest
import torch

from attention.multi_head_attention import MultiHeadAttention
from attention.scaled_dot_product_attention import ScaledDotProductAttention
from utils import check_closeness

torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)
np.random.seed(69)
torch.manual_seed(69)


@pytest.mark.parametrize(
    "batch_size, causal_mask, d_model, seq_len",
    [
        (1, True, 4, 5),
        (2, False, 8, 10),
        (4, True, 16, 20),
    ],
)
def test_scaled_dot_product_attention(
    batch_size, causal_mask, d_model, seq_len
):

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    sdpa_cus = ScaledDotProductAttention()
    output_cus = sdpa_cus.forward(q, k, v, causal_mask=causal_mask)

    sdpa_pt = torch.nn.functional.scaled_dot_product_attention
    output_pt = sdpa_pt(q, k, v, is_causal=causal_mask)

    assert check_closeness(
        output_cus.detach().numpy(), output_pt.detach().numpy())


@pytest.mark.parametrize(
    "batch_size, d_model, n_heads, seq_len",
    [
        (1, 4, 2, 5),
        (2, 8, 4, 10),
        (4, 16, 8, 20),
    ],
)
def test_multi_head_attention(batch_size, d_model, n_heads, seq_len):
    dim_kqv = d_model

    x = torch.randint(
        low=0,
        high=10,
        size=(batch_size, seq_len, d_model),
        dtype=torch.float32,
    )

    mha_cus = MultiHeadAttention(
        d_model=d_model,
        num_heads=n_heads,
        dim_q=dim_kqv,
        dim_k=dim_kqv,
        dim_v=dim_kqv,
    )
    all_proj_wt = []
    all_proj_bias = []
    for projection in [
        mha_cus.q_proj_layer,
        mha_cus.k_proj_layer,
        mha_cus.v_proj_layer,
    ]:
        all_proj_wt.append(projection._weights.T)
        all_proj_bias.append(projection._bias)

    all_proj_wt = torch.cat(all_proj_wt, dim=0)
    all_proj_bias = torch.cat(all_proj_bias, dim=-1)

    in_proj_wt = torch.vstack(list(all_proj_wt))
    in_proj_bias = torch.hstack(list(all_proj_bias))

    mha_pt = torch.nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=n_heads,
        batch_first=True,
    )

    mha_pt.in_proj_weight = torch.nn.Parameter(in_proj_wt.detach().clone())
    mha_pt.in_proj_bias = torch.nn.Parameter(in_proj_bias.detach().clone())
    mha_pt.out_proj.weight = torch.nn.Parameter(
        mha_cus.out_proj_layer._weights.T.detach().clone()
    )
    mha_pt.out_proj.bias = torch.nn.Parameter(
        mha_cus.out_proj_layer._bias.detach().clone()
    )
    output_cus = mha_cus.forward(x, x, x)
    output_pt, _ = mha_pt.forward(x, x, x, need_weights=True)

    assert check_closeness(
        output_cus.detach().numpy(), output_pt.detach().numpy())
