import numpy as np
import pytest
import torch

from attention.multi_head_attention import MultiHeadAttentionNaive
from utils import check_closeness

torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)
np.random.seed(69)
torch.manual_seed(69)


# TODO: batch_size = 1
@pytest.mark.parametrize(
    "batch_size",
    [
        2,
        # 64
    ],
)
def test_multi_head_attention(batch_size):
    d_model = 4
    seq_len = 5
    dim_kqv = d_model
    n_heads = 2
    learning_rate = 0.001

    x = torch.randint(
        low=0,
        high=10,
        size=(batch_size, seq_len, d_model),
        dtype=torch.float32,
    )
    y = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

    mha_cus = MultiHeadAttentionNaive(
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
    output_cus = mha_cus.forward(x)
    output_pt, attn_weights = mha_pt.forward(x, x, x, need_weights=True)
    attn_weights.retain_grad()
    # print("attn_weights", attn_weights)
    # print("before", in_proj_wt, mha_pt.in_proj_weight)
    # Initialize loss functions and optimizers for each framework
    loss_cus = torch.nn.MSELoss()
    loss_torch = torch.nn.MSELoss()

    optimizer_cus = torch.optim.Adam(
        params=[
            mha_cus.q_proj_layer._weights,
            mha_cus.q_proj_layer._bias,
            mha_cus.k_proj_layer._weights,
            mha_cus.k_proj_layer._bias,
            mha_cus.v_proj_layer._weights,
            mha_cus.v_proj_layer._bias,
            mha_cus.out_proj_layer._weights,
            mha_cus.out_proj_layer._bias,
        ],
        lr=learning_rate,
    )
    # pt_training_params = [sdpa_pt.W_key, sdpa_pt.W_query, sdpa_pt.W_value]

    optimizer_torch = torch.optim.Adam(
        params=mha_pt.parameters(),
        lr=learning_rate,
    )

    cost_cus = torch.sqrt(loss_cus(output_cus, y))
    output_pt.retain_grad()
    cost_pt = torch.sqrt(loss_torch(output_pt, y))

    # print("cost", cost_cus, cost_pt)

    # Backward pass and optimization
    cost_cus.backward()
    cost_pt.backward()
    # print("pt attn weight grad", attn_weights.grad)
    # # print("DL", output_pt.grad)
    # print("mha_pt.out", mha_pt.out_proj.weight.grad)
    # print("mha_pt.out", mha_pt.out_proj.bias.grad)
    # print("mha_pt.in_proj_weight", mha_pt.in_proj_weight.grad)
    # print("mha_pt.in_proj_bias", mha_pt.in_proj_bias.grad)
    optimizer_cus.step()
    optimizer_cus.zero_grad()
    optimizer_torch.step()
    optimizer_torch.zero_grad()

    # Check closeness of outputs
    assert check_closeness(
        output_cus.detach().numpy(), output_pt.detach().numpy()
    )  # , f"{get_output_template('pt')}"

    assert check_closeness(
        cost_cus.detach().numpy(), cost_pt.item()
    )  # , f"{get_loss_template('pt')}"
    # print(output_cus.detach().numpy(), output_pt.detach().numpy())
    # print("after", in_proj_wt, mha_pt.in_proj_weight)
    # Check closeness of weights

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
    print(in_proj_wt.detach().numpy(), mha_pt.in_proj_weight.detach().numpy())
    assert check_closeness(
        in_proj_wt.detach().numpy(),
        mha_pt.in_proj_weight.detach().numpy(),
    )  # , f"{get_weight_template('pt')}"
    print(
        ">>",
        in_proj_bias.detach().numpy().reshape(3, -1),
        "\n",
        mha_pt.in_proj_bias.detach().numpy().reshape(3, -1),
    )
    assert check_closeness(
        in_proj_bias.detach().numpy(),
        mha_pt.in_proj_bias.detach().numpy(),
    )  # , f"{get_bias_template('pt')}"

    assert check_closeness(
        mha_cus.out_proj_layer._weights.T.detach().clone(),
        mha_pt.out_proj.weight.detach().numpy(),
    )  # , f"{get_weight_template('pt')}"

    assert check_closeness(
        mha_cus.out_proj_layer._bias.detach().clone(),
        mha_pt.out_proj.bias.detach().numpy(),
    )  # , f"{get_bias_template('pt')}"
    # print(mha_cus.out_proj_layer._weights.T.detach().clone(),
    #     mha_pt.out_proj.weight.detach().numpy(),
    # )#, f"{get_weight_template('pt')}"

    # print(
    #     mha_cus.out_proj_layer._bias.detach().clone(),
    #     mha_pt.out_proj.bias.detach().numpy())
    # assert False
