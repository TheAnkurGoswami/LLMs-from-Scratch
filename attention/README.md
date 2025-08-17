# Attention Module

The `attention` module provides the building blocks for the attention mechanism in the Transformer model.

## Scaled Dot-Product Attention

The `ScaledDotProductAttention` class implements the scaled dot-product attention mechanism.

The attention score is calculated as follows:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{k}}}\right)V
$$

Where:
- $Q$ is the query matrix.
- $K$ is the key matrix.
- $V$ is the value matrix.
- $d_{k}$ is the dimension of the key vectors.

A causal mask can be applied to prevent positions from attending to subsequent positions.

## Multi-Head Attention

The `MultiHeadAttention` class implements the multi-head attention mechanism. It allows the model to jointly attend to information from different representation subspaces at different positions.

The input is projected into $h$ different subspaces, and scaled dot-product attention is applied in each subspace. The outputs are then concatenated and projected back to the original dimension.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}\_{1}, \dots, \text{head}_{h})W^O
$$

where $\text{head}\_{i} = \text{Attention}(QW_{i}^Q, KW_{i}^K, VW_{i}^V)$.

The projection matrices $W_{i}^Q$, $W_{i}^K$, $W_{i}^V$ and $W^O$ are learned during training.

## Projection

The `Projection` class is a simple linear layer that projects the input from `in_features` to `out_features`. It can optionally include a bias term.

The operation is:

$$
\text{output} = \text{input} \cdot W + b
$$

Where:
- $W$ is the weight matrix.
- $b$ is the bias vector.


## Feed-Forward Network

The `FeedForwardNetwork` class implements the position-wise feed-forward network used in the Transformer encoder and decoder layers. It consists of two linear transformations with a ReLU activation in between.

$$
\text{FFN}(x) = \max(0, xW_{1} + b_{1})W_{2} + b_{2}
$$


## Layer Normalization

The `LayerNorm` class implements layer normalization. It normalizes the inputs across the features.

The operation is:

$$
\text{output} = \frac{\text{input} - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

Where:
- $\mu$ is the mean of the input.
- $\sigma^2$ is the variance of the input.
- $\epsilon$ is a small value to prevent division by zero.
- $\gamma$ is a learnable gain parameter.
- $\beta$ is a learnable bias parameter.



## Flash Attention

The `FlashAttention` class implements a memory-efficient exact attention mechanism. It avoids materializing the large $N \times N$ attention matrix by using tiling and an online softmax algorithm.

The online softmax algorithm updates the attention output block by block, using the following update rule:

$$
m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}})
$$

$$
l_{\text{new}} = l_{\text{old}} e^{m_{\text{old}} - m_{\text{new}}} + l_{\text{block}} e^{m_{\text{block}} - m_{\text{new}}}
$$

$$
O_{\text{new}} = \frac{l_{\text{old}} e^{m_{\text{old}} - m_{\text{new}}} O_{\text{old}} + l_{\text{block}} e^{m_{\text{block}} - m_{\text{new}}} O_{\text{block}}}{l_{\text{new}}}
$$

Where:
- $m$ is the running maximum of the logits, used for numerical stability. For a block of scores $S_{ij} = Q_i K_j^T / \sqrt{d_k}$, the block-level statistics are:
  $$
  m_{\text{block}} = \max(S_{ij})
  $$
- $l$ is the running sum of the exponentials of the scaled logits.
  $$
  l_{\text{block}} = \sum \exp(S_{ij} - m_{\text{block}})
  $$
- $O$ is the attention output.