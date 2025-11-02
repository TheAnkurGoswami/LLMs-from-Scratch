import torch

from attention.layernorm import LayerNorm
from attention.multi_head_attention import MultiHeadAttentionNaive
from attention.projection import FeedForwardNetwork


class DecoderLayer(torch.nn.Module):
    """
    Implements a single layer of the Transformer Decoder.

    A decoder layer consists of three main sub-layers:
    1.  A masked multi-head self-attention mechanism. This ensures that
        predictions for a position can only depend on known outputs at
        positions less than the current position.
    2.  A multi-head attention mechanism over the output of the encoder stack.
        This is where the decoder "attends" to the input sequence.
    3.  A position-wise fully connected feed-forward network.

    Each of these sub-layers has a residual connection around it, followed by
    layer normalization.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048):
        """
        Initializes the DecoderLayer module.

        Args:
            d_model (int): The dimensionality of the input and output.
            num_heads (int): The number of heads in the multi-head attention.
            d_ff (int, optional): The dimensionality of the inner-layer of the
                feed-forward network. Defaults to 2048.
        """
        super().__init__()
        # Masked Multi-Head Attention for self-attention on the decoder side
        self.mmha = MultiHeadAttentionNaive(
            d_model=d_model, num_heads=num_heads, causal_mask=True
        )
        self.layer_norm_1 = LayerNorm(d_model=d_model)
        # Multi-Head Attention for attending to the encoder output
        self.mha = MultiHeadAttentionNaive(
            d_model=d_model, num_heads=num_heads
        )
        self.layer_norm_2 = LayerNorm(d_model=d_model)
        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
        self.layer_norm_3 = LayerNorm(d_model=d_model)

    def forward(
        self, in_embeddings: torch.Tensor, encoder_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the DecoderLayer.

        Args:
            in_embeddings (torch.Tensor): The input tensor from the previous
                decoder layer, of shape (batch_size, seq_len, d_model).
            encoder_out (torch.Tensor): The output from the encoder stack, of
                shape (batch_size, seq_len_enc, d_model).

        Returns:
            torch.Tensor: The output tensor of the same shape as
                `in_embeddings`.
        """
        # Masked Multi-Head Self-Attention sub-layer
        mmha_out = self.mmha(in_embeddings, in_embeddings, in_embeddings)
        # Residual connection and layer normalization
        norm_out_1 = self.layer_norm_1(mmha_out + in_embeddings)

        # Multi-Head Attention over encoder output
        mha_out = self.mha(encoder_out, encoder_out, norm_out_1)
        # Residual connection and layer normalization
        norm_out_2 = self.layer_norm_2(mha_out + norm_out_1)

        # Feed-Forward Network sub-layer
        ffn_out = self.ffn(norm_out_2)
        # Residual connection and layer normalization
        output = self.layer_norm_3(ffn_out + norm_out_2)

        return output
