import torch


class RotaryPositionalEncoding(torch.nn.Module):
    """
    Implements Rotary Positional Encoding (RoPE).

    RoPE is a type of positional encoding that injects positional information
    into a sequence of tokens by rotating the embedding vectors. Unlike
    sinusoidal positional encodings which are added to the embeddings, RoPE is
    applied multiplicatively.

    The core idea is to represent the position `m` as a rotation matrix `R_m` &
    apply it to the input embedding `x_m`. For a `d`-dimensional embedding, the
    dimensions are paired up. For each pair `(x_{2i-1}, x_{2i})`, the rotation
    is:

    [x'_{m, 2i-1}] = [cos(m*theta_i)  -sin(m*theta_i)] [x_{m, 2i-1}]
    [x'_{m, 2i}]   = [sin(m*theta_i)   cos(m*theta_i)] [x_{m, 2i}]

    where `theta_i = 10000^(-2i/d)`.

    This can be expressed using complex numbers as
    `x'_m = x_m * exp(i * m * theta)`.
    The key property of RoPE is that the dot product between rotated vectors
    `<R_m(q), R_n(k)>` only depends on the relative position `m-n`.
    """

    def __init__(self, d_model: int):
        """
        Initializes the RotaryPositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the model.
        """
        super().__init__()
        self.wavelength = 10_000
        self.d_model = d_model

        assert d_model % 2 == 0, "d_model must be even for RoPE."

        # Create the theta values for the rotation
        ix = torch.arange(self.d_model) // 2
        # ix -> [0, 0, 1, 1, .... d/2 - 1, d/2 - 1]
        self.theta = self.wavelength ** (-2 * ix / self.d_model)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Applies Rotary Positional Encoding to the input tensor.

        Args:
            input_ (torch.Tensor): The input tensor of shape
                (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The tensor with applied rotary positional encoding,
                of the same shape as the input.
        """
        # input_ -> (batch_size, seq_len, d_model)
        seq_len = input_.shape[1]

        # Create the position indices `m`
        m = torch.arange(seq_len).view(-1, 1)

        # Calculate the rotation angles `m * theta_i`
        theta_m = self.theta * m
        # theta_m -> (seq_len, d_model)

        # Apply the rotation using sine and cosine
        # This is equivalent to:
        # x_rotated_{2i-1} = x_{2i-1}*cos(m*theta_i) - x_{2i}*sin(m*theta_i)
        # x_rotated_{2i}   = x_{2i}*cos(m*theta_i) + x_{2i-1}*sin(m*theta_i)
        first_comp = input_ * torch.cos(theta_m)
        second_comp = torch.stack(
            (-input_[:, :, 1::2], input_[:, :, ::2]), dim=-1
        ).contiguous().reshape(input_.shape) * torch.sin(theta_m)

        return first_comp + second_comp
        # (batch_size, seq_len, d_model)
