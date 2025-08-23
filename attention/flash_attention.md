## A Deep Dive into Flash Attention: A Step-by-Step Tutorial 🚀

The **Transformer** architecture is a cornerstone of modern AI, but at its heart lies a demanding process: the **attention mechanism**. This mechanism scales quadratically ($O(N^2)$) with the length of the input sequence, making it a major bottleneck in terms of speed and memory.

**Flash Attention** is a groundbreaking algorithm that re-imagines this calculation to be I/O-aware, meaning it's optimized for how GPUs access memory. It dramatically reduces memory usage and increases speed, all without changing the final output. This tutorial breaks down how it works by walking through a single computational step with concrete matrices.

-----

## The Problem with Standard Attention

Standard scaled dot-product attention is defined by the formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The main bottleneck is the explicit creation of the massive score matrix $S = QK^T$, which requires $O(N^2)$ memory. Flash Attention cleverly avoids creating this full matrix by processing the inputs in smaller blocks.

-----

## Our Example Setup

We'll use the following matrices and parameters for our demonstration:

  * **Query ($Q$)**, **Key ($K$)**, and **Value ($V$)** matrices of size $9 \times 4$.
  * Query block size $B_{q} = 2$.
  * Key/Value block size $B_{kv} = 3$.

Here are the first few rows of our input matrices:

$$
Q = \begin{bmatrix}
0.15 & 0.82 & 0.21 & 0.55 \\
0.91 & 0.37 & 0.75 & 0.28 \\
0.48 & 0.63 & 0.95 & 0.12 \\
\vdots & \vdots & \vdots & \vdots \\
\end{bmatrix}
\quad
K = \begin{bmatrix}
0.28 & 0.71 & 0.19 & 0.42 \\
0.83 & 0.25 & 0.69 & 0.31 \\
0.59 & 0.52 & 0.87 & 0.24 \\
\vdots & \vdots & \vdots & \vdots \\
\end{bmatrix}
\quad
V = \begin{bmatrix}
0.11 & 0.95 & 0.38 & 0.62 \\
0.72 & 0.49 & 0.84 & 0.17 \\
0.68 & 0.31 & 0.77 & 0.39 \\
\vdots & \vdots & \vdots & \vdots \\
\end{bmatrix}
$$

-----

## A Single Computation Step in Detail

We will trace the very first step, where the first Query block ($Q_{1..2}$) is processed against the first Key/Value block ($K_{1..3}, V_{1..3}$).

### Step 1.1: Logits Calculation

First, we compute the logits block by multiplying the query block $Q_{1..2}$ with the key block $K_{1..3}^T$ and scaling by $1/\sqrt{d_k}$.
$$
S_{block} = S_{1..2,1..3} = \frac{Q_{1..2}K_{1..3}^T}{\sqrt{d_k}} 
$$

$$
\begin{bmatrix}
0.15 & 0.82 & 0.21 & 0.55 \\
0.91 & 0.37 & 0.75 & 0.28
\end{bmatrix}
\times
\begin{bmatrix}
0.28 & 0.83 & 0.59 \\
0.71 & 0.25 & 0.52 \\
0.19 & 0.69 & 0.87 \\
0.42 & 0.31 & 0.24
\end{bmatrix}
\quad \xrightarrow{\times \frac{1}{\sqrt{4}}} \quad
\begin{bmatrix}
0.48 & 0.60 & 0.54 \\
0.58 & 0.71 & 0.82
\end{bmatrix}
$$

### Step 1.2: Unnormalized Weights Calculation

Next, we stabilize the softmax by subtracting the row-wise maximum from the logits before exponentiating. The formula is $A_{\text{unnormalized}} = \exp(S_{block} - m_{block}) = \text{e}^{(S_{block} - m_{block})}$.

1.  **Find the row-wise maximum** of the logits block, which is $m_{block}$:
    $$
    m_{block} = 
    \begin{bmatrix} 
    0.60 
    \\ 0.82 
    \end{bmatrix}
    $$


2.  **Subtract the (broadcasted) maximum** from the logits block:

    $$
    S_{block} - m_{block} = 
    \begin{bmatrix} 
    0.48 & 0.60 & 0.54 \\
    0.58 & 0.71 & 0.82
    \end{bmatrix}
    \quad - \quad
    \begin{bmatrix} 
    0.60 & 0.60 & 0.60 \\ 
    0.82 & 0.82 & 0.82 
    \end{bmatrix}
    $$ 
    $$
    = \quad
    \begin{bmatrix}
    -0.12 & 0.00 & -0.06 \\
    -0.24 & -0.11 & 0.00
    \end{bmatrix}
    $$
3.  **Apply the exponential function** to get the unnormalized weights, $A_{\text{unnormalized}}$:

    $$
    A_{\text{unnormalized}} = 
    \exp \left( 
        \begin{bmatrix} 
        -0.12 & 0.00 & -0.06 \\ 
        -0.24 & -0.11 & 0.00 
        \end{bmatrix} \right)
    \quad = \quad
    \begin{bmatrix}
    0.89 & 1.00 & 0.94 \\
    0.79 & 0.89 & 1.00
    \end{bmatrix}
    $$
### Step 1.3: Online Softmax Statistics Update

The algorithm updates running statistics. For the first pass, the old stats ($m_{old}, l_{old}$) are initialized. We calculate the new stats ($m_{new}, l_{new}$) from the current block's values.

$$
m_{old} = \begin{bmatrix} -\infty \\ -\infty \end{bmatrix}, \quad
l_{old} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\quad \longrightarrow \quad
m_{new} = \begin{bmatrix} 0.60 \\ 0.82 \end{bmatrix}, \quad
l_{new} = \sum_{\text{cols}} A_{\text{unnormalized}} = \begin{bmatrix} 2.83 \\ 2.68 \end{bmatrix}
$$

### Step 1.4: Calculate Current Output

The unnormalized weights are multiplied with the Value block ($V_{1..3}$) to get this block's contribution to the output, which we'll call $O_{current}$.

$$
\begin{bmatrix} 
0.89 & 1.00 & 0.94 \\ 
0.79 & 0.89 & 1.00 
\end{bmatrix} 
\times 
\begin{bmatrix} 
0.11 & 0.95 & 0.38 & 0.62 \\ 
0.72 & 0.49 & 0.84 & 0.17 \\ 
0.68 & 0.31 & 0.77 & 0.39 
\end{bmatrix}
\quad = \quad
\begin{bmatrix}
1.46 & 1.62 & 1.90 & 1.09 \\
1.41 & 1.51 & 1.82 & 0.93
\end{bmatrix}
$$

### Step 1.5: Update Final Output 

Finally, the old output ($O_{old}$) is updated by combining it with the current output ($O_{current}$). This is the core of the online softmax. 

#### The General Update Formula
The update rule rescales the old output and the current output based on changes in the maximum logit before adding them. The matrix formula is as follows, where $\odot$ denotes broadcasted element-wise multiplication: 

$$O_{new} = 
\frac
{
    (l_{old} \odot e^{m_{old} - m_{new}}) \odot O_{old} + 
    (e^{m_{block} - m_{new}}) \odot O_{current} }
{ l_{new} }
$$ 

#### First-Pass Simplification 

Since this is the very first time we've processed these query rows, the calculation simplifies dramatically: 

* The old output $O_{old}$ is a matrix of zeros, so the first term in the numerator disappears. 
* The new maximum logit $m_{new}$ is just the maximum from the current block, so $m_{new} = m_{block}$. 
* This means the scaling factor for the current output becomes $e^{m_{block} - m_{new}} = e^0 = 1$. Therefore, for the first pass, the formula reduces to a simple normalization: $$O_{new} = \frac{O_{current}}{l_{new}}$$ We take $O_{current}$ from Step 1.4 and normalize it by dividing each row by the corresponding value in $l_{new}$ from Step 1.3: 
$$
O_{new} = 
\begin{bmatrix} 
1.46 / 2.83 & 1.62 / 2.83 & 1.90 / 2.83 & 1.09 / 2.83 \\ 
1.41 / 2.68 & 1.51 / 2.68 & 1.82 / 2.68 & 0.93 / 2.68 
\end{bmatrix}
$$
$$ 
O_{new} = O_{1..2} = 
\begin{bmatrix}
0.51 & 0.57 & 0.67 & 0.38 \\
0.53 & 0.56 & 0.68 & 0.35
\end{bmatrix}
$$


In later iterations (e.g., when this query block is processed against the next Key/Value block), the full, more complex update rule would be used to correctly merge the old results with the new ones.
<!-- ---- -->

## Conclusion

By breaking the attention calculation into tiled blocks and using an online softmax algorithm, Flash Attention avoids storing the massive $N \times N$ matrix. This makes it substantially faster and more memory-efficient, unlocking the ability to train on much longer sequences. This step-by-step walkthrough demonstrates the core mechanics that make this efficiency possible.