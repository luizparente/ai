# Long Short-Term Memory (LSTM) for Sentiment Classification

## Recurrent Neural Networks (RNNs)

Traditional neural networks, such as feedforward networks, process inputs independently, assuming no temporal dependencies. However, many real-world data types, including text sequences for sentiment classification, exhibit sequential structure where the order of elements (e.g., words in a sentence) matters.

Recurrent Neural Networks (RNNs) address this by introducing loops that allow information to persist across time steps. In an RNN, the hidden state $h_t$ at time step $t$ is computed as:

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

where $x_t$ is the input at time $t$, $W_{xh}$ and $W_{hh}$ are weight matrices, $b_h$ is a bias, and $\tanh$ is the hyperbolic tangent activation function. The output $y_t$ can then be derived from $h_t $, often via another transformation.

For sentiment classification, an input sequence (e.g., a sentence tokenized into words) is fed into the RNN one token at a time, with the final hidden state summarizing the sequence for a classification decision (positive or negative).

### Limitations of Standard RNNs: Vanishing and Exploding Gradients

While RNNs are theoretically capable of capturing long-range dependencies, they suffer from practical issues during training via backpropagation through time (BPTT). Gradients propagated backward can either *vanish* (become exponentially small) or *explode* (become exponentially large), especially over long sequences. This is due to repeated multiplications by the weight matrix $W_{hh}$ in the gradient computation.

Vanishing gradients prevent the model from learning long-term dependencies, which is critical in sentiment analysis where sentiment may depend on words far apart (e.g., "The movie was not great until the twist ending" – the negation "not" affects distant words).

### Long Short-Term Memory (LSTM) Networks: Addressing RNN Limitations

LSTMs, introduced by Hochreiter and Schmidhuber in 1997, extend RNNs by incorporating a *memory cell* and gating mechanisms to selectively remember or forget information over long sequences. This mitigates vanishing gradients by allowing gradients to flow through linear paths (via the cell state) without repeated non-linear transformations.

The core of an LSTM is the *LSTM cell*, which maintains two key states:

- **Hidden state** $h_t$: Similar to RNNs, this is the output passed to the next time step or used for predictions.
- **Cell state** $c_t$: A long-term memory vector that carries information across the sequence with minimal alteration.

The LSTM cell uses three gates to control information flow:

- **Forget gate** $f_t$: Decides what information to discard from the previous cell state $c_{t-1}$. It is computed as:

$$
f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)
$$

where $\sigma$ is the sigmoid function ($ \sigma(z) = \frac{1}{1 + e^{-z}} $), outputting values between 0 (forget) and 1 (keep).
- **Input gate** $i_t$ and candidate cell state $\tilde{c}_t$: The input gate determines what new information to add, while the candidate computes potential updates:

$$
i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)
$$

$$
\tilde{c}_t = \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c)
$$

The new cell state is then:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

where $\odot$ denotes element-wise multiplication.

- **Output gate** $o_t$: Controls what parts of the cell state to output to the hidden state:

$$
o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o)
$$

$$
h_t = o_t \odot \tanh(c_t)
$$

These gates enable the LSTM to learn which information is relevant, making it robust for tasks requiring long-term memory.

## LSTMs in Sentiment Classification

Sentiment classification is a binary classification task (e.g., positive vs. negative) applied to text sequences. The process involves:

- **Preprocessing**: Convert text to numerical vectors via tokenization and embedding (e.g., mapping words to dense vectors).
- **Sequence Processing**: Feed embedded tokens into LSTM cells sequentially. The final hidden state $h_T$ (at the last time step $T$) encodes the entire sequence.
- **Classification**: Pass $h_T$ through a fully connected layer with a sigmoid activation for binary output:

$$
\hat{y} = \sigma(W_y h_T + b_y)
$$

where $\hat{y} > 0.5$ indicates positive sentiment, otherwise negative.

LSTMs excel here because they capture contextual nuances, such as sarcasm or negation, by preserving long-range dependencies.

### Training and Optimization

LSTMs are trained using gradient descent with BPTT. The loss function for binary classification is typically binary cross-entropy:

$$
\mathcal{L} = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
$$