# Recurrent Neural Networks (RNNs)

RNNs are a class of neural networks designed to handle sequential data, where the order of elements matters, such as time series, text, or speech. Unlike feedforward networks, RNNs maintain a hidden state that captures information from previous inputs, allowing them to process sequences of variable length. This makes them suitable for tasks like predicting the next character in a sequence.

The core idea is a recurrent loop: at each time step $t$, the network takes an input $x_t$, updates its hidden state $h_t$ based on $x_t$ and the previous hidden state $h_{t-1}$, and produces an output $y_t$. Mathematically, this is expressed as:

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

$$
y_t = W_{hy} h_t + b_y
$$

Here, $W_{xh}$, $W_{hh}$, and $W_{hy}$ are weight matrices, $b_h$ and $b_y$ are biases, and $\tanh$ is the activation function. For prediction tasks like character generation, we apply a softmax function to $y_t$ to obtain probabilities.

Training involves backpropagation through time (BPTT), where gradients are computed backward across the sequence to update weights. We use cross-entropy loss for classification tasks.

## Key Parameters and Components of the RNN

To enable accurate predictions, a typical RNN implementation includes some typical parameters:

- **Input size**: Dimensionality of each input (e.g., one-hot vector size for characters).
- **Hidden size**: Dimensionality of the hidden state, controlling model capacity.
- **Output size**: Dimensionality of the output (e.g., number of possible characters).
- **Learning rate**: Step size for gradient descent updates.
- **Weights and biases**: $W_{xh}$, $W_{hh}$, $W_{hy}$, $b_h$, $b_y$, initialized randomly.
- **Activation**: Tanh for hidden states, softmax for outputs.
- **Loss function**: Cross-entropy for next-character prediction.
- **Gradient clipping**: To mitigate exploding gradients during BPTT.

The provided class `SimpleRNN` encapsulates forward propagation, backward propagation, training, and prediction logic.

This implementation assumes inputs are lists of one-hot encoded vectors. The `train` method processes batches of sequences, computes loss, and updates via BPTT. The `predict` method returns the index of the most likely output.

### What is One-Hot Encoding?

One-hot encoding is a fundamental preprocessing technique in machine learning and artificial intelligence, designed to transform categorical data—such as labels like "red," "blue," or "green" for a color variable—into a numerical format that algorithms can effectively process, assuming no prior knowledge of data encoding methods. 

At its core, this method creates a binary vector for each unique category within the variable, where the vector consists of zeros everywhere except for a single '1' at the position corresponding to the specific category, thereby representing each category independently without implying any ordinal relationship or hierarchy that could mislead models, such as assuming "blue" is greater than "red" if encoded numerically. 

For instance, if a dataset includes a "fruit" column with categories "apple," "banana," and "cherry," one-hot encoding would generate three new binary columns: one for each fruit, with a '1' indicating the presence of that fruit in a given row and '0' otherwise, thus expanding the dataset's dimensionality but ensuring compatibility with algorithms like linear regression or neural networks that require numerical inputs. This approach preserves the distinctiveness of categories, enhances model performance by avoiding artificial rankings, and is particularly useful for nominal data where no inherent order exists, though it can increase computational demands if the number of categories is large, potentially leading to sparse matrices. By converting categorical variables in this manner, one-hot encoding facilitates more accurate predictions in tasks ranging from natural language processing to classification problems, making it an essential tool for handling non-numeric data in AI and ML workflows.