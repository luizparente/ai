# Neural Networks

Neural networks form a cornerstone of modern artificial intelligence and machine learning, inspired by the structure and function of biological neural systems. These computational models are designed to approximate complex functions by processing input data through interconnected nodes, enabling tasks such as pattern recognition, classification, and prediction. This overview begins with the foundational perceptron, progresses to simple neural networks with a single hidden layer, and extends to deep neural networks and convolutional neural networks. Explanations assume no prior knowledge and proceed in a logical, structured manner.

## The Perceptron: The Foundation

The perceptron represents the simplest form of an artificial neural network, serving as a building block for more advanced architectures. Developed in the late 1950s, it models a single neuron that processes inputs to produce a binary output.

Consider a perceptron as a mathematical function that takes multiple input values, each associated with a weight, and computes a weighted sum. This sum is then passed through an activation function to determine the output. Formally, for inputs $x_1, x_2, \dots, x_n$ with corresponding weights $w_1, w_2, \dots, w_n$ and a bias term $b$, the output $y$ is given by:

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

$$
y = \begin{cases} 
1 & \text{if } z \geq 0 \\
0 & \text{otherwise}
\end{cases}
$$

Here, the activation function is a step function that thresholds the result at zero. The weights and bias are adjustable parameters learned during training, typically through an algorithm that minimizes prediction errors by iteratively updating these values based on the difference between predicted and actual outputs.

A single perceptron can solve linearly separable problems, such as classifying data points that can be divided by a straight line in two-dimensional space. However, it fails for nonlinear problems, such as the exclusive-or (XOR) function, where no single line separates the classes.

## Evolving from Perceptrons

To address the limitations of a single perceptron, multiple perceptrons can be combined into a multilayer structure, forming a simple neural network. This evolution introduces a *hidden layer*—an intermediate layer of nodes between the input and output layers—enabling the model to capture nonlinear relationships.

A basic neural network with one hidden layer consists of three layers:
- **Input Layer**: Receives the raw input features, with each node corresponding to one feature.
- **Hidden Layer**: Comprises a small number of artificial neurons that process the inputs. Each neuron in this layer computes a weighted sum of the input layer's outputs, applies a nonlinear activation function, and passes the result forward.
- **Output Layer**: Produces the final prediction, often using a similar computation as the hidden layer.

For illustration, suppose an input vector $\mathbf{x} = [x_1, x_2]$ feeds into a hidden layer with two neurons. The hidden layer outputs $\mathbf{h}$ are computed as:

$$
h_j = \sigma \left( \sum_{i} w_{ji} x_i + b_j \right), \quad j = 1, 2
$$

where $\sigma$ is a nonlinear activation function, such as the classic sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The output $y$ is then:

$$
y = \sigma \left( \sum_{j} v_j h_j + c \right)
$$

Here, $w_{ji}$, $b_j$, $v_j$, and $c$ are learned parameters. Training involves backpropagation, an algorithm that computes gradients of the error with respect to each weight and updates them using gradient descent to minimize a loss function, such as mean squared error.

This architecture, known as a multilayer perceptron (MLP) with one hidden layer, can approximate any continuous function given sufficient hidden nodes, as per the universal approximation theorem. It evolves directly from the perceptron by stacking layers, allowing the network to learn hierarchical representations and solve nonlinear tasks.

## Deep Neural Networks

Finally, deep neural networks (DNNs) extend the simple neural network by incorporating many hidden layers. This depth enables the model to learn increasingly abstract features from data, making DNNs powerful for complex tasks like image recognition and natural language processing.

In a DNN, each layer transforms the output of the previous layer through weighted connections and activations. The architecture can be represented as a sequence of transformations:

$$
\mathbf{h}^{(1)} = \sigma(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)})
$$

$$
\mathbf{h}^{(k)} = \sigma(\mathbf{W}^{(k)} \mathbf{h}^{(k-1)} + \mathbf{b}^{(k)}), \quad k = 2, \dots, L
$$

$$
\mathbf{y} = f(\mathbf{W}^{(L+1)} \mathbf{h}^{(L)} + \mathbf{b}^{(L+1)})
$$

where $L$ is the number of hidden layers, $\mathbf{W}^{(k)}$ and $\mathbf{b}^{(k)}$ are the weights and biases for layer $k$, and $f$ is the output activation (e.g., softmax for classification).

Training DNNs requires large datasets and computational resources, often using techniques like dropout to prevent overfitting and advanced optimizers like Adam. The depth allows for feature hierarchy: early layers detect basic patterns (e.g., edges in images), while deeper layers combine these into sophisticated concepts (e.g., objects). Challenges include vanishing gradients, addressed by activations like ReLU:

$$
\text{ReLU}(z) = \max(0, z)
$$

DNNs power applications in various domains, demonstrating superior performance on high-dimensional data.

Neural networks encompass a diverse array of architectures beyond perceptrons, multilayer perceptrons, and deep neural networks, such as:

* Convolutional neural networks, 
* Recurrent neural networks, 
* Transformer neural networks, 
* Generative adversarial networks, 
* Autoencoders, 
* and more. 

These variations are tailored to specific data types and tasks, such as sequential processing, generative modeling, or efficient handling of long-range dependencies. These types can be found in this repository in dedicated examples.