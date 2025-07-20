import numpy as np

class DeepNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(self.layer_sizes)
        self.learning_rate = learning_rate
        
        self.weights = []
        self.biases = []

        for i in range(1, self.num_layers):
            W = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * np.sqrt(2 / self.layer_sizes[i-1])
            b = np.zeros((self.layer_sizes[i], 1))
            self.weights.append(W)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))

        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, X):
        activation = X
        self.activations = [activation]
        self.preactivations = []
        
        for i in range(len(self.weights) - 1):
            Z = np.dot(self.weights[i], activation) + self.biases[i]
            activation = self.relu(Z)
            self.preactivations.append(Z)
            self.activations.append(activation)
        
        # Output layer with softmax.
        Z_output = np.dot(self.weights[-1], activation) + self.biases[-1]
        output = self.softmax(Z_output)
        self.preactivations.append(Z_output)
        self.activations.append(output)
        
        return output

    def backward(self, X, Y, output):
        m = X.shape[1]
        deltas = [None] * (self.num_layers - 1)
        
        # Output delta for cross-entropy + softmax.
        deltas[-1] = output - Y
        
        # Backpropagating.
        for i in range(self.num_layers - 2, 0, -1):
            dA = np.dot(self.weights[i].T, deltas[i])
            dZ = dA * self.relu_derivative(self.preactivations[i-1])
            deltas[i-1] = dZ
        
        # Applying updates.
        for i in range(self.num_layers - 1):
            prev_activation = self.activations[i]
            dW = (1/m) * np.dot(deltas[i], prev_activation.T)
            db = (1/m) * np.sum(deltas[i], axis=1, keepdims=True)
            
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db

    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, Y, output)

            if (epoch + 1) % 100 == 0:
                loss = -np.mean(np.sum(Y * np.log(output + 1e-8), axis=0)) # Cross-entropy loss.
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6f}")

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=0)
