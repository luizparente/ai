import numpy as np

class SimpleNeuralNetwork:
    """
    Implements a neural network with one hidden layer.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initializing weights and biases with small random values.
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01   # weights for input to hidden.
        self.b1 = np.zeros((hidden_size, 1))                        # bias for hidden layer.
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01  # weights for hidden to output.
        self.b2 = np.zeros((output_size, 1))                        # bias for output layer.
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)

        return s * (1 - s)

    def forward(self, X):
        # X is of shape (input_size, number_of_samples).
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.sigmoid(self.Z1) # Hidden layer activation.
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2) # Output layer activation.

        return self.A2

    def backward(self, X, Y, output):
        m = X.shape[1]
        dZ2 = output - Y
        dW2 = (1/m) * np.dot(dZ2, self.A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.sigmoid_derivative(self.Z1)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # Gradient descent updates.
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, Y, output)

            if (epoch + 1) % 100 == 0:
                loss = np.mean((Y - output) ** 2)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6f}")

    def predict(self, X):
        output = self.forward(X)

        return np.argmax(output, axis=0)
