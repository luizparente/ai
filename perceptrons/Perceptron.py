import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(num_inputs)
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        summation = np.dot(self.weights, inputs) + self.bias

        return 1 if summation >= 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error