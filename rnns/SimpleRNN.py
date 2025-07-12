import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initializing parameters.
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden.
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden.
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output.
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias.
        self.by = np.zeros((output_size, 1))  # Output bias.

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))

        return e_x / e_x.sum(axis=0)

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))  # Initial hidden state.
        self.last_inputs = inputs
        self.last_hs = {-1: h}

        for t, x in enumerate(inputs):
            x = x.reshape(-1, 1)  # Ensuring column vector.
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            self.last_hs[t] = h

        y = np.dot(self.Why, h) + self.by
        p = self.softmax(y)

        return p, h

    def backward(self, d_y):
        n = len(self.last_inputs)
        d_Why = np.dot(d_y, self.last_hs[n-1].T)
        d_by = d_y
        d_Whh = np.zeros_like(self.Whh)
        d_Wxh = np.zeros_like(self.Wxh)
        d_bh = np.zeros_like(self.bh)
        d_h = np.dot(self.Why.T, d_y)

        for t in reversed(range(n)):
            temp = (1 - self.last_hs[t] ** 2) * d_h  # Derivative of tanh.
            d_bh += temp
            d_Wxh += np.dot(temp, self.last_inputs[t].reshape(1, -1))  # Corrected line.
            d_Whh += np.dot(temp, self.last_hs[t-1].T)
            d_h = np.dot(self.Whh.T, temp)

        # Gradient clipping.
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -5, 5, out=d)

        # Updating parameters.
        self.Wxh -= self.learning_rate * d_Wxh
        self.Whh -= self.learning_rate * d_Whh
        self.Why -= self.learning_rate * d_Why
        self.bh -= self.learning_rate * d_bh
        self.by -= self.learning_rate * d_by

    def train(self, inputs, targets, epochs=10):
        for epoch in range(epochs):
            loss = 0.0

            for x_seq, y_target in zip(inputs, targets):
                p, _ = self.forward(x_seq)
                loss += -np.log(p[y_target, 0])  # Cross-entropy loss
                d_y = p.copy()
                d_y[y_target, 0] -= 1
                self.backward(d_y)

            avg_loss = loss / len(inputs)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, inputs):
        p, _ = self.forward(inputs)
        
        return np.argmax(p)