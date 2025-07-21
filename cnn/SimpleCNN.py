import numpy as np
from scipy.signal import convolve2d, correlate2d

class SimpleCNN:
    """
    A simple Convolutional Neural Network implementation with one hidden layer.
    """
    def __init__(self, num_filters=8, filter_size=3, pool_size=2, hidden_size=128, output_size=10, input_shape=(28, 28)):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.input_shape = input_shape
        self.output_size = output_size
        
        # Initializing filters (num_filters, filter_size, filter_size).
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / np.sqrt(filter_size * filter_size)
        
        # Calculating the size of the fully connected layer input.
        conv_output_h = (input_shape[0] - filter_size + 1) // pool_size
        conv_output_w = (input_shape[1] - filter_size + 1) // pool_size
        self.fc_input_size = num_filters * conv_output_h * conv_output_w

        # Initializing weights and biases for the fully connected layers.
        self.w1 = np.random.randn(self.fc_input_size, hidden_size) / np.sqrt(self.fc_input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_deriv(self, x):
        return x > 0
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))

        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def convolve(self, x, filt, mode='valid'):
        return convolve2d(x, filt, mode=mode)
    
    def max_pool(self, x, pool_size):
        h, w = x.shape
        new_h = h // pool_size
        new_w = w // pool_size
        pooled = np.zeros((new_h, new_w))
        
        for i in range(new_h):
            for j in range(new_w):
                pooled[i, j] = np.max(x[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size])

        return pooled
    
    def forward(self, x):
        # Initializing the output of the convolutional layer.
        conv_out = np.zeros((self.num_filters, self.input_shape[0] - self.filter_size + 1, self.input_shape[1] - self.filter_size + 1))
        
        # Convolution operation.
        for i in range(self.num_filters):
            conv_out[i] = self.convolve(x, self.filters[i])

        conv_out = self.relu(conv_out)
        
        # Pooling operation.
        pooled_out = np.zeros((self.num_filters, conv_out.shape[1] // self.pool_size, conv_out.shape[2] // self.pool_size))
        
        for i in range(self.num_filters):
            pooled_out[i] = self.max_pool(conv_out[i], self.pool_size)
        
        # Flatten and fully connected.
        flattened = pooled_out.reshape(1, -1)
        hidden = self.relu(np.dot(flattened, self.w1) + self.b1)
        output = np.dot(hidden, self.w2) + self.b2
        probs = self.softmax(output)

        return probs, hidden, flattened, pooled_out, conv_out
    
    def backward(self, x, y, probs, hidden, flattened, pooled_out, conv_out, learning_rate=0.01):
        # Output layer gradients.
        d_output = probs
        d_output[range(1), y] -= 1
        
        dw2 = np.dot(hidden.T, d_output)
        db2 = np.sum(d_output, axis=0, keepdims=True)
        d_hidden = np.dot(d_output, self.w2.T) * self.relu_deriv(hidden)
        
        dw1 = np.dot(flattened.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0, keepdims=True)
        
        # Reshaping d_flattened back to pooled shape.
        d_flattened = np.dot(d_hidden, self.w1.T)
        d_pooled = d_flattened.reshape(pooled_out.shape)
        
        # Upsampling pooling gradient (simple approximation for max pool).
        d_conv = np.zeros_like(conv_out)

        for f in range(self.num_filters):
            for i in range(d_pooled.shape[1]):
                for j in range(d_pooled.shape[2]):
                    region = conv_out[f, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                    max_mask = region == np.max(region)
                    d_conv[f, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size] = d_pooled[f, i, j] * max_mask
        
        d_conv *= self.relu_deriv(conv_out)
        
        # Filtering gradients.
        d_filters = np.zeros_like(self.filters)

        for i in range(self.num_filters):
            d_filters[i] = correlate2d(x, d_conv[i], mode='valid')
        
        # Updating parameters.
        self.filters -= learning_rate * d_filters
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, epochs=1, learning_rate=0.01):
        for epoch in range(epochs):
            for i in range(len(X)):
                probs, hidden, flattened, pooled_out, conv_out = self.forward(X[i])
                self.backward(X[i], y[i], probs, hidden, flattened, pooled_out, conv_out, learning_rate)

            print(f"Epoch {epoch + 1}/{epochs} completed.")
    
    def predict(self, x):
        probs, _, _, _, _ = self.forward(x)

        return np.argmax(probs, axis=1)
