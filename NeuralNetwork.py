import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        # Initialize the weights and biases for the layers
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Initialize weights with random values
        self.weights = [np.random.randn(layer_sizes[i-1], layer_sizes[i]) for i in range(1, self.num_layers)]

        # Initialize biases with zeros
        self.biases = [np.zeros((1, layer_sizes[i])) for i in range(1, self.num_layers)]

    # Sigmoid activation function        
    def activation_func(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Forward pass through the network
    def forward(self, inputs):
        self.inputs = inputs
        self.layer_outputs = [inputs]
        
        for i in range(self.num_layers - 1):
            layer_input = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            layer_output = self.activation_func(layer_input)
            self.layer_outputs.append(layer_output)
        
        return self.layer_outputs[-1]

    # Mutate the weights and biases of the neural network
    def mutate(self, mutation_rate=0.01):
        for i in range(self.num_layers - 1):
            self.weights[i] += np.random.randn(*self.weights[i].shape) * mutation_rate
            self.biases[i] += np.random.randn(*self.biases[i].shape) * mutation_rate

    # Make predictions with the trained network
    def predict(self, inputs):
        return self.forward(inputs)
    
    # Create a copy of the neural network
    def copy(self):
        copied_nn = NeuralNetwork(self.layer_sizes)
        copied_nn.weights = [np.copy(w) for w in self.weights]
        copied_nn.biases = [np.copy(b) for b in self.biases]
        return copied_nn


