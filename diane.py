#! python3
import numpy as np
import nnfs
from nnfs.datasets import spiral_data;

np.random.seed(0);
nnfs.init()

#initializing data
X, y = spiral_data(100, 3);

#layer's weights and biases
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons);
        self.biases = np.zeros((1, n_neurons));
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases;

#rectified linear unit - activation function
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
layer1 = Layer(2, 5);
activation1 = ActivationReLU();

layer1.forward(X);
print(layer1.output);

activation1.forward(layer1.output);
print(activation1.output);

