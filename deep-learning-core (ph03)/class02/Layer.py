import math 


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))


class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        else:
            import random
            self.weights = [[random.uniform(-1, 1) for _ in range(n_inputs)] for _ in range(n_neurons)]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0.0] * n_neurons
    
    def forward(self, inputs):
        self.last_input = inputs
        self.last_output = []
        for neuron_idx in range(len(self.weights)):
            z = sum(w * x for w, x in zip(self.weights[neuron_idx], inputs))
            z += self.biases[neuron_idx]
            self.last_output.append(sigmoid(z))
        return self.last_output
