class Network:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def count_parameters(self):
        n_parameters = 0
        for layer in self.layers:
            n_parameters += len(layer.weights) * len(layer.weights[0])
            n_parameters += len(layer.biases)
        return n_parameters
