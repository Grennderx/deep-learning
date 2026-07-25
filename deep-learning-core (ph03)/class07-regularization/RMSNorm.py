import math


class RMSNorm:
    def __init__(self, num_features, eps=1e-6):
        self.gamma = [1.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        rms = math.sqrt(1 / len(x) * sum(xi ** 2 for xi in x))
        output = []
        for j in range(self.num_features):
            output.append(self.gamma[j] * (x[j]/(rms + self.eps)))
        return output
        