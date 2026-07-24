import random
from activation_funcs import sigmoid, leaky_relu, leaky_relu_derivative


def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x, y = random.uniform(-2, 2), random.uniform(-2, 2)
        data.append(([x, y], 1.0 if x * x + y * y < 1.5 else 0.0))
    return data

class Net:
    def __init__(self, act, act_d, lr=0.1, hidden=8, learn_alpha=False, alpha=0.01):
        random.seed(0)
        self.act, self.act_d, self.lr = act, act_d, lr
        self.H = hidden
        self.learn_alpha = learn_alpha
        self.alpha = alpha
        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden)]
        self.b2 = 0.0
    def forward(self, x):
        self.x, self.z1, self.h = x, [], []
        for i in range(self.H):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(self.act(z, self.alpha))
        self.out = sigmoid(sum(w * h for w, h in zip(self.w2, self.h)) + self.b2)
        return self.out

    
    def train(self, data, epochs=200):
        for ep in range(epochs):
            loss = correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                loss += (pred - y) ** 2
                correct += (pred >= 0.5) == (y >= 0.5)
            if ep % 50 == 0 or ep == epochs - 1:
                print(f"    ep {ep:3d}: loss={loss/len(data):.4f} acc={correct/len(data)*100:.1f}%"
                      + (f" alpha={self.alpha:.4f}" if self.learn_alpha else ""))
data = make_circle_data()
print("=== Leaky ReLU (alpha=0.01 fixed) ===")
leaky = Net(leaky_relu, leaky_relu_derivative, learn_alpha=False, alpha=0.01)
leaky.train(data)
print("\n=== PReLU (alpha learnable) ===")
prelu = Net(leaky_relu, leaky_relu_derivative, learn_alpha=True, alpha=0.01)
prelu.train(data)
print(f"\nLearned alpha: {prelu.alpha:.4f}  (Leaky fixed: 0.01)")