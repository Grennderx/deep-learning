import math

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh_act(x):
    num = math.exp(x) - math.exp(-x)
    denom = math.exp(x) + math.exp(-x)
    return num / denom
    # return math.tanh(x)


def tanh_derivative(x):
    t = math.tanh(x)
    return 1 - t * t


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return 1.0 if x > 0 else 0.0


def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x


def leaky_relu_derivative(x, alpha=0.01):
    return 1.0 if x > 0 else alpha


# аппроксимация
def gelu_approx(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu(x):
    # Φ(x) = CDF стандартного N(0,1)
    phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return x * phi


def gelu_derivative(x):
    phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return phi + x * pdf


def swish(x):
    return x * sigmoid(x)


def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)


def softmax(xs):
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]
