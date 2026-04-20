import random
from engine import Value


class Neuron:
    def __init__(self, nin):
        # nin = number of inputs to this neuron

        # create weights (one per input)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]

        # bias (extra trainable parameter)
        self.b = Value(0.0)

    def __call__(self, x):
        # x = list of inputs [x1, x2, ...]

        # weighted sum: w1*x1 + w2*x2 + ... + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # non-linearity (IMPORTANT)
        out = act.tanh()

        return out

    def parameters(self):
        # return all parameters of this neuron
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        # nout = number of neurons in this layer

        # create multiple neurons
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # pass input through each neuron

        outs = [n(x) for n in self.neurons]

        # if only one neuron → return scalar
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # collect all neuron parameters
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        # nouts = list like [4, 4, 1]

        # example:
        # nin=3, nouts=[4,4,1]
        # means:
        # 3 → 4 → 4 → 1

        sz = [nin] + nouts

        self.layers = [
            Layer(sz[i], sz[i + 1]) for i in range(len(nouts))
        ]

    def __call__(self, x):
        # forward pass through all layers

        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        # collect all parameters from all layers
        return [p for layer in self.layers for p in layer.parameters()]