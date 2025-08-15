import numpy as np


class ActivationFunction:
    def __init__(self):
        raise NotImplementedError()

    def __call__(self, x):
        raise NotImplementedError()

    def derivative(self, x):
        raise NotImplementedError()

class Sigmoid(ActivationFunction):
    def __init__(self):
        ()

    def __call__(self, x: np.array):
        return 1/(1+np.exp(-x))

    def derivative(self, x):
        return Sigmoid(x) * (1 - Sigmoid(x))

class ReLU(ActivationFunction):
    def __init__(self):
        ()

    def __call__(self, x: np.array):
        return np.maximum(x, 0)

    def derivative(self, x: np.array):
        return np.where(x >= 0, 1, 0)
