import numpy as np
from activationfunction import ActivationFunction


class Layer:
    def __init__(
            self,
            n_inputs: int,
            n_neurons: int,
            activation_function: ActivationFunction,
            name: str=None,
            init_method: str='zeros'):
        if init_method == 'zeros':
            self.weights = np.zeros((n_inputs, n_neurons))
        elif init_method == 'random':
            self.weights = np.random.rand(n_inputs, n_neurons)
        else:
            raise ValueError("init_method not in {'zeros', 'random'}")
        self.b = np.zeros((1, n_neurons))
        self.activation_function = activation_function
        self.name = name

        self._input = None
        self._z = None

    def _forward(self, X: np.array):
        self._input = X
        self._z = np.dot(X, self.weights) + self.b
        return self.activation_function(self._z)

    def _backward(self, da: np.array, learning_rate: float):
        dz = da * self.activation_function.derivative(self._z)
        dw = np.dot(self._input.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        dx = np.dot(dz, self.weights.T)
        self.weights -= learning_rate * dw
        self.b -= learning_rate * db
        return dx
