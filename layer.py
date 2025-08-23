"""
Contains a Layer class to be included in a neural network class.
"""

import numpy as np
from .activationfunction import ActivationFunction


class Layer:
    """
    Fully connected neural network layer.

    This class represents a single layer in a neural network, including its
    weights, biases, and activation function. Supports forward and backward
    propagation.
    """

    def __init__(
            self,
            n_inputs: int,
            n_neurons: int,
            activation_function: ActivationFunction,
            name: str=None,
            init_method: str='random'):
        """
        Initialize the layer with weights, biases, and an activation function.

        Parameters
        ----------
        n_inputs: int
            Number of input features to this layer.
        n_neurons: int
            Number of neurons in this layer.
        activation_function: ActivationFunction
            Activation function object to apply on the linear combination.
        name: str, optional
            Optional name for the layer.
        init_method: str, default 'random'
            Method to initialize weights. Options: 'random', 'he', 'xavier', 'zeros'.

        Raises
        ------
        ValueError
            If init_method is not options.
        """
        if init_method == 'random':
            self.weights = np.random.rand(n_inputs, n_neurons) / 100
        elif init_method == 'he':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        elif init_method == 'xavier':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        elif init_method == 'zeros':
            self.weights = np.zeros((n_inputs, n_neurons))
        else:
            raise ValueError("init_method not in {'zeros', 'random'}")
        self.b = np.zeros((1, n_neurons))
        self.activation_function = activation_function
        self.name = name

        self._input = None
        self._z = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through this layer.

        Parameters
        ----------
        x: np.ndarray
            Input data of shape (n_samples, n_inputs).

        Returns
        -------
        np.ndarray
            Output of the layer after applying the activation function.
        """
        self._input = x
        self._z = np.dot(x, self.weights) + self.b
        return self.activation_function(self._z)

    def backward(self, da: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Perform backward propagation and update weights and biases.

        Parameters
        ----------
        da: np.ndarray
            Gradient of the loss with respect to the output of this layer.
        learning_rate: float
            Learning rate used to update weights and biases.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the input of this layer.
        """
        dz = da * self.activation_function.derivative(self._z)
        dw = np.dot(self._input.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        dx = np.dot(dz, self.weights.T)
        self.weights -= learning_rate * dw
        self.b -= learning_rate * db
        return dx
