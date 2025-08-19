"""
Contains a fully connected (feedforward) neural network class.
"""

import numpy as np
from .layer import Layer
from .lossfunction import LossFunction


class FullyConnectedNetwork:
    """
    Fully Connected Neural Network.

    This class allows you to build a network of layers, perform predictions,
    compute backpropagation, and train the network.

    Attributes
    ----------
    layers : list[Layer]
        List of layers in the network in forward order.
    """

    def __init__(self, layers: 'list[Layer]'):
        """
        Initialize the network with a list of layers.

        Parameters
        ----------
        layers : list[Layer]
            List of Layer objects representing the layers of the network.
        """
        self.layers = layers

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass on the input data.

        Parameters
        ----------
        x : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Network output after forward propagation.
        """
        y_n = x
        for layer in self.layers:
            y_n = layer.forward(y_n)
        return y_n

    def _backward(self, dloss: np.ndarray, learning_rate: float):
        """
        Perform backpropagation and update the weights.

        Parameters
        ----------
        dloss : np.ndarray
            Gradient of the loss function with respect to the network output.
        learning_rate : float
            Learning rate for weight updates.
        """
        grad = dloss
        for layer in self.layers[::-1]:
            grad = layer.backward(grad, learning_rate)

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            epochs: int,
            learning_rate: float,
            loss: LossFunction):
        """
        Train the network on the provided data.

        Parameters
        ----------
        x : np.ndarray
            Input data of shape (n_samples, n_features).
        y : np.ndarray
            Target labels.
        epochs : int
            Number of iterations over the dataset.
        learning_rate : float
            Learning rate for weight updates.
        loss : LossFunction
            Loss function object to use during training.

        Prints
        ------
        Loss value at each epoch.
        """
        for epoch in range(epochs):
            y_pred = self.predict(x)
            loss_val = loss(y_pred, y)
            dloss = loss.derivative(y_pred, y)
            self._backward(dloss, learning_rate)
            print(f'Epoch {epoch} - Loss: {loss_val}')
