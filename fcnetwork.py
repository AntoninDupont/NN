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
    layers: list[Layer]
        List of layers in the network in forward order.
    """

    def __init__(self, layers: 'list[Layer]'):
        """
        Initialize the network with a list of layers.

        Parameters
        ----------
        layers: list[Layer]
            List of Layer objects representing the layers of the network.
        """
        self.layers = layers

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass on the input data.

        Parameters
        ----------
        x: np.ndarray
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
        dloss: np.ndarray
            Gradient of the loss function with respect to the network output.
        learning_rate: float
            Learning rate for weight updates.
        """
        grad = dloss
        for layer in self.layers[::-1]:
            grad = layer.backward(grad, learning_rate)

    def fit(
            self,
            train_set: 'tuple[np.ndarray, np.ndarray]',
            val_set: 'tuple[np.ndarray, np.ndarray]',
            epochs: int,
            learning_rate: float,
            loss: LossFunction):
        """
        Train the network on the provided data.

        Parameters
        ----------
        train_set: tuple[np.ndarray, np.ndarray]
            Tuple (x_train, y_train) with training data and labels.
        val_set: tuple[np.ndarray, np.ndarray]
            Tuple (x_val, y_val) with validation data and labels, or None if no validation is used.
        epochs: int
            Number of training epochs.
        learning_rate: float
            Step size for parameter updates.
        loss: LossFunction
            Loss function instance with __call__ and derivative methods.
        """
        x_train, y_train = train_set
        for epoch in range(epochs):
            y_pred = self.predict(x_train)
            loss_value = loss(y_pred, y_train)
            dloss = loss.derivative(y_pred, y_train)
            self._backward(dloss, learning_rate)
            if val_set:
                y_pred_test = self.predict(val_set[0])
                loss_value_val = loss(y_pred_test, val_set[1])
                print(f'Epoch {epoch} - Loss: {loss_value} - Loss (val): {loss_value_val}')
            else:
                print(f'Epoch {epoch} - Loss: {loss_value}')
