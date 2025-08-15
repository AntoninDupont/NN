import numpy as np
from layer import Layer
from lossfunction import LossFunction


class FullyConnectedNetwork:
    def __init__(self, layers: 'list[Layer]'):
        self.layers = layers

    def _forward(self, X: np.array):
        y_n = X
        for layer in self.layers:
            y_n = layer._forward(y_n)
        return y_n

    def _backward(self, dloss: np.array, learning_rate: float):
        grad = dloss
        for layer in self.layers[::-1]:
            grad = layer._backward(grad, learning_rate)

    def fit(self, X: np.array, y: np.array, epochs: int, learning_rate: float, loss: LossFunction):
        for epoch in range(epochs):
            y_pred = self._forward(X)
            loss_val = loss(y_pred, y)
            dloss = loss.derivative(y_pred, y)
            self._backward(dloss, learning_rate)
            print(f'Epoch {epoch} - Loss: {loss_val}')
