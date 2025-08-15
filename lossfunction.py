import numpy as np


class LossFunction:
    def __init__(self):
        raise NotImplementedError()

    def __call__(self, x):
        raise NotImplementedError()

    def derivative(self, x):
        raise NotImplementedError()

class MSE(LossFunction):
    def __init__(self):
        ()

    def __call__(self, y_pred: np.array, y_true: np.array):
        return np.mean((y_pred - y_true) ** 2)

    def derivative(self, y_pred: np.array, y_true: np.array):
        return 2 * (y_pred - y_true) / y_true.shape[0]