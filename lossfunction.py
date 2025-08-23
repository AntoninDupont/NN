"""
Contains loss function classes made for neural network training.
"""

from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    """
    Abstract base class for loss functions.

    All loss functions must implement the __call__ method for computing the
    loss and the derivative method for computing the gradient.
    """

    @abstractmethod
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the loss.

        Parameters
        ----------
        y_pred: np.ndarray
            Predicted values of shape (n_samples, ...).
        y_true: np.ndarray
            True target values of shape (n_samples, ...).

        Returns
        -------
        float
            The computed loss value.
        """

    @abstractmethod
    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the loss with respect to predictions.

        Parameters
        ----------
        y_pred: np.ndarray
            Predicted values.
        y_true: np.ndarray
            True values.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to predictions.
        """


class MSE(LossFunction):
    """
    Mean Squared Error loss function.

    Computes the mean squared error between predicted and true values,
    and its derivative.
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the mean squared error.

        Parameters
        ----------
        y_pred: np.ndarray
            Predicted values of shape (n_samples, ...).
        y_true: np.ndarray
            True target values of shape (n_samples, ...).

        Returns
        -------
        float
            Mean squared error.
        """
        return np.mean((y_pred - y_true) ** 2)

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the MSE with respect to predictions.

        Parameters
        ----------
        y_pred: np.ndarray
            Predicted values.
        y_true: np.ndarray
            True values.

        Returns
        -------
        np.ndarray
            Gradient of the MSE with respect to y_pred.
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]

MSE = MSE()


class BinaryCrossEntropy(LossFunction):
    """
    Binary Cross Entropy loss function.

    Computes the binary cross entropy between predicted and true values,
    and its derivative.
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the binary cross entropy.

        Parameters
        ----------
        y_pred: np.ndarray
            Predicted values of shape (n_samples, ...).
        y_true: np.ndarray
            True target values of shape (n_samples, ...).

        Returns
        -------
        float
            Binary cross entropy.
        """
        y_true = y_true.reshape(y_pred.shape)
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the binary cross entropy with respect to predictions.

        Parameters
        ----------
        y_pred: np.ndarray
            Predicted values.
        y_true: np.ndarray
            True values.

        Returns
        -------
        np.ndarray
            Gradient of the binary cross entropy with respect to y_pred.
        """
        y_true = y_true.reshape(y_pred.shape)
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])

BinaryCrossEntropy = BinaryCrossEntropy()
