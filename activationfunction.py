"""
Contains activation functions designed for neural networks.
"""

from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    """
    Abstract base class for activation functions.

    Subclasses must implement both the __call__ method (the function itself)
    and the derivative method (its first derivative).
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the activation function.

        Parameters
        ----------
        x: np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Output after applying the activation function element-wise.
        """

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function.

        Parameters
        ----------
        x: np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Derivative of the activation function evaluated element-wise.
        """


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.

    The sigmoid is defined as:

        σ(x) = 1 / (1 + exp(-x))

    Its derivative is:

        σ'(x) = σ(x) * (1 - σ(x))
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid function element-wise to the input array.

        Parameters
        ----------
        x: np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Output after applying the activation function element-wise.
        """
        return 1/(1+np.exp(-x))

    def derivative(self, x) -> np.ndarray:
        """
        Compute the derivative of the sigmoid function element-wise.

        Parameters
        ----------
        x: np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Derivative of the activation function evaluated element-wise.
        """
        return Sigmoid(x) * (1 - Sigmoid(x))


Sigmoid = Sigmoid()

class ReLU(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.

    The ReLU is defined as:

        ReLU(x) = max(0, x)

    Its derivative is:

        ReLU'(x) = 1 if x > 0, else 0
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the ReLU function element-wise to the input array.

        Parameters
        ----------
        x: np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Output after applying the activation function element-wise.
        """
        return np.maximum(x, 0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the ReLU function element-wise.

        Parameters
        ----------
        x: np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Derivative of the activation function evaluated element-wise.
        """
        return np.where(x > 0, 1, 0)

ReLU = ReLU()
