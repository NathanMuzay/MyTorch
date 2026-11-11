"""Couches du réseau : interface commune, Dense, Dropout et BatchNorm."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.neural_network.initializers import get_initializer, Initializer


class Layer(ABC):
    """
    Abstract base class for all neural network layers.
    """

    def __init__(self):
        self.trainable = True
        self.params = {}
        self.grads = {}

    @abstractmethod
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Compute forward pass.
        :param inputs: Input tensor of shape (batch_size, input_dim)
        :param training: Flag indicating training mode or inference mode
        :return: Output tensor
        """
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Compute backward pass.
        :param output_gradient: Gradient of loss with respect to layer output
        :return: Gradient of loss with respect to layer input
        """
        pass


class Dense(Layer):
    """
    Fully connected (dense) linear layer: Y = X @ W + b
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_initializer: str | Initializer = "he",
        bias_initializer: str | Initializer = "zeros",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w_init = get_initializer(weight_initializer) if isinstance(weight_initializer, str) else weight_initializer
        b_init = get_initializer(bias_initializer) if isinstance(bias_initializer, str) else bias_initializer

        self.weights = w_init((in_features, out_features))
        self.bias = b_init((1, out_features))

        self.params = {
            "weights": self.weights,
            "bias": self.bias,
        }
        self.grads = {
            "weights": np.zeros_like(self.weights),
            "bias": np.zeros_like(self.bias),
        }

        self.last_input: np.ndarray | None = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.last_input = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        if self.last_input is None:
            raise RuntimeError("Cannot run backward before forward has been called.")

        # Gradients of parameters
        self.grads["weights"] = np.dot(self.last_input.T, output_gradient)
        self.grads["bias"] = np.sum(output_gradient, axis=0, keepdims=True)

        # Gradient with respect to inputs to pass to previous layer
        input_gradient = np.dot(output_gradient, self.weights.T)
        return input_gradient


class Dropout(Layer):
    """
    Dropout layer for regularization to prevent overfitting.
    Includes Monte Carlo (MC) Dropout capability for inference uncertainty.
    """

    def __init__(self, p: float = 0.5, mc_dropout: bool = False):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability p must be in [0, 1), got {p}")
        self.p = p
        self.mc_dropout = mc_dropout
        self.trainable = False
        self.mask: np.ndarray | None = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if (training or self.mc_dropout) and self.p > 0.0:
            # Inverted dropout: scaling by 1 / (1 - p) keeps expected value consistent
            self.mask = (np.random.rand(*inputs.shape) >= self.p) / (1.0 - self.p)
            return inputs * self.mask
        self.mask = None
        return inputs

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        if self.mask is not None:
            return output_gradient * self.mask
        return output_gradient


class BatchNorm1d(Layer):
    """
    Batch Normalization 1D for fully connected activations:
    y = gamma * (x - mean) / sqrt(var + eps) + beta
    Accelerates deep network convergence and acts as mild regularizer.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters (scale and shift)
        self.gamma = np.ones((1, num_features), dtype=np.float32)
        self.beta = np.zeros((1, num_features), dtype=np.float32)

        self.params = {
            "gamma": self.gamma,
            "beta": self.beta,
        }
        self.grads = {
            "gamma": np.zeros_like(self.gamma),
            "beta": np.zeros_like(self.beta),
        }

        # Running statistics for inference
        self.running_mean = np.zeros((1, num_features), dtype=np.float32)
        self.running_var = np.ones((1, num_features), dtype=np.float32)

        # Cache for backpropagation
        self.x_hat: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None
        self.std_inv: Optional[np.ndarray] = None
        self.x_centered: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            batch_size = inputs.shape[0]
            mean = np.mean(inputs, axis=0, keepdims=True)
            self.x_centered = inputs - mean
            self.var = np.var(inputs, axis=0, keepdims=True)
            self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
            self.x_hat = self.x_centered * self.std_inv

            # Update running statistics using exponential moving average
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * self.var

            return self.gamma * self.x_hat + self.beta
        else:
            std_inv = 1.0 / np.sqrt(self.running_var + self.eps)
            x_hat = (inputs - self.running_mean) * std_inv
            return self.gamma * x_hat + self.beta

    def backward(self, loss_gradient: np.ndarray) -> np.ndarray:
        batch_size = loss_gradient.shape[0]

        # Gradients for gamma and beta
        self.grads["gamma"] = np.sum(loss_gradient * self.x_hat, axis=0, keepdims=True)
        self.grads["beta"] = np.sum(loss_gradient, axis=0, keepdims=True)

        # Gradient with respect to normalized inputs
        dx_hat = loss_gradient * self.gamma

        # Analytical gradient with respect to inputs
        dvar = np.sum(dx_hat * self.x_centered * -0.5 * (self.std_inv ** 3), axis=0, keepdims=True)
        dmean = np.sum(dx_hat * -self.std_inv, axis=0, keepdims=True) + dvar * np.mean(-2.0 * self.x_centered, axis=0, keepdims=True)

        dx = dx_hat * self.std_inv + dvar * 2.0 * self.x_centered / batch_size + dmean / batch_size
        return dx


__all__ = ["Layer", "Dense", "Dropout", "BatchNorm1d"]
