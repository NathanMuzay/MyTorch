import numpy as np
from src.neural_network.layers import Layer


class Activation(Layer):
    """Base class for activation layers."""
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.last_input: np.ndarray | None = None
        self.last_output: np.ndarray | None = None


class ReLU(Activation):
    """Rectified Linear Unit: f(x) = max(0, x)"""
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.last_input = inputs
        return np.maximum(0.0, inputs)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        if self.last_input is None:
            raise RuntimeError("Cannot run backward before forward has been called.")
        grad = output_gradient.copy()
        grad[self.last_input <= 0.0] = 0.0
        return grad


class LeakyReLU(Activation):
    """Leaky ReLU to prevent dying neurons: f(x) = x if x > 0 else alpha * x"""
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.last_input = inputs
        return np.where(inputs > 0.0, inputs, self.alpha * inputs)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        if self.last_input is None:
            raise RuntimeError("Cannot run backward before forward has been called.")
        dx = np.where(self.last_input > 0.0, 1.0, self.alpha)
        return output_gradient * dx


class Sigmoid(Activation):
    """Sigmoid activation: f(x) = 1 / (1 + exp(-x))"""
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        # Clip to prevent numerical overflow in exp(-x)
        clipped = np.clip(inputs, -500.0, 500.0)
        self.last_output = 1.0 / (1.0 + np.exp(-clipped))
        return self.last_output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        if self.last_output is None:
            raise RuntimeError("Cannot run backward before forward has been called.")
        return output_gradient * (self.last_output * (1.0 - self.last_output))


class Tanh(Activation):
    """Hyperbolic tangent: f(x) = tanh(x)"""
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.last_output = np.tanh(inputs)
        return self.last_output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        if self.last_output is None:
            raise RuntimeError("Cannot run backward before forward has been called.")
        return output_gradient * (1.0 - self.last_output ** 2)


class Softmax(Activation):
    """
    Numerically stable Softmax for multi-class classification:
    f(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        # Shift inputs by max for numerical stability (avoids inf)
        shifted = inputs - np.max(inputs, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        self.last_output = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        return self.last_output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Generic vector Jacobian product for Softmax when not directly fused with CrossEntropy.
        """
        if self.last_output is None:
            raise RuntimeError("Cannot run backward before forward has been called.")
        s = self.last_output
        # dS/dx_i = s_i * (1 - s_i) if i == j else -s_i * s_j
        # Efficient batch vector Jacobian product:
        sum_grad_s = np.sum(output_gradient * s, axis=-1, keepdims=True)
        return s * (output_gradient - sum_grad_s)


def get_activation(name: str) -> Activation:
    name_clean = name.lower().replace("_", "").replace("-", "")
    mapping = {
        "relu": ReLU(),
        "leakyrelu": LeakyReLU(),
        "sigmoid": Sigmoid(),
        "tanh": Tanh(),
        "softmax": Softmax(),
    }
    if name_clean in mapping:
        return mapping[name_clean]
    raise ValueError(f"Unknown activation function: {name}")


__all__ = ['Activation', 'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'Softmax', 'get_activation']
