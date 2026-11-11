from abc import ABC, abstractmethod
import numpy as np
from typing import List
from src.neural_network.layers import Layer


class Optimizer(ABC):
    """Abstract base optimizer."""

    def __init__(self, learning_rate: float = 0.001, weight_decay: float = 0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.iterations = 0

    @abstractmethod
    def update(self, layer: Layer):
        pass

    def step(self, layers: List[Layer]):
        self.iterations += 1
        for layer in layers:
            if layer.trainable and layer.params:
                self.update(layer)


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and L2 regularization (weight decay).
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0):
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.velocities = {}

    def update(self, layer: Layer):
        layer_id = id(layer)
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {k: np.zeros_like(v) for k, v in layer.params.items()}

        for name, param in layer.params.items():
            grad = layer.grads[name]
            if self.weight_decay > 0.0 and name == "weights":
                grad = grad + self.weight_decay * param

            if self.momentum > 0.0:
                v = self.velocities[layer_id][name]
                v = self.momentum * v - self.learning_rate * grad
                self.velocities[layer_id][name] = v
                param += v
            else:
                param -= self.learning_rate * grad


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}

    def update(self, layer: Layer):
        layer_id = id(layer)
        if layer_id not in self.m:
            self.m[layer_id] = {k: np.zeros_like(val) for k, val in layer.params.items()}
            self.v[layer_id] = {k: np.zeros_like(val) for k, val in layer.params.items()}

        for name, param in layer.params.items():
            grad = layer.grads[name]
            if self.weight_decay > 0.0 and name == "weights":
                grad = grad + self.weight_decay * param

            # Update biased 1st and 2nd moment estimates
            m = self.m[layer_id][name]
            v = self.v[layer_id][name]

            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)

            self.m[layer_id][name] = m
            self.v[layer_id][name] = v

            # Compute bias-corrected estimates
            m_hat = m / (1.0 - (self.beta1 ** self.iterations))
            v_hat = v / (1.0 - (self.beta2 ** self.iterations))

            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(Optimizer):
    """
    Root Mean Square Propagation (RMSprop) optimizer.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(learning_rate, weight_decay)
        self.rho = rho
        self.epsilon = epsilon
        self.accumulators = {}

    def update(self, layer: Layer):
        layer_id = id(layer)
        if layer_id not in self.accumulators:
            self.accumulators[layer_id] = {k: np.zeros_like(val) for k, val in layer.params.items()}

        for name, param in layer.params.items():
            grad = layer.grads[name]
            if self.weight_decay > 0.0 and name == "weights":
                grad = grad + self.weight_decay * param

            acc = self.accumulators[layer_id][name]
            acc = self.rho * acc + (1.0 - self.rho) * (grad ** 2)
            self.accumulators[layer_id][name] = acc

            param -= self.learning_rate * grad / (np.sqrt(acc) + self.epsilon)


def get_optimizer(name: str, learning_rate: float = 0.001, **kwargs) -> Optimizer:
    name_clean = name.lower().replace("_", "").replace("-", "")
    if name_clean == "adam":
        return Adam(learning_rate=learning_rate, **kwargs)
    elif name_clean in ("sgd", "momentum"):
        return SGD(learning_rate=learning_rate, **kwargs)
    elif name_clean == "rmsprop":
        return RMSprop(learning_rate=learning_rate, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")


__all__ = ['Optimizer', 'SGD', 'Adam', 'RMSprop', 'get_optimizer']
