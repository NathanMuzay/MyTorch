import numpy as np


class Loss:
    """Abstract base class for loss functions."""

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return self.forward(predictions, targets)

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross Entropy loss for multi-class classification.
    Expects one-hot encoded targets or class indices.
    Supports optional class_weights (shape: (C,)) to prioritize rare or critical classes (e.g. Checkmates).
    """

    def __init__(self, epsilon: float = 1e-15, class_weights: np.ndarray | None = None):
        self.epsilon = epsilon
        self.class_weights = class_weights
        self.last_predictions: np.ndarray | None = None
        self.last_targets: np.ndarray | None = None
        self.weights_expanded: np.ndarray | None = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        self.last_predictions = predictions

        # Convert targets to one-hot if given as 1D class indices
        if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
            flat_targets = targets.flatten().astype(int)
            num_classes = predictions.shape[1]
            one_hot = np.zeros_like(predictions)
            one_hot[np.arange(len(flat_targets)), flat_targets] = 1.0
            self.last_targets = one_hot
        else:
            self.last_targets = targets

        clipped = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)

        if self.class_weights is not None:
            # Broadcast weights: (1, C)
            weights = np.asarray(self.class_weights).reshape(1, -1)
            self.weights_expanded = np.sum(self.last_targets * weights, axis=-1, keepdims=True)
            weighted_losses = -np.sum(self.last_targets * np.log(clipped) * weights, axis=-1)
            return float(np.mean(weighted_losses))

        sample_losses = -np.sum(self.last_targets * np.log(clipped), axis=-1)
        return float(np.mean(sample_losses))

    def backward(self) -> np.ndarray:
        if self.last_predictions is None or self.last_targets is None:
            raise RuntimeError("Cannot compute backward before forward.")
        clipped = np.clip(self.last_predictions, self.epsilon, 1.0 - self.epsilon)
        batch_size = self.last_predictions.shape[0]
        targets = self.last_targets
        if self.class_weights is not None:
            targets = targets * np.asarray(self.class_weights).reshape(1, -1)
        return -targets / clipped / batch_size

    def backward_softmax(self) -> np.ndarray:
        """Stable derivative with respect to logits for a terminal softmax."""
        if self.last_predictions is None or self.last_targets is None:
            raise RuntimeError("Cannot compute backward before forward.")
        targets = self.last_targets
        if self.class_weights is not None:
            targets = targets * np.asarray(self.class_weights).reshape(1, -1)
        return (self.last_predictions * targets.sum(axis=-1, keepdims=True) - targets) / len(targets)



class MeanSquaredError(Loss):
    """Mean Squared Error (MSE) loss: L = 1/N * sum((y_pred - y_true)^2)"""

    def __init__(self):
        self.last_predictions: np.ndarray | None = None
        self.last_targets: np.ndarray | None = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        self.last_predictions = predictions
        self.last_targets = targets
        return float(np.mean((predictions - targets) ** 2))

    def backward(self) -> np.ndarray:
        if self.last_predictions is None or self.last_targets is None:
            raise RuntimeError("Cannot compute backward before forward.")
        batch_size = self.last_predictions.shape[0]
        return 2.0 * (self.last_predictions - self.last_targets) / batch_size


class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy loss."""

    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon
        self.last_predictions: np.ndarray | None = None
        self.last_targets: np.ndarray | None = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        self.last_predictions = predictions
        self.last_targets = targets
        clipped = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        loss = - (targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        if self.last_predictions is None or self.last_targets is None:
            raise RuntimeError("Cannot compute backward before forward.")
        batch_size = self.last_predictions.shape[0]
        clipped = np.clip(self.last_predictions, self.epsilon, 1.0 - self.epsilon)
        grad = ((clipped - self.last_targets) / (clipped * (1.0 - clipped))) / batch_size
        return grad


class FocalLoss(Loss):
    """
    Multi-Class Focal Loss: FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples and forces the model to focus on hard misclassifications
    (such as distinguishing subtle Check vs Checkmate positions).
    """

    def __init__(self, gamma: float = 2.0, alpha: np.ndarray | None = None, epsilon: float = 1e-15):
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative")
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.last_predictions: np.ndarray | None = None
        self.last_targets: np.ndarray | None = None

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        self.last_predictions = predictions

        # Convert targets to one-hot if given as 1D class indices
        if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
            flat_targets = targets.flatten().astype(int)
            num_classes = predictions.shape[1]
            one_hot = np.zeros_like(predictions)
            one_hot[np.arange(len(flat_targets)), flat_targets] = 1.0
            self.last_targets = one_hot
        else:
            self.last_targets = targets

        clipped = np.clip(predictions, self.epsilon, 1.0 - self.epsilon)
        p_t = np.sum(self.last_targets * clipped, axis=-1, keepdims=True)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_tensor = np.asarray(self.alpha).reshape(1, -1)
            alpha_t = np.sum(self.last_targets * alpha_tensor, axis=-1, keepdims=True)
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * np.sum(self.last_targets * np.log(clipped), axis=-1, keepdims=True)
        return float(np.mean(loss))

    def _gradient_terms(self):
        if self.last_predictions is None or self.last_targets is None:
            raise RuntimeError("Cannot compute backward before forward.")
        clipped = np.clip(self.last_predictions, self.epsilon, 1.0 - self.epsilon)
        p_t = np.sum(self.last_targets * clipped, axis=-1, keepdims=True)
        remaining = np.maximum(1.0 - p_t, self.epsilon)
        modulation = remaining ** self.gamma
        derivative = (-self.gamma * remaining ** (self.gamma - 1.0)
                      if self.gamma else np.zeros_like(p_t))
        cross_entropy = -np.sum(self.last_targets * np.log(clipped), axis=-1, keepdims=True)
        alpha_t = 1.0
        if self.alpha is not None:
            alpha_t = np.sum(self.last_targets * np.asarray(self.alpha).reshape(1, -1),
                             axis=-1, keepdims=True)
        return clipped, p_t, modulation, derivative, cross_entropy, alpha_t

    def backward(self) -> np.ndarray:
        """Derivative with respect to probabilities, including the focal factor."""
        probabilities, _, modulation, derivative, cross_entropy, alpha_t = self._gradient_terms()
        return (alpha_t * self.last_targets *
                (derivative * cross_entropy - modulation / probabilities) / len(probabilities))

    def backward_softmax(self) -> np.ndarray:
        """Fused derivative with respect to logits; no second softmax backward."""
        _, p_t, modulation, derivative, cross_entropy, alpha_t = self._gradient_terms()
        probabilities = self.last_predictions
        targets = self.last_targets
        grad = (derivative * cross_entropy * probabilities * (targets - p_t)
                + modulation * (probabilities * targets.sum(axis=-1, keepdims=True) - targets))
        return alpha_t * grad / len(probabilities)


def get_loss(name: str) -> Loss:
    name_clean = name.lower().replace("_", "").replace("-", "")
    mapping = {
        "categoricalcrossentropy": CategoricalCrossEntropy(),
        "crossentropy": CategoricalCrossEntropy(),
        "cce": CategoricalCrossEntropy(),
        "focalloss": FocalLoss(),
        "focal": FocalLoss(),
        "meansquarederror": MeanSquaredError(),
        "mse": MeanSquaredError(),
        "binarycrossentropy": BinaryCrossEntropy(),
        "bce": BinaryCrossEntropy(),
    }
    if name_clean in mapping:
        return mapping[name_clean]
    raise ValueError(f"Unknown loss function: {name}")


__all__ = ['Loss', 'CategoricalCrossEntropy', 'MeanSquaredError', 'BinaryCrossEntropy', 'FocalLoss', 'get_loss']
