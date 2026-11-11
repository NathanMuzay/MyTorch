import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from src.neural_network.layers import Layer, Dense, Dropout, BatchNorm1d
from src.neural_network.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
from src.neural_network.losses import Loss, CategoricalCrossEntropy, FocalLoss, get_loss
from src.neural_network.optimizers import Optimizer, get_optimizer
from src.neural_network.schedulers import LRScheduler
from src.neural_network.serialization import save_network, load_network, ndarray_to_b64, b64_to_ndarray


class Sequential:
    """
    Sequential container of neural network layers.
    """

    def __init__(self, layers: Optional[List[Layer]] = None):
        self.layers: List[Layer] = []
        if layers:
            for l in layers:
                self.add(l)
        self.optimizer: Optional[Optimizer] = None
        self.loss: Optional[Loss] = None
        self.classes: Optional[List[str]] = None
        self.metadata: Dict[str, Any] = {}
        self.best_epoch: Optional[int] = None

    @property
    def output_dim(self) -> int:
        for layer in reversed(self.layers):
            if hasattr(layer, "out_features"):
                return layer.out_features
        return 3

    def add(self, layer: Layer):
        self.layers.append(layer)

    def compile(self, optimizer: Optimizer | str = "adam", loss: Loss | str = "crossentropy"):
        if isinstance(optimizer, str):
            self.optimizer = get_optimizer(optimizer)
        else:
            self.optimizer = optimizer

        if isinstance(loss, str):
            self.loss = get_loss(loss)
        else:
            self.loss = loss

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        out = inputs
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, loss_gradient: np.ndarray, *, from_logits: bool = False):
        """Skip terminal softmax only when the loss already differentiated it."""
        if from_logits and (not self.layers or not isinstance(self.layers[-1], Softmax)):
            raise ValueError("A logits gradient requires a terminal Softmax layer")
        grad = loss_gradient
        layers = self.layers[:-1] if from_logits else self.layers
        for layer in reversed(layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs, training=False)

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray, *, return_predictions: bool = False):
        """
        Evaluate model on dataset.
        Returns (loss, accuracy), optionally followed by predictions for diagnostics.
        """
        preds = self.predict(inputs)
        if self.loss is None:
            self.loss = CategoricalCrossEntropy()
        loss_val = self.loss(preds, targets)

        # Accuracy
        if targets.ndim > 1 and targets.shape[1] > 1:
            true_classes = np.argmax(targets, axis=1)
        else:
            true_classes = targets.flatten().astype(int)
        pred_classes = np.argmax(preds, axis=1)
        acc = float(np.mean(true_classes == pred_classes))
        if return_predictions:
            return loss_val, acc, preds
        return loss_val, acc

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        shuffle: bool = True,
        verbose: bool = True,
        early_stopping_patience: Optional[int] = None,
        scheduler: Optional[LRScheduler] = None,
        restore_best_weights: bool = False,
        monitor: str = "val_loss",
    ) -> Dict[str, List[float]]:
        """
        Train neural network with mini-batch gradient descent and optional learning rate scheduler.
        """
        if self.optimizer is None or self.loss is None:
            raise RuntimeError("Model must be compiled with an optimizer and loss before training.")
        if monitor not in ("val_loss", "val_accuracy"):
            raise ValueError("monitor must be val_loss or val_accuracy")
        if epochs <= 0 or batch_size <= 0 or len(X_train) == 0:
            raise ValueError("Training data, epochs and batch_size must be non-empty/positive")
        if early_stopping_patience is not None and early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")

        history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "lr": [],
        }

        num_samples = X_train.shape[0]
        best_value = float("inf") if monitor == "val_loss" else -float("inf")
        best_checkpoint_loss = float("inf")
        best_state = None
        self.best_epoch = None
        patience_counter = 0
        fused_softmax = (bool(self.layers) and isinstance(self.layers[-1], Softmax)
                         and isinstance(self.loss, (CategoricalCrossEntropy, FocalLoss)))

        for epoch in range(1, epochs + 1):
            if shuffle:
                indices = np.random.permutation(num_samples)
                X_train_shuffled = X_train[indices]
                y_train_shuffled = y_train[indices]
            else:
                X_train_shuffled = X_train
                y_train_shuffled = y_train

            total_loss = 0.0
            correct = 0

            for i in range(0, num_samples, batch_size):
                batch_X = X_train_shuffled[i : i + batch_size]
                batch_y = y_train_shuffled[i : i + batch_size]

                # Forward pass
                preds = self.forward(batch_X, training=True)

                # Compute loss
                loss_val = self.loss(preds, batch_y)
                total_loss += loss_val * len(batch_X)

                # Accuracy tracking
                if batch_y.ndim > 1 and batch_y.shape[1] > 1:
                    batch_labels = np.argmax(batch_y, axis=1)
                else:
                    batch_labels = batch_y.flatten().astype(int)
                correct += np.sum(np.argmax(preds, axis=1) == batch_labels)

                # Backward pass & parameter update
                loss_grad = self.loss.backward_softmax() if fused_softmax else self.loss.backward()
                self.backward(loss_grad, from_logits=fused_softmax)
                self.optimizer.step(self.layers)

            train_loss = float(total_loss / num_samples)
            train_acc = float(correct / num_samples)
            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["lr"].append(self.optimizer.learning_rate)

            # Validation step
            if validation_data is not None:
                val_X, val_y = validation_data
                val_loss, val_acc = self.evaluate(val_X, val_y)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

                if verbose:
                    print(
                        f"Epoch {epoch}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} "
                        f"- val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                    )

                value = val_loss if monitor == "val_loss" else val_acc
                if not np.isfinite(value) or not np.isfinite(val_loss):
                    raise ValueError("Non-finite validation metric")
                improved = value < best_value if monitor == "val_loss" else value > best_value
                better_checkpoint = improved or (monitor == "val_accuracy" and value == best_value
                                                   and val_loss < best_checkpoint_loss)
                if improved:
                    best_value = value
                    patience_counter = 0
                else:
                    patience_counter += 1
                if better_checkpoint:
                    self.best_epoch = epoch
                    best_checkpoint_loss = val_loss
                    if restore_best_weights:
                        # Include BatchNorm running statistics, not just trainable parameters.
                        best_state = [
                            {name: value.copy() for name, value in vars(layer).items()
                             if name in ("weights", "bias", "gamma", "beta", "running_mean", "running_var")}
                            for layer in self.layers
                        ]
                if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}")

            # Step learning rate scheduler if present
            if scheduler is not None:
                scheduler.step()

        if best_state is not None:
            for layer, state in zip(self.layers, best_state):
                for name, value in state.items():
                    getattr(layer, name)[...] = value
            if verbose:
                print(f"Restored best weights from epoch {self.best_epoch} ({monitor}={best_value:.4f})")
        return history

    def save(self, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """Save model architecture and weights to .nn file."""
        layers_desc = []
        weights_data = []

        for layer in self.layers:
            if isinstance(layer, Dense):
                layers_desc.append({
                    "type": "Dense",
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                })
                weights_data.append({
                    "weights": ndarray_to_b64(layer.weights),
                    "bias": ndarray_to_b64(layer.bias),
                })
            elif isinstance(layer, BatchNorm1d):
                layers_desc.append({
                    "type": "BatchNorm1d",
                    "num_features": layer.num_features,
                    "eps": layer.eps,
                    "momentum": layer.momentum,
                })
                weights_data.append({
                    "gamma": ndarray_to_b64(layer.gamma),
                    "beta": ndarray_to_b64(layer.beta),
                    "running_mean": ndarray_to_b64(layer.running_mean),
                    "running_var": ndarray_to_b64(layer.running_var),
                })
            elif isinstance(layer, Dropout):
                layers_desc.append({
                    "type": "Dropout",
                    "p": layer.p,
                    "mc_dropout": layer.mc_dropout,
                })
            elif isinstance(layer, ReLU):
                layers_desc.append({"type": "ReLU"})
            elif isinstance(layer, LeakyReLU):
                layers_desc.append({"type": "LeakyReLU", "alpha": layer.alpha})
            elif isinstance(layer, Sigmoid):
                layers_desc.append({"type": "Sigmoid"})
            elif isinstance(layer, Tanh):
                layers_desc.append({"type": "Tanh"})
            elif isinstance(layer, Softmax):
                layers_desc.append({"type": "Softmax"})

        meta = dict(self.metadata)
        if metadata:
            meta.update(metadata)
        if self.classes and "classes" not in meta:
            meta["classes"] = list(self.classes)
        if isinstance(self.loss, FocalLoss):
            meta["loss"] = {"name": "focal", "gamma": self.loss.gamma, "epsilon": self.loss.epsilon,
                            "alpha": np.asarray(self.loss.alpha).tolist() if self.loss.alpha is not None else None}
        elif isinstance(self.loss, CategoricalCrossEntropy):
            meta["loss"] = {"name": "cce", "epsilon": self.loss.epsilon,
                            "class_weights": np.asarray(self.loss.class_weights).tolist()
                            if self.loss.class_weights is not None else None}

        save_network(filepath, layers_desc, weights_data, metadata=meta)

    @classmethod
    def load(cls, filepath: str) -> "Sequential":
        """Load model architecture and weights from .nn file."""
        data = load_network(filepath)
        model = cls()
        model.metadata = data.get("metadata", {})
        if "classes" in model.metadata:
            model.classes = list(model.metadata["classes"])
        weights_idx = 0

        for layer_desc in data["layers"]:
            ltype = layer_desc["type"]
            if ltype == "Dense":
                dense_layer = Dense(
                    in_features=layer_desc["in_features"],
                    out_features=layer_desc["out_features"],
                )
                w_info = data["weights"][weights_idx]
                dense_layer.weights = b64_to_ndarray(w_info["weights"])
                dense_layer.bias = b64_to_ndarray(w_info["bias"])
                dense_layer.params["weights"] = dense_layer.weights
                dense_layer.params["bias"] = dense_layer.bias
                model.add(dense_layer)
                weights_idx += 1
            elif ltype == "BatchNorm1d":
                bn_layer = BatchNorm1d(
                    num_features=layer_desc["num_features"],
                    eps=layer_desc.get("eps", 1e-5),
                    momentum=layer_desc.get("momentum", 0.1),
                )
                w_info = data["weights"][weights_idx]
                bn_layer.gamma = b64_to_ndarray(w_info["gamma"])
                bn_layer.beta = b64_to_ndarray(w_info["beta"])
                bn_layer.running_mean = b64_to_ndarray(w_info["running_mean"])
                bn_layer.running_var = b64_to_ndarray(w_info["running_var"])
                bn_layer.params["gamma"] = bn_layer.gamma
                bn_layer.params["beta"] = bn_layer.beta
                model.add(bn_layer)
                weights_idx += 1
            elif ltype == "Dropout":
                model.add(Dropout(p=layer_desc["p"], mc_dropout=layer_desc.get("mc_dropout", False)))
            elif ltype == "ReLU":
                model.add(ReLU())
            elif ltype == "LeakyReLU":
                model.add(LeakyReLU(alpha=layer_desc.get("alpha", 0.01)))
            elif ltype == "Sigmoid":
                model.add(Sigmoid())
            elif ltype == "Tanh":
                model.add(Tanh())
            elif ltype == "Softmax":
                model.add(Softmax())
            else:
                raise ValueError(f"Unknown layer type in saved model: {ltype}")

        loss_config = dict(model.metadata.get("loss", {}))
        loss_name = loss_config.pop("name", None)
        if loss_name == "focal":
            model.loss = FocalLoss(**loss_config)
        elif loss_name == "cce":
            model.loss = CategoricalCrossEntropy(**loss_config)
        return model
