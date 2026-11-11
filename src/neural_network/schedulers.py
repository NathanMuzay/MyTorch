import math
from src.neural_network.optimizers import Optimizer


class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.base_lr = optimizer.learning_rate
        self.last_epoch = last_epoch
        self.step()

    def get_lr(self) -> float:
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        new_lr = self.get_lr()
        self.optimizer.learning_rate = new_lr
        return new_lr


class StepLR(LRScheduler):
    """
    Decays the learning rate by gamma every step_size epochs:
    lr = base_lr * (gamma ** (epoch // step_size))
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.5, last_epoch: int = -1):
        self.step_size = max(1, step_size)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))


class ExponentialLR(LRScheduler):
    """
    Decays learning rate exponentially each epoch:
    lr = base_lr * (gamma ** epoch)
    """

    def __init__(self, optimizer: Optimizer, gamma: float = 0.95, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** self.last_epoch)


class CosineAnnealingLR(LRScheduler):
    """
    Cosine Annealing learning rate schedule:
    lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max))
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 1e-6, last_epoch: int = -1):
        self.T_max = max(1, T_max)
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        epoch = min(self.last_epoch, self.T_max)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + math.cos(math.pi * epoch / self.T_max))


__all__ = ['LRScheduler', 'StepLR', 'ExponentialLR', 'CosineAnnealingLR']
