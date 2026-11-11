"""
MyTorch: Neural network library implemented from scratch for EPITECH.
"""

from src.neural_network.model import Sequential
from src.neural_network.layers import Dense, Dropout
from src.neural_network.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
from src.neural_network.losses import CategoricalCrossEntropy, MeanSquaredError, BinaryCrossEntropy
from src.neural_network.optimizers import SGD, Adam, RMSprop
from src.neural_network.initializers import HeNormal, HeUniform, XavierNormal, XavierUniform, LeCunNormal

__all__ = [
    "Sequential",
    "Dense",
    "Dropout",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "CategoricalCrossEntropy",
    "MeanSquaredError",
    "BinaryCrossEntropy",
    "SGD",
    "Adam",
    "RMSprop",
    "HeNormal",
    "HeUniform",
    "XavierNormal",
    "XavierUniform",
    "LeCunNormal",
]
