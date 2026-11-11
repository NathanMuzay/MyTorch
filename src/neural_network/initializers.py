import numpy as np


class Initializer:
    def __call__(self, shape: tuple) -> np.ndarray:
        raise NotImplementedError


class HeNormal(Initializer):
    """He (Kaiming) normal initialization. Best for ReLU / LeakyReLU."""
    def __call__(self, shape: tuple) -> np.ndarray:
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape) * std


class HeUniform(Initializer):
    """He (Kaiming) uniform initialization."""
    def __call__(self, shape: tuple) -> np.ndarray:
        fan_in = shape[0]
        limit = np.sqrt(6.0 / fan_in)
        return np.random.uniform(-limit, limit, size=shape)


class XavierNormal(Initializer):
    """Xavier (Glorot) normal initialization. Best for Sigmoid / Tanh."""
    def __call__(self, shape: tuple) -> np.ndarray:
        fan_in, fan_out = shape[0], shape[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape) * std


class XavierUniform(Initializer):
    """Xavier (Glorot) uniform initialization."""
    def __call__(self, shape: tuple) -> np.ndarray:
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)


class LeCunNormal(Initializer):
    """LeCun normal initialization."""
    def __call__(self, shape: tuple) -> np.ndarray:
        fan_in = shape[0]
        std = np.sqrt(1.0 / fan_in)
        return np.random.randn(*shape) * std


class Zeros(Initializer):
    def __call__(self, shape: tuple) -> np.ndarray:
        return np.zeros(shape)


def get_initializer(name: str) -> Initializer:
    name_clean = name.lower().replace("_", "").replace("-", "")
    mapping = {
        "henormal": HeNormal(),
        "he": HeNormal(),
        "heuniform": HeUniform(),
        "xaviernormal": XavierNormal(),
        "xavier": XavierNormal(),
        "glorot": XavierNormal(),
        "xavieruniform": XavierUniform(),
        "lecunnormal": LeCunNormal(),
        "lecun": LeCunNormal(),
        "zeros": Zeros(),
    }
    if name_clean in mapping:
        return mapping[name_clean]
    raise ValueError(f"Unknown initializer: {name}")


__all__ = ['Initializer', 'HeNormal', 'HeUniform', 'XavierNormal', 'XavierUniform', 'LeCunNormal', 'Zeros', 'get_initializer']
