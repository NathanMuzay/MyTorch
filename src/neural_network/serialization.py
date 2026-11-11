import json
import base64
import numpy as np
from typing import Dict, Any, List


def ndarray_to_b64(arr: np.ndarray) -> Dict[str, Any]:
    """Serialize numpy array to compact base64 representation."""
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def b64_to_ndarray(obj: Dict[str, Any]) -> np.ndarray:
    """Deserialize numpy array from base64 representation."""
    raw_bytes = base64.b64decode(obj["data"])
    arr = np.frombuffer(raw_bytes, dtype=np.dtype(obj["dtype"]))
    return arr.reshape(obj["shape"]).copy()


def save_network(filepath: str, layers_desc: List[Dict[str, Any]], weights: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
    payload = {
        "format": "my_torch_network",
        "version": "1.0",
        "metadata": metadata or {},
        "layers": layers_desc,
        "weights": weights,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_network(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "my_torch_network":
        raise ValueError(f"Invalid neural network format in {filepath}")
    return data
