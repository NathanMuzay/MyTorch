import sys
import os

from src.neural_network.model import Sequential
from src.neural_network.layers import Dense, Dropout
from src.neural_network.activations import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
from src.chess_engine.encoders import input_dimension, model_encoding
from src.data.labels import CLASSES_3, CLASSES_5


def print_help():
    print(
        "USAGE\n"
        "./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]\n\n"
        "DESCRIPTION\n"
        "config_file_i Configuration file containing description of a neural network we want to\n"
        "generate.\n"
        "nb_i Number of neural networks to generate based on the configuration file."
    )


def error_exit(msg: str):
    sys.stderr.write(f"Error: {msg}\n")
    sys.exit(84)


def parse_config_line(line: str):
    parts = line.strip().split()
    if not parts:
        return None
    ltype = parts[0].lower()
    kwargs = {}
    for item in parts[1:]:
        if "=" in item:
            k, v = item.split("=", 1)
            kwargs[k.lower()] = v

    if ltype == "dense":
        in_f = int(kwargs.get("in", kwargs.get("in_features", 773)))
        out_f = int(kwargs.get("out", kwargs.get("out_features", 64)))
        init = kwargs.get("init", "he")
        return Dense(in_features=in_f, out_features=out_f, weight_initializer=init)
    elif ltype in ("batchnorm", "batchnorm1d"):
        from src.neural_network.layers import BatchNorm1d
        num_f = int(kwargs.get("features", kwargs.get("num_features", kwargs.get("dim", 64))))
        eps = float(kwargs.get("eps", 1e-5))
        mom = float(kwargs.get("momentum", 0.1))
        return BatchNorm1d(num_features=num_f, eps=eps, momentum=mom)
    elif ltype == "dropout":
        p = float(kwargs.get("p", 0.2))
        return Dropout(p=p)
    elif ltype == "relu":
        return ReLU()
    elif ltype == "leakyrelu":
        alpha = float(kwargs.get("alpha", 0.01))
        return LeakyReLU(alpha=alpha)
    elif ltype == "sigmoid":
        return Sigmoid()
    elif ltype == "tanh":
        return Tanh()
    elif ltype == "softmax":
        return Softmax()
    else:
        raise ValueError(f"Unknown layer type: {ltype}")


def build_model_from_config(config_path: str) -> Sequential:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    model = Sequential()
    with open(config_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            try:
                parts = line_str.split()
                if parts[0].lower() == "encoding":
                    if len(parts) != 2 or "encoding" in model.metadata:
                        raise ValueError("Use a single 'encoding basic' or 'encoding tactical' directive")
                    input_dimension(parts[1])
                    model.metadata.update(encoding=parts[1], encoding_version=1)
                    continue
                layer = parse_config_line(line_str)
                if layer:
                    model.add(layer)
            except Exception as e:
                raise ValueError(f"Error parsing {config_path}:{line_num}: {e}")

    if not model.layers:
        raise ValueError(f"Config file {config_path} resulted in an empty model.")
    if "encoding" in model.metadata:
        model_encoding(model)
        if model.output_dim in (3, 5):
            model.classes = list(CLASSES_5 if model.output_dim == 5 else CLASSES_3)
    return model


def main():
    args = sys.argv[1:]
    if not args:
        error_exit("No arguments provided. Run with --help for usage.")

    if "-h" in args or "--help" in args:
        print_help()
        sys.exit(0)

    if len(args) % 2 != 0:
        error_exit("Arguments must be pairs of [config_file nb].")

    pairs = []
    for i in range(0, len(args), 2):
        conf_file = args[i]
        try:
            nb = int(args[i + 1])
            if nb <= 0:
                raise ValueError()
        except ValueError:
            error_exit(f"Invalid count '{args[i + 1]}'. Must be a positive integer.")
        pairs.append((conf_file, nb))

    for conf_file, count in pairs:
        base_name = os.path.splitext(conf_file)[0]
        for idx in range(1, count + 1):
            try:
                model = build_model_from_config(conf_file)
                out_filename = f"{base_name}_{idx}.nn"
                model.save(out_filename)
            except Exception as e:
                error_exit(f"Failed to generate network from {conf_file}: {e}")

    sys.exit(0)


if __name__ == "__main__":
    main()
