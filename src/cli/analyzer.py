#!/usr/bin/env python3
import sys
import os

import numpy as np
from src.neural_network.model import Sequential
from src.chess_engine.encoders import encode_boards_parallel, model_encoding

from src.data.labels import CLASSES_3, MAPPING_5_TO_3, get_model_classes


def print_help():
    print(
        "USAGE\n"
        "./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE\n\n"
        "DESCRIPTION\n"
        "--train Launch the neural network in training mode. Each chessboard in FILE must\n"
        "contain inputs to send to the neural network in FEN notation and the expected output separated by\n"
        "space. If specified, the newly trained neural network will be saved in SAVEFILE. Otherwise,\n"
        "it will be saved in the original LOADFILE.\n"
        "--predict Launch the neural network in prediction mode. Each chessboard in FILE must\n"
        "contain inputs to send to the neural network in FEN notation, and optionally an expected output.\n"
        "--save Save neural network into SAVEFILE. Only works in train mode.\n"
        "LOADFILE File containing an artificial neural network\n"
        "CHESSFILE File containing chessboards"
    )


def error_exit(msg: str):
    sys.stderr.write(f"Error: {msg}\n")
    sys.exit(84)


def parse_arguments():
    args = sys.argv[1:]

    if not args:
        error_exit("No arguments provided. Run with --help for usage.")

    if "-h" in args or "--help" in args:
        print_help()
        sys.exit(0)

    mode = None
    save_file = None
    positional = []

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--predict":
            if mode is not None:
                error_exit("Cannot specify multiple modes.")
            mode = "predict"
        elif arg == "--train":
            if mode is not None:
                error_exit("Cannot specify multiple modes.")
            mode = "train"
        elif arg == "--save":
            if i + 1 >= len(args):
                error_exit("Missing argument for --save.")
            save_file = args[i + 1]
            i += 1
        elif arg.startswith("--"):
            error_exit(f"Unknown option: {arg}")
        else:
            positional.append(arg)
        i += 1

    if mode is None:
        error_exit("Either --predict or --train must be specified.")

    if mode == "predict" and save_file is not None:
        error_exit("--save can only be used with --train.")

    if len(positional) != 2:
        error_exit(f"Expected LOADFILE and CHESSFILE, got {len(positional)} positional arguments.")

    load_file, chess_file = positional[0], positional[1]

    if not os.path.exists(load_file):
        error_exit(f"Model file does not exist: {load_file}")

    if not os.path.exists(chess_file):
        error_exit(f"Chessboard file does not exist: {chess_file}")

    target_save_file = save_file if save_file else load_file
    return mode, load_file, chess_file, target_save_file


def run_predict(model: Sequential, chess_file: str):
    try:
        with open(chess_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        error_exit(f"Failed to read {chess_file}: {e}")

    if not lines:
        sys.exit(0)

    classes = get_model_classes(model)

    fen_candidates = []
    for line_idx, line in enumerate(lines, 1):
        tokens = line.split()
        if len(tokens) >= 6:
            fen_candidates.append(" ".join(tokens[:6]))
        else:
            fen_candidates.append(line)

    try:
        X = encode_boards_parallel(fen_candidates, encoding=model_encoding(model))
    except Exception as e:
        error_exit(f"Invalid FEN in chessboard file: {e}")

    predictions = model.predict(X)
    pred_classes = np.argmax(predictions, axis=1)

    for cls_idx in pred_classes:
        if cls_idx < len(classes):
            print(classes[cls_idx])
        else:
            print(CLASSES_3[0])


def run_train(model: Sequential, chess_file: str, save_file: str):
    try:
        with open(chess_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        error_exit(f"Failed to read {chess_file}: {e}")

    if not lines:
        error_exit("Training file is empty.")

    target_classes = get_model_classes(model)
    class_to_index = {c: i for i, c in enumerate(target_classes)}

    fens = []
    labels = []

    for line_idx, line in enumerate(lines, 1):
        tokens = line.split()
        if len(tokens) < 7:
            error_exit(f"Line {line_idx} missing expected label.")

        fen_str = " ".join(tokens[:6])
        label_str = " ".join(tokens[6:])

        # Map 5-class labels to 3-class if model is 3-class
        if label_str not in class_to_index and len(target_classes) == 3 and label_str in MAPPING_5_TO_3:
            label_str = MAPPING_5_TO_3[label_str]

        if label_str not in class_to_index:
            error_exit(f"Line {line_idx} has unknown label: {label_str}")

        fens.append(fen_str)
        labels.append(class_to_index[label_str])

    try:
        X = encode_boards_parallel(fens, encoding=model_encoding(model))
    except Exception as e:
        error_exit(f"Invalid FEN in training file: {e}")

    y = np.array(labels, dtype=np.int32)

    if model.optimizer is None or model.loss is None:
        model.compile(optimizer="adam", loss="crossentropy")

    model.classes = target_classes
    model.fit(X, y, epochs=15, batch_size=32, verbose=False)

    try:
        model.save(save_file)
    except Exception as e:
        error_exit(f"Failed to save trained model to {save_file}: {e}")


def main():
    mode, load_file, chess_file, save_file = parse_arguments()

    try:
        model = Sequential.load(load_file)
    except Exception as e:
        error_exit(f"Failed to load neural network from {load_file}: {e}")

    if mode == "predict":
        run_predict(model, chess_file)
    elif mode == "train":
        run_train(model, chess_file, save_file)
    sys.exit(0)


if __name__ == "__main__":
    main()
