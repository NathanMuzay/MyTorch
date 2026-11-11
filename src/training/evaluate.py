"""Evaluate a saved Sequential network on every labeled position in a file."""

import numpy as np

from src.chess_engine.encoders import encode_board, model_encoding
from src.data.labels import get_model_classes, MAPPING_5_TO_3
from src.neural_network.model import Sequential


def classification_metrics(confusion, classes):
    """Summarize a confusion matrix (rows: expected, columns: predicted)."""
    confusion = np.asarray(confusion, dtype=np.int64)
    if not classes or confusion.shape != (len(classes), len(classes)) or np.any(confusion < 0):
        raise ValueError("Invalid confusion matrix for the model classes")
    true_positives = confusion.diagonal().astype(float)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    precision = np.divide(true_positives, predicted, out=np.zeros_like(true_positives), where=predicted > 0)
    recall = np.divide(true_positives, support, out=np.zeros_like(true_positives), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(true_positives), where=precision + recall > 0)
    return {"classes": list(classes), "confusion_matrix": confusion.tolist(),
            "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
            "support": support.tolist(), "macro_f1": float(f1.mean())}


def print_classification_metrics(metrics):
    print("Class                 Precision   Recall       F1   Support")
    for index, label in enumerate(metrics["classes"]):
        print(f"{label:<21} {metrics['precision'][index]:>8.2%} "
              f"{metrics['recall'][index]:>8.2%} {metrics['f1'][index]:>8.2%} "
              f"{metrics['support'][index]:>9}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("Confusion matrix (rows=expected, columns=predicted):")
    print("Columns: " + ", ".join(metrics["classes"]))
    for label, row in zip(metrics["classes"], metrics["confusion_matrix"]):
        print(f"{label:<21} " + " ".join(f"{value:>6}" for value in row))


def evaluate_model(model_path, dataset, batch_size=32):
    """Return loss and accuracy without training or saving the model."""
    if batch_size <= 0:
        raise ValueError("batch_size must be strictly positive")
    model = Sequential.load(model_path)
    encoding = model_encoding(model)
    classes = get_model_classes(model)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    class_to_index = {label: index for index, label in enumerate(classes)}
    inputs, targets = [], []
    total_loss = total_accuracy = 0.0
    count = 0

    def evaluate_batch():
        expected = np.asarray(targets, dtype=np.int32)
        loss, accuracy, predictions = model.evaluate(np.asarray(inputs, dtype=np.float32), expected,
                                                     return_predictions=True)
        np.add.at(confusion, (expected, np.argmax(predictions, axis=1)), 1)
        return loss, accuracy

    with open(dataset, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 7:
                raise ValueError(f"Line {line_number} missing expected label")
            label = " ".join(parts[6:])
            if len(classes) == 3:
                label = MAPPING_5_TO_3.get(label, label)
            if label not in class_to_index:
                raise ValueError(f"Line {line_number} has unknown label: {label}")
            try:
                inputs.append(encode_board(" ".join(parts[:6]), encoding=encoding))
            except ValueError as error:
                raise ValueError(f"Line {line_number}: {error}") from error
            targets.append(class_to_index[label])
            if len(inputs) == batch_size:
                loss, accuracy = evaluate_batch()
                total_loss += loss * len(inputs)
                total_accuracy += accuracy * len(inputs)
                count += len(inputs)
                inputs.clear()
                targets.clear()
    if inputs:
        loss, accuracy = evaluate_batch()
        total_loss += loss * len(inputs)
        total_accuracy += accuracy * len(inputs)
        count += len(inputs)
    if count == 0:
        raise ValueError("Evaluation file is empty")
    loss, accuracy = total_loss / count, total_accuracy / count
    print(f"Evaluation: {count} positions - Accuracy: {accuracy * 100:.2f}% (Loss: {loss:.4f})")
    print_classification_metrics(classification_metrics(confusion, classes))
    return loss, accuracy
