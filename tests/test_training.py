"""Numerical gradients, checkpoint selection and reproducible data sampling."""

from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.neural_network.activations import Softmax
from src.neural_network.layers import BatchNorm1d, Dense
from src.neural_network.losses import CategoricalCrossEntropy, FocalLoss
from src.neural_network.model import Sequential
from src.neural_network.optimizers import SGD
from src.training.train import load_dataset_balanced, stratified_split, train_model, board_keys
from src.training.evaluate import classification_metrics
from src.my_torch import parse_arguments


def numerical_gradient(function, values, step=1e-5):
    gradient = np.empty_like(values)
    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus) - function(minus)) / (2 * step)
    return gradient


class TrainingTests(unittest.TestCase):
    def test_fixed_validation_is_shared_across_limits_encodings_and_initializations(self):
        fens = [f"{'1' * column}k{'1' * (7 - column)}/8/8/8/8/8/8/4K3 w - - 0 1"
                for column in range(8)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, validation = root / "train.txt", root / "validation.txt"
            dataset.write_text("".join(f"{fen} Nothing\n" for fen in fens))
            validation.write_text(f"{fens[0]} Nothing\n{fens[1]} Nothing\n")
            reference = None
            original_fit = Sequential.fit
            for encoding, limit, init_seed in (("basic", 100, 42), ("tactical", 100, 43), ("basic", 3, 43)):
                captured = {}

                def fit(model, inputs, targets, **kwargs):
                    captured["train_keys"] = {row.tobytes() for row in board_keys(inputs)}
                    captured["val_keys"] = {row.tobytes() for row in board_keys(kwargs["validation_data"][0])}
                    return original_fit(model, inputs, targets, **kwargs)

                arguments = ["-train", "--dataset", str(dataset), "--validation-dataset", str(validation),
                             "--output", str(root / "model.nn"), "--epochs", "1", "--encoding", encoding,
                             "--init-seed", str(init_seed), "--max-per-class", str(limit), "--no-augment-colors"]
                with patch.object(Sequential, "fit", autospec=True, side_effect=fit), redirect_stdout(StringIO()):
                    model = train_model(parse_arguments(arguments))
                metadata = model.metadata["training"]
                self.assertFalse(captured["train_keys"] & captured["val_keys"])
                self.assertEqual(metadata["validation_samples"], 2)
                self.assertEqual(metadata["seed"], 42)
                self.assertEqual(metadata["init_seed"], init_seed)
                self.assertEqual(metadata["partition"]["method"], "external")
                fingerprint = metadata["partition"]["validation_fingerprint"]
                if reference is None:
                    reference = fingerprint
                self.assertEqual(fingerprint, reference)
                if limit == 100:
                    self.assertEqual(metadata["train_samples"], 6)
                    self.assertEqual(metadata["partition"]["excluded_training_overlaps"], 2)
            arguments = ["-train", "--dataset", str(validation), "--validation-dataset", str(validation),
                         "--output", str(root / "unused.nn")]
            with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "No training positions"):
                train_model(parse_arguments(arguments))
            validation.write_text("")
            arguments[arguments.index("--dataset") + 1] = str(dataset)
            with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "Validation dataset is empty"):
                train_model(parse_arguments(arguments))
            for content, message in ((f"{fens[0]} Unknown\n", "unknown label"),
                                     (f"{fens[0]}\n", "missing expected label")):
                validation.write_text(content)
                with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, message):
                    train_model(parse_arguments(arguments))

    def test_augmented_training_records_counts_and_keeps_validation_unmodified(self):
        classes = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "positions.txt"
            dataset.write_text("".join(f"{fen} {label}\n" for label in classes * 4))
            args = parse_arguments(["-train", "--dataset", str(dataset), "--output", str(root / "model.nn"),
                                    "--classes", "5", "--epochs", "1", "--augment-colors"])
            with redirect_stdout(StringIO()):
                model = train_model(args)
            training = Sequential.load(root / "model.nn").metadata["training"]
            self.assertEqual(training["train_samples_before_augmentation"], 15)
            self.assertEqual(training["train_samples"], 30)
            self.assertEqual(training["validation_samples"], 5)
            self.assertEqual(training["augmentation"], {"enabled": True, "original_samples": 15,
                                                      "mirrored_samples": 15, "excluded_validation_matches": 0})
            self.assertEqual(sum(training["validation_metrics"]["support"]), 5)
            self.assertEqual(model.layers[0].in_features, 773)
            disabled = parse_arguments(["-train", "--no-augment-colors"])
            self.assertFalse(disabled.augment_colors)

    def test_smoothing_changes_training_targets_only_and_is_saved(self):
        classes = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "positions.txt"
            dataset.write_text("".join(f"{fen} {label}\n" for label in classes * 4))
            for loss, configured, expected in (("cce", None, 0), ("cce", "0.05", 0.05),
                                               ("cce", "0", 0), ("focal", None, 0)):
                with self.subTest(loss=loss, configured=configured):
                    arguments = ["-train", "--dataset", str(dataset), "--output", str(root / "model.nn"),
                                 "--classes", "5", "--epochs", "1", "--loss", loss]
                    if configured is not None:
                        arguments += ["--label-smoothing", configured]
                    captured = {}
                    original_fit = Sequential.fit

                    def fit(model, inputs, targets, **kwargs):
                        captured["train"] = targets.copy()
                        captured["validation"] = kwargs["validation_data"][1].copy()
                        return original_fit(model, inputs, targets, **kwargs)

                    with patch.object(Sequential, "fit", autospec=True, side_effect=fit), redirect_stdout(StringIO()):
                        model = train_model(parse_arguments(arguments))
                    np.testing.assert_allclose(captured["train"].sum(axis=1), 1, atol=1e-7)
                    np.testing.assert_allclose(captured["train"].max(axis=1), 1 - expected + expected / 5)
                    np.testing.assert_allclose(captured["train"].min(axis=1), expected / 5)
                    self.assertTrue(np.isin(captured["validation"], [0, 1]).all())
                    np.testing.assert_array_equal(captured["validation"].sum(axis=1), 1)
                    self.assertEqual(model.metadata["training"]["label_smoothing"], expected)
                    self.assertEqual(model.metadata["training"]["checkpoint_tie_breaker"], "val_loss")
                    loaded = Sequential.load(root / "model.nn")
                    self.assertEqual(loaded.metadata["training"]["label_smoothing"], expected)

    def test_smoothing_rejects_invalid_values_and_unsupported_loss(self):
        for loss, value in (("cce", "-0.1"), ("cce", "1"), ("cce", "nan"),
                            ("cce", "inf"), ("focal", "0.05")):
            with self.subTest(loss=loss, value=value):
                args = parse_arguments(["-train", "--dataset", "missing.txt", "--loss", loss,
                                        "--label-smoothing", value])
                with self.assertRaisesRegex(ValueError, "label_smoothing"):
                    train_model(args)

    def test_accuracy_ties_choose_lower_loss_without_extending_patience(self):
        inputs = np.array([[1., 2.], [2., 1.]])
        targets = np.array([0, 1])
        model = Sequential([Dense(2, 2), Softmax()])
        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0))
        states = []

        def validate(*_):
            states.append(model.layers[0].weights.copy())
            return ([0.8, 0.6, 0.9][len(states) - 1], 0.7)

        with patch.object(model, "evaluate", side_effect=validate):
            history = model.fit(inputs, targets, epochs=8, batch_size=2, shuffle=False, verbose=False,
                                validation_data=(inputs, targets), restore_best_weights=True,
                                monitor="val_accuracy", early_stopping_patience=2)
        self.assertEqual(len(history["loss"]), 3)
        self.assertEqual(model.best_epoch, 2)
        np.testing.assert_array_equal(model.layers[0].weights, states[1])

    def test_default_training_targets_accuracy_without_class_weight_bias(self):
        args = parse_arguments(["-train"])
        self.assertTrue(args.augment_colors)
        self.assertEqual(args.monitor, "val_accuracy")
        self.assertEqual(args.patience, 6)
        self.assertEqual(args.loss, "cce")
        self.assertFalse(args.boost_mate)
        # The previous experiment remains available with explicit arguments.
        previous = parse_arguments(["-train", "--loss", "focal", "--boost-mate",
                                    "--monitor", "val_loss", "--patience", "4"])
        self.assertEqual(previous.loss, "focal")
        self.assertTrue(previous.boost_mate)
        self.assertEqual(previous.monitor, "val_loss")
        self.assertEqual(previous.patience, 4)

    def test_accuracy_monitor_keeps_later_improvement_despite_increasing_loss(self):
        inputs = np.array([[1., 2.], [2., 1.]])
        targets = np.array([0, 1])
        for monitor, expected_epoch in (("val_accuracy", 3), ("val_loss", 2)):
            with self.subTest(monitor=monitor):
                model = Sequential([Dense(2, 2), Softmax()])
                model.compile(optimizer=SGD(learning_rate=0.01, momentum=0))
                states = []

                def validate(*_):
                    states.append(model.layers[0].weights.copy())
                    index = len(states) - 1
                    return ([0.39, 0.38, 0.44][index], [0.62, 0.65, 0.69][index])

                with patch.object(model, "evaluate", side_effect=validate):
                    model.fit(inputs, targets, epochs=3, batch_size=2, shuffle=False, verbose=False,
                              validation_data=(inputs, targets), restore_best_weights=True, monitor=monitor)
                self.assertEqual(model.best_epoch, expected_epoch)
                np.testing.assert_array_equal(model.layers[0].weights, states[expected_epoch - 1])

    def test_classification_metrics_handle_missing_and_unpredicted_classes(self):
        metrics = classification_metrics([[2, 1, 0], [1, 1, 0], [0, 0, 0]],
                                         ["Nothing", "Check", "Checkmate"])
        self.assertEqual(metrics["support"], [3, 2, 0])
        np.testing.assert_allclose(metrics["precision"], [2 / 3, 1 / 2, 0])
        np.testing.assert_allclose(metrics["recall"], [2 / 3, 1 / 2, 0])
        self.assertAlmostEqual(metrics["macro_f1"], 7 / 18)

    def test_probability_and_fused_gradients_match_finite_differences(self):
        logits = np.array([[0.7, -0.3, 1.2], [1.0, 0.2, -0.4]])
        weights = np.array([1.0, 1.2, 2.0])
        for targets in (np.array([2, 0]), np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])):
            for loss in (CategoricalCrossEntropy(), CategoricalCrossEntropy(class_weights=weights),
                         FocalLoss(gamma=0), FocalLoss(gamma=2), FocalLoss(gamma=2, alpha=weights)):
                with self.subTest(loss=type(loss).__name__, targets=targets.ndim,
                                  gamma=getattr(loss, "gamma", None)):
                    probabilities = Softmax().forward(logits)
                    loss(probabilities, targets)
                    actual_probabilities = loss.backward().copy()
                    actual_logits = loss.backward_softmax().copy()
                    expected_probabilities = numerical_gradient(lambda p: loss(p, targets), probabilities)
                    expected_logits = numerical_gradient(lambda z: loss(Softmax().forward(z), targets), logits)
                    np.testing.assert_allclose(actual_probabilities, expected_probabilities, rtol=1e-6, atol=1e-8)
                    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-6, atol=1e-8)

    def test_fit_updates_weights_with_the_correct_end_to_end_gradient(self):
        inputs = np.array([[0.2, 0.8], [1.0, -0.5]])
        targets = np.array([2, 0])
        for loss in (CategoricalCrossEntropy(class_weights=np.array([1., 1.2, 2.])),
                     FocalLoss(alpha=np.array([1., 1.2, 2.]))):
            model = Sequential([Dense(2, 3), Softmax()])
            layer = model.layers[0]
            original_weights = layer.weights.copy()

            def objective(weights):
                layer.weights[...] = weights
                return loss(model.predict(inputs), targets)

            gradient = numerical_gradient(objective, original_weights)
            layer.weights[...] = original_weights
            model.compile(optimizer=SGD(learning_rate=0.01, momentum=0), loss=loss)
            model.fit(inputs, targets, epochs=1, batch_size=2, shuffle=False, verbose=False)
            np.testing.assert_allclose(layer.weights, original_weights - 0.01 * gradient, atol=1e-9)

    def test_best_weights_and_batchnorm_statistics_are_restored(self):
        inputs = np.array([[1., 2.], [2., -1.], [-1., 1.], [3., 4.]])
        targets = np.array([0, 1, 0, 1])
        for epochs, monitor in ((3, "val_loss"), (8, "val_loss"), (8, "val_accuracy")):
            with self.subTest(epochs=epochs, monitor=monitor):
                model = Sequential([Dense(2, 3), BatchNorm1d(3), Dense(3, 2), Softmax()])
                model.compile(optimizer=SGD(learning_rate=0.01, momentum=0))
                states = []

                def validation(*_):
                    states.append([{name: value.copy() for name, value in vars(layer).items()
                                    if name in ("weights", "bias", "gamma", "beta", "running_mean", "running_var")}
                                   for layer in model.layers])
                    index = len(states) - 1
                    return ([1.0, 0.5, 0.8, 0.9][index], [0.5, 0.8, 0.6, 0.7][index])

                with patch.object(model, "evaluate", side_effect=validation):
                    history = model.fit(inputs, targets, epochs=epochs, batch_size=2,
                                        shuffle=False, verbose=False, validation_data=(inputs, targets),
                                        early_stopping_patience=2, restore_best_weights=True, monitor=monitor)
                self.assertEqual(len(history["loss"]), min(epochs, 4))
                self.assertEqual(model.best_epoch, 2)
                for layer, expected in zip(model.layers, states[1]):
                    for name, values in expected.items():
                        np.testing.assert_array_equal(getattr(layer, name), values)
                        if name in layer.params:
                            self.assertIs(getattr(layer, name), layer.params[name])

    def test_loss_configuration_survives_save_reload(self):
        inputs = np.array([[1., 2.], [2., -1.]])
        targets = np.array([0, 1])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.nn"
            for loss in (FocalLoss(gamma=1.5, alpha=np.array([1., 2.])),
                         CategoricalCrossEntropy(class_weights=np.array([1., 2.]))):
                model = Sequential([Dense(2, 2), Softmax()])
                model.compile(loss=loss)
                expected = model.evaluate(inputs, targets)
                model.save(path)
                loaded = Sequential.load(path)
                self.assertIsInstance(loaded.loss, type(loss))
                np.testing.assert_allclose(loaded.evaluate(inputs, targets), expected)

    def test_epoch_loss_weights_the_final_small_batch_by_sample_count(self):
        inputs = np.zeros((3, 2))
        targets = np.array([0, 0, 1])
        model = Sequential([Dense(2, 2), Softmax()])
        model.layers[0].weights[...] = 0
        model.layers[0].bias[...] = np.log([0.9, 0.1])
        model.compile(optimizer=SGD(learning_rate=0, momentum=0))
        expected_loss, _ = model.evaluate(inputs, targets)
        history = model.fit(inputs, targets, epochs=1, batch_size=2, shuffle=False, verbose=False)
        self.assertAlmostEqual(history["loss"][0], expected_loss)

    def test_stratified_split_is_disjoint_complete_and_reproducible(self):
        labels = np.repeat(np.arange(5), 20)
        train, validation = stratified_split(labels, seed=42)
        self.assertEqual(len(train), 85)
        self.assertEqual(len(validation), 15)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(range(100)))
        np.testing.assert_array_equal(np.bincount(labels[validation]), [3] * 5)
        again = stratified_split(labels, seed=42)
        np.testing.assert_array_equal(train, again[0])
        np.testing.assert_array_equal(validation, again[1])
        # A singleton remains in training; a dataset with only singletons has no validation.
        train, validation = stratified_split(np.array([0, 1, 1]))
        self.assertIn(0, train)
        with self.assertRaises(ValueError):
            stratified_split(np.array([0, 1]))

    def test_reservoir_samples_beyond_the_first_positions(self):
        fen = "4k3/8/8/8/8/8/8/4K3 w - -"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "positions.txt"
            path.write_text("".join(f"{fen} {i} 1 Nothing\n" for i in range(100)))
            with patch("src.training.train.encode_boards_parallel", side_effect=lambda fens: np.array(fens)):
                first, labels = load_dataset_balanced(path, ["Nothing"], max_per_class=5, seed=42)
                second, _ = load_dataset_balanced(path, ["Nothing"], max_per_class=5, seed=42)
            self.assertEqual(len(first), 5)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(labels, [0] * 5)
            self.assertTrue(any(int(fen.split()[4]) >= 5 for fen in first))


if __name__ == "__main__":
    unittest.main()
