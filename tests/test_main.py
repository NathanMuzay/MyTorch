"""Exercise training and evaluation through the unified entry point."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.chess_engine.encoders import encode_board
from src.neural_network.model import Sequential
from src.training.evaluate import evaluate_model
from src.data.labels import CLASSES_3, CLASSES_5, MAPPING_5_TO_3


PROJECT = Path(__file__).resolve().parents[1]
FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


class MainTests(unittest.TestCase):
    def run_cli(self, *args, module=False):
        entry = ["-m", "src.my_torch"] if module else ["src/my_torch.py"]
        return subprocess.run([sys.executable, *entry, *map(str, args)], cwd=PROJECT,
                              capture_output=True, text=True)

    def test_train_save_reload_and_evaluate_for_both_entry_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "positions.txt"
            labels = CLASSES_5 * 2 + ["Nothing"]
            dataset.write_text("".join(f"{FEN} {label}\n" for label in labels), encoding="utf-8")
            for module, classes, encoding in ((False, CLASSES_3, "basic"), (True, CLASSES_5, "basic"),
                                               (False, CLASSES_5, "tactical")):
                with self.subTest(module=module, encoding=encoding):
                    model_path = root / f"model-{len(classes)}-{encoding}.nn"
                    trained = self.run_cli("-train", "--dataset", dataset, "--output", model_path,
                                           "--classes", len(classes), "--epochs", 1,
                                           "--batch-size", 4, "--encoding", encoding, module=module)
                    self.assertEqual(trained.returncode, 0, trained.stderr)
                    self.assertIn("Final Validation Accuracy", trained.stdout)
                    original_bytes = model_path.read_bytes()
                    model = Sequential.load(model_path)
                    self.assertEqual(model.classes, classes)
                    self.assertEqual(model.metadata["encoding"], encoding)
                    indices = [classes.index(MAPPING_5_TO_3[label] if len(classes) == 3 else label)
                               for label in labels]
                    expected = model.evaluate(np.asarray([encode_board(FEN, encoding=encoding)] * len(labels)),
                                              np.asarray(indices))
                    actual = evaluate_model(model_path, dataset, batch_size=4)
                    np.testing.assert_allclose(actual, expected, rtol=1e-6)
                    evaluated = self.run_cli("--eval", "--model", model_path,
                                             "--dataset", dataset, "--batch-size", 4,
                                             module=module)
                    self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
                    self.assertIn("Evaluation: 11 positions", evaluated.stdout)
                    self.assertEqual(model_path.read_bytes(), original_bytes)
                    predicted = subprocess.run([sys.executable, "-m", "src.cli.analyzer", "--predict",
                                                str(model_path), str(dataset)], cwd=PROJECT,
                                               capture_output=True, text=True)
                    self.assertEqual(predicted.returncode, 0, predicted.stderr)
                    self.assertEqual(len(predicted.stdout.splitlines()), len(labels))
                    continued_path = root / f"continued-{len(classes)}-{encoding}.nn"
                    continued = subprocess.run([sys.executable, "-m", "src.cli.analyzer", "--train", "--save",
                                                str(continued_path), str(model_path), str(dataset)], cwd=PROJECT,
                                               capture_output=True, text=True)
                    self.assertEqual(continued.returncode, 0, continued.stderr)
                    self.assertEqual(Sequential.load(continued_path).metadata["encoding"], encoding)
                    self.assertEqual(model_path.read_bytes(), original_bytes)

    def test_invalid_mode_combinations(self):
        for args in (("-train", "-eval"), ("-eval",),
                     ("-clean", "--epochs", "1"), ("--dataset", "data.txt"),
                     ("-train", "--batch-size", "0"),
                     ("-clean", "--no-bn"), ("--loss", "focal"),
                     ("-clean", "--seed", "42"), ("-train", "--patience", "0"),
                     ("-clean", "--label-smoothing", "0.05"), ("-clean", "--augment-colors"),
                     ("-eval", "--model", "model.nn", "--encoding", "tactical"),
                     ("-clean", "--init-seed", "43"), ("-eval", "--model", "model.nn", "--validation-dataset", "v.txt"),
                     ("-train", "--augment-colors", "--no-augment-colors")):
            with self.subTest(args=args):
                self.assertEqual(self.run_cli(*args).returncode, 2)

    def test_runtime_errors_use_code_84(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "positions.txt"
            output = root / "model.nn"
            for content in ("", f"{FEN} Nothing\n"):
                dataset.write_text(content, encoding="utf-8")
                result = self.run_cli("-train", "--dataset", dataset, "--output", output)
                self.assertEqual(result.returncode, 84, result.stderr)
                self.assertFalse(output.exists())
            result = self.run_cli("-train", "--dataset", root / "missing.txt", "--output", output)
            self.assertEqual(result.returncode, 84)
            for content in ("", FEN, f"{FEN} Unknown", "invalid FEN w - - 0 1 Nothing"):
                dataset.write_text(content, encoding="utf-8")
                result = self.run_cli("-eval", "--model", "my_torch_network_basic.nn",
                                      "--dataset", dataset)
                self.assertEqual(result.returncode, 84, result.stderr)
                self.assertIn("my_torch :", result.stderr)
            result = self.run_cli("-eval", "--model", root / "missing.nn", "--dataset", dataset)
            self.assertEqual(result.returncode, 84)


if __name__ == "__main__":
    unittest.main()
