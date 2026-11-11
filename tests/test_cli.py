import unittest
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path

from src.neural_network.model import Sequential


class TestCLI(unittest.TestCase):
    def test_analyzer_help(self):
        result = subprocess.run(
            [sys.executable, "my_torch_analyzer", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("USAGE", result.stdout)

    def test_generator_help(self):
        result = subprocess.run(
            [sys.executable, "my_torch_generator", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("USAGE", result.stdout)

    def test_analyzer_error_no_args(self):
        result = subprocess.run(
            [sys.executable, "my_torch_analyzer"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 84)
        self.assertIn("Error:", result.stderr)

    def test_analyzer_error_missing_file(self):
        result = subprocess.run(
            [sys.executable, "my_torch_analyzer", "--predict", "missing.nn", "missing.txt"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 84)
        self.assertIn("Error:", result.stderr)

    def test_analyzer_error_unknown_option(self):
        result = subprocess.run(
            [sys.executable, "my_torch_analyzer", "--unknown-flag", "model.nn", "fens.txt"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 84)
        self.assertIn("Error:", result.stderr)

    def test_analyzer_predict_valid(self):
        result = subprocess.run(
            [
                sys.executable,
                "my_torch_analyzer",
                "--predict",
                "my_torch_network_basic.nn",
                "dataset/examples/sample_chessboards.txt",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(len(lines), 5)
        for line in lines:
            self.assertIn(line, ["Nothing", "Check", "Checkmate"])

    def test_analyzer_train_and_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temp_in, temp_out = root / "input.nn", root / "trained.nn"
            shutil.copyfile("my_torch_network_basic.nn", temp_in)
            result = subprocess.run(
                [
                    sys.executable,
                    "my_torch_analyzer",
                    "--train",
                    "--save",
                    temp_out,
                    temp_in,
                    "dataset/examples/train_chessboards.txt",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertTrue(temp_out.is_file())

    def test_analyzer_train_error_no_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_in = Path(directory) / "input.nn"
            shutil.copyfile("my_torch_network_basic.nn", temp_in)
            result = subprocess.run(
                [
                    sys.executable,
                    "my_torch_analyzer",
                    "--train",
                    temp_in,
                    "dataset/examples/sample_chessboards.txt",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 84)
            self.assertIn("Error:", result.stderr)

    def test_generator_valid_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "basic_network.conf"
            shutil.copyfile("basic_network.conf", config)
            result = subprocess.run(
                [sys.executable, "my_torch_generator", config, "2"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "basic_network_1.nn").is_file())
            self.assertTrue((root / "basic_network_2.nn").is_file())

    def test_generator_error_odd_arguments(self):
        result = subprocess.run(
            [sys.executable, "my_torch_generator", "basic_network.conf"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 84)
        self.assertIn("Error:", result.stderr)

    def test_generator_with_batchnorm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conf_file = root / "batchnorm.conf"
            conf_file.write_text("Dense in=773 out=32\nBatchNorm dim=32\nReLU\nDense in=32 out=3\nSoftmax\n")
            result = subprocess.run(
                [sys.executable, "my_torch_generator", conf_file, "1"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0)
            output = root / "batchnorm_1.nn"
            self.assertTrue(output.is_file())
            model = Sequential.load(output)
            self.assertEqual(len(model.layers), 5)


if __name__ == "__main__":
    unittest.main()

