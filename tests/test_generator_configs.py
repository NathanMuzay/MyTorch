"""Exercise shipped presets, encoding metadata and generic legacy configs."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.chess_engine.encoders import encode_board, model_encoding
from src.cli.generator import build_model_from_config
from src.data.labels import get_model_classes
from src.neural_network.model import Sequential


PROJECT = Path(__file__).resolve().parents[1]
FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


class GeneratorConfigTests(unittest.TestCase):
    def test_presets_generate_reload_and_predict_with_their_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = []
            for name in ("basic", "medium", "full", "tactical"):
                configuration = root / f"{name}_network.conf"
                shutil.copyfile(PROJECT / configuration.name, configuration)
                arguments.extend((str(configuration), "1"))
            generated = subprocess.run([sys.executable, "-m", "src.cli.generator", *arguments],
                                       cwd=PROJECT, capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            positions = root / "positions.txt"
            positions.write_text(FEN + "\n")
            for name, encoding, outputs in (("basic", "basic", 3), ("medium", "basic", 3),
                                             ("full", "basic", 5), ("tactical", "tactical", 5)):
                with self.subTest(name=name):
                    path = root / f"{name}_network_1.nn"
                    model = Sequential.load(path)
                    self.assertEqual(model_encoding(model), encoding)
                    self.assertEqual(model.output_dim, outputs)
                    predictions = model.predict(encode_board(FEN, encoding=encoding)[None])
                    self.assertEqual(predictions.shape, (1, outputs))
                    self.assertTrue(np.isfinite(predictions).all())
                    np.testing.assert_allclose(predictions.sum(axis=1), 1)
                    predicted = subprocess.run([sys.executable, "-m", "src.cli.analyzer", "--predict",
                                                str(path), str(positions)], cwd=PROJECT,
                                               capture_output=True, text=True)
                    self.assertEqual(predicted.returncode, 0, predicted.stderr)
                    self.assertIn(predicted.stdout.strip(), get_model_classes(model))
                    if name in ("medium", "full"):
                        reference = json.loads((PROJECT / f"my_torch_network_{name}.nn").read_text())
                        self.assertEqual(json.loads(path.read_text())["layers"], reference["layers"])

    def test_invalid_encoding_directives_and_incompatible_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "invalid.conf"
            for directive in ("encoding unknown", "encoding", "encoding tactical extra",
                              "encoding basic\nencoding basic", "encoding tactical"):
                with self.subTest(directive=directive):
                    config.write_text(directive + "\ndense in=773 out=5\nsoftmax\n")
                    with self.assertRaises(ValueError):
                        build_model_from_config(config)

    def test_generic_configs_without_encoding_keep_working(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "generic.conf"
            config.write_text("dense in=2 out=2\nsoftmax\n")
            model = build_model_from_config(config)
            self.assertNotIn("encoding", model.metadata)
            self.assertEqual(model.predict(np.zeros((1, 2))).shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
