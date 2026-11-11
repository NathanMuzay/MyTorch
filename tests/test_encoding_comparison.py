"""Check the comparison command keeps identical partitions and saves both runs."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from src.data.labels import CLASSES_5
from src.experiments.compare_encodings import main
from src.neural_network.model import Sequential
from src.chess_engine.encoders import model_encoding


class EncodingComparisonTests(unittest.TestCase):
    def test_paired_runs_share_partitions_and_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "positions.txt"
            dataset.write_text("".join(f"4k3/8/8/8/8/8/8/4K3 w - - 0 1 {label}\n"
                                       for label in CLASSES_5 * 4))
            output = root / "comparison"
            arguments = ["--dataset", str(dataset), "--output-dir", str(output),
                         "--max-per-class", "4", "--epochs", "1", "--init-seeds", "42"]
            with redirect_stdout(StringIO()):
                main(arguments)
            runs = json.loads((output / "results.json").read_text())
            self.assertEqual([run["encoding"] for run in runs], ["basic", "tactical"])
            self.assertEqual(runs[0]["training"]["partition"], runs[1]["training"]["partition"])
            self.assertEqual(runs[0]["training"]["augmentation"], runs[1]["training"]["augmentation"])
            for row in runs:
                loaded = Sequential.load(row["model"])
                self.assertEqual(model_encoding(loaded), row["encoding"])
                self.assertEqual(loaded.metadata["training"]["validation_samples"], 5)
            with self.assertRaises(FileExistsError):
                main(arguments)


if __name__ == "__main__":
    unittest.main()
