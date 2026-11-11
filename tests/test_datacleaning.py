"""Vérifier les pertes de données et les décisions de nettoyage sur petits corpus."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.data.cleaning import clean_dataset, parse_line


FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


class ParserTests(unittest.TestCase):
    def test_normalization_and_counters_do_not_change_key(self):
        sample = parse_line("  4k3/44/8/8/8/8/8/4K3\tw qK - 000 02 Nothing\r\n")
        self.assertEqual(sample.fen, "4k3/8/8/8/8/8/8/4K3 w Kq - 0 2")
        self.assertEqual(parse_line(FEN + " Nothing").position_key,
                         parse_line(FEN.replace("0 1", "7 90") + " Nothing").position_key)

    def test_invalid_structure_labels_and_metadata(self):
        invalid = [FEN.replace("4k3", "8"), FEN.replace("4k3", "9k"),
                   FEN.replace("4k3", "3pk3"), FEN.replace(" w ", " x "),
                   FEN.replace("- -", "KK -"), FEN.replace("- -", "- a3"),
                   FEN.replace("0 1", "-1 1"), FEN.replace("0 1", "0 0")]
        for fen in invalid:
            with self.subTest(fen=fen), self.assertRaises(ValueError):
                parse_line(fen + " Nothing")
        for label in ("Check", "nothing", "Check White", "Nothing extra"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_line(FEN + " " + label)

    def test_all_supported_annotations(self):
        for label in ("Nothing", "Check Black", "Checkmate Black"):
            self.assertEqual(parse_line(FEN + " " + label).label, label)
        for label in ("Check White", "Checkmate White"):
            self.assertEqual(parse_line(FEN.replace(" w ", " b ") + " " + label).label, label)


class PipelineTests(unittest.TestCase):
    def test_duplicates_conflicts_rejections_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for category in ("check", "nothing", "for_train"):
                (root / category).mkdir()
            # Cette clé apparaîtra d'abord dans check/, puis sera contredite.
            original = (FEN + " Check Black\n\ninvalid\n").encode() + b"\xff\n"
            (root / "check/a.txt").write_bytes(original)
            other = FEN.replace(" w ", " b ")
            (root / "nothing/b.txt").write_text(
                FEN + " Nothing\n" + other + " Nothing\n"
                + other.replace("0 1", "9 12") + " Nothing\n", encoding="utf-8")
            (root / "for_train/train.txt").write_text("invalid\n", encoding="utf-8")
            report = clean_dataset(root, root / "cleaned")
            self.assertEqual(report["totals"], dict(lines_read=7, blank_lines=1,
                             invalid_lines=2, valid_lines=4, duplicate_lines=1, conflicting_lines=1))
            self.assertEqual(report["unique_positions"], 2)
            self.assertEqual(report["quarantined_positions"], 1)
            self.assertEqual(report["kept_positions"], 1)
            self.assertEqual(report["for_train"]["train.txt"]["totals"]["invalid_lines"], 1)
            self.assertEqual((root / "cleaned/train.txt").read_text(), "")
            output = root / "cleaned"
            self.assertEqual((output / "cleaned.txt").read_text(), other + " Nothing\n")
            self.assertEqual(sorted(path.name for path in output.iterdir()),
                             ["cleaned.txt", "train.txt"])
            self.assertEqual((root / "check/a.txt").read_bytes(), original)
            with self.assertRaises(FileExistsError):
                clean_dataset(root, output)
            second = root / "second"
            self.assertEqual(clean_dataset(root, second), report)
            self.assertEqual((second / "cleaned.txt").read_bytes(), (output / "cleaned.txt").read_bytes())

    def test_training_files_keep_names_and_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nothing").mkdir()
            (root / "nothing/a.txt").write_text(FEN + " Nothing\n", encoding="utf-8")
            (root / "for_train").mkdir()
            for name in ("train.txt", "test.txt", "validation.txt", "extra.txt"):
                (root / "for_train" / name).write_text(
                    FEN + " Nothing\n" + FEN + " Nothing\ninvalid\n", encoding="utf-8")
            report = clean_dataset(root, root / "cleaned")
            for name, summary in report["for_train"].items():
                self.assertEqual((root / "cleaned" / name).read_text(), FEN + " Nothing\n")
                self.assertEqual(summary["totals"]["duplicate_lines"], 1)
                self.assertEqual(summary["totals"]["invalid_lines"], 1)
            self.assertEqual(len(report["for_train"]), 4)
            self.assertEqual(sorted(path.name for path in (root / "cleaned").iterdir()),
                             ["cleaned.txt", "extra.txt", "test.txt", "train.txt", "validation.txt"])

    def test_no_input_and_unsafe_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                clean_dataset(root, root / "cleaned")
            self.assertFalse((root / "cleaned").exists())
            (root / "nothing").mkdir()
            (root / "nothing/a.txt").write_text(FEN + " Nothing\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                clean_dataset(root, root / "nothing/cleaned")


class CommandLineTests(unittest.TestCase):
    def test_clean_flag_for_script_and_module(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nothing").mkdir()
            (root / "nothing/a.txt").write_text(FEN + " Nothing\n", encoding="utf-8")
            for index, command in enumerate(([sys.executable, "src/my_torch.py"],
                                              [sys.executable, "-m", "src.my_torch"])):
                with self.subTest(command=command):
                    output = root / f"cleaned-{index}"
                    result = subprocess.run(command + ["-clean", "--input", str(root),
                                            "--output", str(output)], cwd=project,
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual((output / "cleaned.txt").read_text(), FEN + " Nothing\n")
                    # Le point d'entrée propage l'erreur sans écraser la sortie.
                    result = subprocess.run(command + ["-clean", "--input", str(root),
                                            "--output", str(output)], cwd=project,
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 84)

    def test_no_clean_without_flag(self):
        project = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(project / "src/my_torch.py")]
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(command, cwd=temporary, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Sans option : affiche cette aide", result.stdout)
            self.assertEqual(list(Path(temporary).iterdir()), [])
            result = subprocess.run(command + ["--output", "cleaned"], cwd=temporary,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse((Path(temporary) / "cleaned").exists())


if __name__ == "__main__":
    unittest.main()
