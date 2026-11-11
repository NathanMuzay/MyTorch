"""Compare basic and tactical features with fixed data and training settings.

Run from the project root with python3 -m src.experiments.compare_encodings.
Results and models go into a new directory; existing runs are never overwritten.
"""

import argparse
import json
from pathlib import Path
import time

from src.my_torch import parse_arguments
from src.training.train import train_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/cleaned/train.txt")
    parser.add_argument("--validation-dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Fixed sampling and split seed")
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args(argv)
    if len(args.init_seeds) != len(set(args.init_seeds)):
        parser.error("--init-seeds must be distinct")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    reference_partition = None
    reference_augmentation = None
    for init_seed in args.init_seeds:
        for encoding in ("basic", "tactical"):
            model_path = args.output_dir / f"{encoding}-{init_seed}.nn"
            command = ["-train", "--dataset", args.dataset, "--output", str(model_path),
                       "--encoding", encoding, "--classes", "5", "--seed", str(args.seed),
                       "--init-seed", str(init_seed), "--max-per-class", str(args.max_per_class),
                       "--epochs", str(args.epochs)]
            if args.validation_dataset:
                command += ["--validation-dataset", args.validation_dataset]
            started = time.perf_counter()
            model = train_model(parse_arguments(command))
            training = model.metadata["training"]
            partition = training["partition"]
            augmentation = training["augmentation"]
            if reference_partition is None:
                reference_partition, reference_augmentation = partition, augmentation
            if partition != reference_partition or augmentation != reference_augmentation:
                raise RuntimeError("Encoding comparison changed positions or augmentation membership")
            results.append({"encoding": encoding, "init_seed": init_seed,
                            "seconds": time.perf_counter() - started,
                            "model": str(model_path), "training": training})
            # Keep completed runs reviewable even if a later run is interrupted.
            (args.output_dir / "results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
            del model
    for encoding in ("basic", "tactical"):
        runs = [row["training"]["validation_accuracy"] for row in results if row["encoding"] == encoding]
        print(f"{encoding}: mean accuracy {sum(runs) / len(runs):.2%} over {len(runs)} initializations")


if __name__ == "__main__":
    main()
