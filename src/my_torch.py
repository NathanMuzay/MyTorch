"""Point d'entrée : nettoyer les données, entraîner ou évaluer le réseau."""

import argparse
from pathlib import Path
import sys

# Permettre aussi le lancement direct : python3 src/my_torch.py.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import clean_dataset
from src.training.train import add_training_arguments, train_model
from src.training.evaluate import evaluate_model


def helper():
    """Construire les options et l'aide affichée avec -h ou --help."""
    parser = argparse.ArgumentParser(
        description="My Torch : réseau de neurones et nettoyage des données.",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sans option : affiche cette aide.\n\n"
            "Exemples :\n"
            "  python3 src/my_torch.py -train --dataset dataset/cleaned/train.txt --output model.nn\n"
            "  python3 src/my_torch.py -eval --model model.nn --dataset dataset/cleaned/test.txt\n"
            "  python3 src/my_torch.py -clean\n"
            "  python3 src/my_torch.py -clean --input dataset --output dataset/cleaned_v2\n\n"
            "--input nécessite -clean. --output s'utilise avec -clean ou -train.\n"
            "Pour -clean, le dossier de sortie doit être nouveau."
        ),
    )
    parser.add_argument("-h", "--help", action="help", help="afficher cette aide et quitter")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("-clean", "--clean", action="store_true", help="nettoyer les données FEN")
    modes.add_argument("-train", "--train", action="store_true", help="entraîner et sauvegarder le réseau")
    modes.add_argument("-eval", "--eval", action="store_true", help="évaluer un réseau sauvegardé")
    parser.add_argument("--input", type=Path, help="racine des données sources (défaut : dataset)")
    parser.add_argument("--model", type=Path, help="fichier .nn à charger pour -eval")
    add_training_arguments(parser)
    parser.set_defaults(dataset=None, output=None, classes=None, epochs=None,
                        batch_size=None, max_per_class=None, lr=None,
                        use_bn=None, boost_mate=None, loss=None, seed=None,
                        dropout=None, weight_decay=None, patience=None, monitor=None,
                        label_smoothing=None, augment_colors=None, encoding=None,
                        init_seed=None, validation_dataset=None)
    return parser


def parse_arguments(argv=None):
    """Lire les options et vérifier qu'elles correspondent au mode choisi."""
    parser = helper()
    args = parser.parse_args(argv)
    if args.input is not None and not args.clean:
        parser.error("--input nécessite -clean")
    if args.output is not None and not (args.clean or args.train):
        parser.error("--output nécessite -clean ou -train")
    if args.model is not None and not args.eval:
        parser.error("--model nécessite -eval")
    if args.eval and args.model is None:
        parser.error("-eval nécessite --model")
    if args.dataset is not None and not (args.train or args.eval):
        parser.error("--dataset nécessite -train ou -eval")
    training_options = ("classes", "epochs", "max_per_class", "lr", "use_bn", "boost_mate", "loss",
                        "seed", "dropout", "weight_decay", "patience", "monitor", "label_smoothing",
                        "augment_colors", "encoding", "init_seed", "validation_dataset")
    for option in training_options:
        if getattr(args, option) is not None and not args.train:
            parser.error(f"--{option.replace('_', '-')} nécessite -train")
    if args.batch_size is not None and not (args.train or args.eval):
        parser.error("--batch-size nécessite -train ou -eval")
    if args.eval and args.batch_size is None:
        args.batch_size = 32
    defaults = add_training_arguments(argparse.ArgumentParser(add_help=False)).parse_args([])
    for name in (*training_options, "batch_size"):
        if getattr(args, name) is None:
            setattr(args, name, getattr(defaults, name))
    for name in ("epochs", "batch_size", "max_per_class", "patience"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} doit être strictement positif")
    args.input = args.input if args.input is not None else Path("dataset")
    args.output = args.output if args.output is not None else (
        "my_torch_network_full.nn" if args.train else "dataset/cleaned")
    if args.clean:
        args.output = Path(args.output)
    if args.dataset is None:
        args.dataset = "dataset/cleaned/test.txt" if args.eval else "dataset/cleaned/train.txt"
    return args


def run_cleaning(source, output):
    """Lancer le nettoyage, afficher son bilan et renvoyer le code de sortie."""
    try:
        report = clean_dataset(source, output)
    except (OSError, ValueError) as error:
        print(f"dataclean : {error}", file=sys.stderr)
        return 84
    print(f"{report['kept_positions']} positions conservées dans {output / 'cleaned.txt'}")
    for name, summary in report["for_train"].items():
        print(f"{summary['kept_positions']} positions conservées dans {output / name}")
    return 0


def main(argv=None):
    """Choisir le traitement demandé par les arguments de lancement."""
    args = parse_arguments(argv)
    if args.clean:
        return run_cleaning(args.input, args.output)
    try:
        if args.train:
            train_model(args)
        elif args.eval:
            evaluate_model(args.model, args.dataset, args.batch_size)
        else:
            helper().print_help()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"my_torch : {error}", file=sys.stderr)
        return 84
    return 0


if __name__ == "__main__":
    sys.exit(main())
