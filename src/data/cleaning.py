"""Nettoyage FEN : validation, normalisation et déduplication.

Choix et limites documentés dans doc/datacleaning.md.
Ce module expose clean_dataset() ; le lancement se fait via my_torch.py -clean.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import tempfile


@dataclass(frozen=True)
class Sample:
    """Une observation normalisée, sans encodage spécifique à un modèle."""

    fen: str
    label: str

    @property
    def position_key(self) -> str:
        """Ignorer les compteurs regroupe les répétitions d'un même état de jeu."""
        return " ".join(self.fen.split()[:4])


def parse_line(line: str) -> Sample:
    """Lever ValueError si une ligne ne respecte pas le contrat du dataset."""
    fields = line.split()
    if len(fields) not in (7, 8):
        raise ValueError("format attendu : six champs FEN puis une annotation")
    board, turn, castling, en_passant, halfmove, fullmove = fields[:6]
    ranks = board.split("/")
    if len(ranks) != 8:
        raise ValueError("l'échiquier doit contenir huit rangées")
    normalized = []
    for rank in ranks:
        cells = []
        for char in rank:
            if char in "12345678":
                cells.extend("." * int(char))
            elif char in "pnbrqkPNBRQK":
                cells.append(char)
            else:
                raise ValueError("symbole d'échiquier inconnu")
        if len(cells) != 8:
            raise ValueError("chaque rangée doit contenir huit cases")
        # Fusionner les suites de cases vides donne une représentation unique.
        normalized.append(re.sub(r"\.+", lambda match: str(len(match[0])), "".join(cells)))
    if board.count("K") != 1 or board.count("k") != 1:
        raise ValueError("un roi de chaque couleur est requis")
    if any(piece in ranks[0] + ranks[-1] for piece in "Pp"):
        raise ValueError("pion sur une rangée de promotion")
    if turn not in ("w", "b"):
        raise ValueError("joueur au trait invalide")
    if castling != "-" and (
        any(char not in "KQkq" for char in castling)
        or len(set(castling)) != len(castling)
    ):
        raise ValueError("droits de roque invalides")
    castling = "".join(char for char in "KQkq" if char in castling) or "-"
    if en_passant != "-":
        if not re.fullmatch(r"[a-h][36]", en_passant):
            raise ValueError("case de prise en passant invalide")
        if en_passant[1] != ("6" if turn == "w" else "3"):
            raise ValueError("prise en passant incompatible avec le joueur au trait")
    if not re.fullmatch(r"[0-9]+", halfmove):
        raise ValueError("compteur de demi-coups invalide")
    if not re.fullmatch(r"[0-9]+", fullmove) or int(fullmove) < 1:
        raise ValueError("numéro de coup invalide")
    label = " ".join(fields[6:])
    allowed = {"Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"}
    if label not in allowed:
        raise ValueError("annotation inconnue ou couleur manquante")
    if label != "Nothing" and fields[-1] != ("Black" if turn == "w" else "White"):
        raise ValueError("la couleur qui donne échec doit être opposée au joueur au trait")
    fen = " ".join(("/".join(normalized), turn, castling, en_passant,
                    str(int(halfmove)), str(int(fullmove))))
    return Sample(fen, label)


def clean_dataset(source: Path, output: Path) -> dict:
    """Nettoyer les catégories sources et chaque fichier de for_train.

    Les sources restent intactes. Un dossier de sortie existant est refusé.
    L'ordre des fichiers puis des lignes détermine l'exemplaire conservé.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    files = sorted(path for category in ("check", "checkmate", "nothing")
                   for path in (source / category).glob("*.txt") if path.is_file())
    if not files:
        raise ValueError("aucun fichier .txt dans check/, checkmate/ ou nothing/")
    # Empêcher qu'une future exécution relise accidentellement ses résultats.
    if any(output.is_relative_to(source / category)
           for category in ("check", "checkmate", "nothing", "for_train")):
        raise ValueError("la sortie doit être hors des dossiers de données d'entrée")
    training_files = sorted(path for path in (source / "for_train").glob("*.txt") if path.is_file())
    if any(path.name == "cleaned.txt" for path in training_files):
        raise ValueError("for_train/cleaned.txt utilise un nom réservé au corpus global")
    output.mkdir(parents=True, exist_ok=False)
    report = _clean_files(source, output / "cleaned.txt", files)
    report["for_train"] = clean_training_files(source, output)
    return report


def clean_training_files(source: Path, output: Path) -> dict:
    """Nettoyer chaque fichier de for_train séparément, en gardant son nom.

    Le dossier de sortie doit déjà exister. Les fichiers existants ne sont
    jamais écrasés. Aucun exemple ne change de jeu d'entraînement ou de test.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    files = sorted(path for path in (source / "for_train").glob("*.txt") if path.is_file())
    if any(output.is_relative_to(source / category)
           for category in ("check", "checkmate", "nothing", "for_train")):
        raise ValueError("la sortie doit être hors des dossiers de données d'entrée")
    for path in files:
        if path.name == "cleaned.txt" or (output / path.name).exists():
            raise FileExistsError(f"sortie déjà existante ou réservée : {output / path.name}")
    summaries = {}
    for path in files:
        summaries[path.name] = _clean_files(source, output / path.name, [path])
    return summaries


def _clean_files(source, output, files):
    """Écrire seulement le TXT nettoyé et retourner les compteurs en mémoire."""
    totals = Counter(lines_read=0, blank_lines=0, invalid_lines=0, valid_lines=0,
                     duplicate_lines=0, conflicting_lines=0)
    per_file = {}
    with tempfile.TemporaryDirectory(prefix="torch-cleaning-") as temporary:
        db = sqlite3.connect(str(Path(temporary) / "positions.sqlite"))
        try:
            db.execute("CREATE TABLE samples (key TEXT PRIMARY KEY, fen TEXT, label TEXT, "
                       "conflicted INTEGER DEFAULT 0)")
            for path in files:
                relative = path.relative_to(source).as_posix()
                counts = Counter()
                # Les octets UTF-8 invalides sont rejetés par ligne, sans perdre le fichier.
                with path.open("rb") as stream:
                    for raw in stream:
                        counts["lines_read"] += 1
                        try:
                            line = raw.decode("utf-8")
                            if not line.strip():
                                counts["blank_lines"] += 1
                                continue
                            sample = parse_line(line)
                        except (ValueError, UnicodeError):
                            counts["invalid_lines"] += 1
                            continue
                        counts["valid_lines"] += 1
                        previous = db.execute(
                            "SELECT label FROM samples WHERE key = ?",
                            (sample.position_key,)).fetchone()
                        if previous is None:
                            db.execute("INSERT INTO samples (key, fen, label) VALUES (?, ?, ?)",
                                       (sample.position_key, sample.fen, sample.label))
                        elif previous[0] == sample.label:
                            counts["duplicate_lines"] += 1
                        else:
                            counts["conflicting_lines"] += 1
                            db.execute("UPDATE samples SET conflicted = 1 WHERE key = ?",
                                       (sample.position_key,))
                db.commit()
                per_file[relative] = dict(counts)
                totals.update(counts)
            labels = Counter()
            with output.open("x", encoding="utf-8") as cleaned:
                for fen, label in db.execute(
                        "SELECT fen, label FROM samples "
                        "WHERE conflicted = 0 ORDER BY key"):
                    cleaned.write(f"{fen} {label}\n")
                    labels[label] += 1
            unique = db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            quarantined = db.execute("SELECT COUNT(*) FROM samples WHERE conflicted = 1").fetchone()[0]
        finally:
            db.close()
    report = {"schema_version": 1, "source": str(source), "files": per_file,
              "totals": dict(totals), "unique_positions": unique,
              "quarantined_positions": quarantined, "kept_positions": sum(labels.values()),
              "labels": dict(sorted(labels.items())),
              "position_key_fields": ["board", "turn", "castling", "en_passant"]}
    return report
