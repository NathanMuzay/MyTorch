"""Display a dataset FEN position in a native window, with optional SVG export."""

import argparse
from pathlib import Path
import re
import sys
import tempfile
import xml.etree.ElementTree as ET


PIECES = dict(zip("kqrbnp", "♚♛♜♝♞♟"))
NAMES = dict(zip("kqrbnp", ("roi", "dame", "tour", "fou", "cavalier", "pion")))


def read_position(path, line_number):
    with path.open(encoding="utf-8") as dataset:
        for number, line in enumerate(dataset, start=1):
            if number == line_number:
                break
        else:
            raise ValueError(f"le fichier ne contient pas de ligne {line_number}")

    fields = line.split()
    if len(fields) < 6:
        raise ValueError("la ligne doit contenir les six champs d'une position FEN")
    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise ValueError("la position FEN doit contenir huit rangées")
    board = []
    for rank in ranks:
        row = []
        for symbol in rank:
            if symbol in "12345678":
                row.extend([None] * int(symbol))
            elif symbol.lower() in PIECES:
                row.append(symbol)
            else:
                raise ValueError(f"pièce FEN inconnue : {symbol}")
        if len(row) != 8:
            raise ValueError("chaque rangée FEN doit contenir huit cases")
        board.append(row)
    if fields[1] not in ("w", "b"):
        raise ValueError("le joueur au trait doit être 'w' ou 'b'")
    if not re.fullmatch(r"-|K?Q?k?q?", fields[2]):
        raise ValueError("droits de roque FEN invalides")
    if not re.fullmatch(r"-|[a-h][36]", fields[3]):
        raise ValueError("case de prise en passant FEN invalide")
    if not fields[4].isascii() or not fields[4].isdigit():
        raise ValueError("compteur de demi-coups FEN invalide")
    if not fields[5].isascii() or not fields[5].isdigit() or int(fields[5]) < 1:
        raise ValueError("numéro de coup FEN invalide")
    return board, fields[:6], " ".join(fields[6:])


def render_svg(board, fen, label, filename, line_number):
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "viewBox": "0 0 640 780",
        "width": "640", "height": "780", "role": "img",
        "aria-labelledby": "title description",
    })

    def element(tag, text=None, **attributes):
        node = ET.SubElement(svg, tag, {
            key.replace("_", "-"): str(value) for key, value in attributes.items()
        })
        node.text = text
        return node

    element("title", f"{filename} — ligne {line_number}", id="title")
    element("desc", f"Position FEN : {' '.join(fen)}. {label}", id="description")
    element("rect", width=640, height=780, fill="#18202c")

    def text(content, x, y, size=18, **attributes):
        return element("text", content, x=x, y=y, font_size=size,
                       font_family="DejaVu Sans, sans-serif", fill="#e8edf4", **attributes)

    text(filename, 320, 35, 22, text_anchor="middle", textLength=min(len(filename) * 13, 590),
         lengthAdjust="spacingAndGlyphs")
    text(f"Ligne {line_number} · Trait aux {'blancs' if fen[1] == 'w' else 'noirs'}", 320, 66,
         text_anchor="middle")
    for row, rank in enumerate(board):
        text(str(8 - row), 28, 140 + row * 68, text_anchor="middle")
        for col, piece in enumerate(rank):
            x, y = 48 + col * 68, 96 + row * 68
            element("rect", x=x, y=y, width=68, height=68,
                    fill="#eadcc2" if (row + col) % 2 == 0 else "#8ba17a")
            if piece:
                color = "blanc" if piece.isupper() else "noir"
                node = element("text", PIECES[piece.lower()], x=x + 34, y=y + 53,
                               font_size=57, font_family="DejaVu Sans, serif",
                               text_anchor="middle", fill="#fffaf0" if piece.isupper() else "#18202c",
                               stroke="#18202c" if piece.isupper() else "#fffaf0",
                               stroke_width=1, paint_order="stroke fill")
                ET.SubElement(node, "title").text = f"{NAMES[piece.lower()]} {color} — {'abcdefgh'[col]}{8-row}"
    for col, file in enumerate("abcdefgh"):
        text(file, 82 + col * 68, 665, text_anchor="middle")
    text(label or "Position sans étiquette", 320, 701, 22, text_anchor="middle")
    text(" ".join(fen), 320, 737, 13, text_anchor="middle",
         textLength=min(len(" ".join(fen)) * 7.8, 590), lengthAdjust="spacingAndGlyphs")
    return ET.tostring(svg, encoding="unicode", xml_declaration=True)


def show_window(board, fen, label, filename, line_number):
    try:
        import tkinter as tk
    except ImportError as error:
        raise RuntimeError("Tkinter est requis pour la fenêtre (paquet python3-tk).") from error
    try:
        window = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(f"impossible d'ouvrir la fenêtre : {error}") from error
    window.title(f"Échecs — {filename} — ligne {line_number}")
    window.resizable(False, False)
    window.bind("<Escape>", lambda event: window.destroy())
    canvas = tk.Canvas(window, width=640, height=780, bg="#18202c", highlightthickness=0)
    canvas.pack()

    def text(content, x, y, size=18, **attributes):
        canvas.create_text(x, y, text=content, font=("DejaVu Sans", size),
                           fill="#e8edf4", **attributes)

    text(filename, 320, 30, 20, width=600)
    text(f"Ligne {line_number} · Trait aux {'blancs' if fen[1] == 'w' else 'noirs'}", 320, 66, 14)
    for row, rank in enumerate(board):
        text(str(8 - row), 28, 130 + row * 68, 14)
        for col, piece in enumerate(rank):
            x, y = 48 + col * 68, 96 + row * 68
            canvas.create_rectangle(x, y, x + 68, y + 68, outline="",
                                    fill="#eadcc2" if (row + col) % 2 == 0 else "#8ba17a")
            if piece:
                # Outline the white silhouettes so they remain visible on light squares.
                if piece.isupper():
                    for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                                   (0, 1), (1, -1), (1, 0), (1, 1)):
                        canvas.create_text(x + 34 + dx, y + 34 + dy,
                                           text=PIECES[piece.lower()], font=("DejaVu Sans", 42),
                                           fill="#18202c")
                canvas.create_text(x + 34, y + 34, text=PIECES[piece.lower()],
                                   font=("DejaVu Sans", 42),
                                   fill="#fffaf0" if piece.isupper() else "#18202c")
    for col, file in enumerate("abcdefgh"):
        text(file, 82 + col * 68, 659, 14)
    text(label or "Position sans étiquette", 320, 701, 18)
    text(" ".join(fen), 320, 743, 10, width=600)
    window.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Affiche une position d'échecs du dataset (lignes numérotées à partir de 1).",
        usage="%(prog)s fichier -N [--output image.svg] [--no-open]",
    )
    parser.add_argument("file", type=Path, help="fichier contenant les positions FEN")
    parser.add_argument("line", help="numéro de ligne, par exemple -30")
    parser.add_argument("--output", type=Path, help="chemin de l'image SVG à enregistrer")
    parser.add_argument("--no-open", action="store_true", help="enregistre sans ouvrir la fenêtre")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"-?[1-9][0-9]*", args.line):
        parser.error("utilisez un numéro de ligne positif, par exemple -30")
    line_number = abs(int(args.line))
    try:
        board, fen, label = read_position(args.file, line_number)
        if args.output or args.no_open:
            svg = render_svg(board, fen, label, args.file.name, line_number)
            if args.output:
                output = args.output.expanduser().resolve()
            else:
                with tempfile.NamedTemporaryFile(prefix=f"visu-ligne-{line_number}-", suffix=".svg", delete=False) as image:
                    output = Path(image.name)
            output.write_text(svg, encoding="utf-8")
            print(f"Image de la ligne {line_number} : {output}")
        if not args.no_open:
            show_window(board, fen, label, args.file.name, line_number)
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        print(f"visu : {error}", file=sys.stderr)
        return 84
    return 0


if __name__ == "__main__":
    sys.exit(main())
