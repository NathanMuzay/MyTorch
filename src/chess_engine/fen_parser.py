from typing import List, Dict, Any

VALID_PIECES = set("pnbrqkPNBRQK")


def parse_fen(fen_string: str) -> Dict[str, Any]:
    """
    Parse a Forsyth-Edwards Notation (FEN) string into a structured board representation.
    Raises ValueError if the FEN is invalid.
    """
    if not isinstance(fen_string, str):
        raise ValueError("FEN must be a string")

    parts = fen_string.strip().split()
    if len(parts) < 1:
        raise ValueError("Empty FEN string")

    piece_placement = parts[0]
    active_color = parts[1] if len(parts) > 1 else "w"
    castling = parts[2] if len(parts) > 2 else "-"
    en_passant = parts[3] if len(parts) > 3 else "-"
    halfmove = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
    fullmove = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 1

    if active_color not in ("w", "b"):
        raise ValueError(f"Invalid active color: {active_color}")

    ranks = piece_placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN must contain 8 ranks, got {len(ranks)}")

    board: List[List[str]] = []
    white_kings = 0
    black_kings = 0

    for rank in ranks:
        row: List[str] = []
        for ch in rank:
            if ch.isdigit():
                digit = int(ch)
                if digit < 1 or digit > 8:
                    raise ValueError(f"Invalid digit in rank: {digit}")
                row.extend(["."] * digit)
            elif ch in VALID_PIECES:
                row.append(ch)
                if ch == "K":
                    white_kings += 1
                elif ch == "k":
                    black_kings += 1
            else:
                raise ValueError(f"Invalid character in FEN rank: {ch}")

        if len(row) != 8:
            raise ValueError(f"Rank '{rank}' does not resolve to 8 squares (got {len(row)})")
        board.append(row)

    if white_kings != 1 or black_kings != 1:
        raise ValueError(f"Board must have exactly 1 white king and 1 black king (W:{white_kings}, B:{black_kings})")

    return {
        "board": board,
        "active_color": active_color,
        "castling": castling,
        "en_passant": en_passant,
        "halfmove": halfmove,
        "fullmove": fullmove,
    }
