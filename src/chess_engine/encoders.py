import numpy as np
from functools import partial
from typing import Dict, Any, Union
from src.chess_engine.fen_parser import parse_fen

PIECE_TO_INDEX = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}

INPUT_DIM = 64 * 12 + 1 + 4  # 773 features
EN_PASSANT = slice(INPUT_DIM, INPUT_DIM + 64)
ATTACK_MAPS = slice(EN_PASSANT.stop, EN_PASSANT.stop + 128)
KING_RAYS = slice(ATTACK_MAPS.stop, ATTACK_MAPS.stop + 2 * 8 * 19)
KING_CONTEXT = slice(KING_RAYS.stop, KING_RAYS.stop + 2 * 8 * 4)
TACTICAL_INPUT_DIM = KING_CONTEXT.stop  # 1333 features, schema tactical-v1
ENCODING_DIMS = {"basic": INPUT_DIM, "tactical": TACTICAL_INPUT_DIM}
DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))


def input_dimension(encoding):
    if encoding not in ENCODING_DIMS:
        raise ValueError(f"Unknown chess encoding: {encoding}")
    return ENCODING_DIMS[encoding]


def model_encoding(model):
    """Old models without encoding metadata keep their original 773 inputs."""
    encoding = model.metadata.get("encoding", "basic")
    if model.metadata.get("encoding_version", 1) != 1:
        raise ValueError("Unsupported chess encoding version")
    dimension = input_dimension(encoding)
    first_input = next((layer.in_features for layer in model.layers
                        if hasattr(layer, "in_features")), None)
    if first_input != dimension:
        raise ValueError(f"Model encoding {encoding} expects {dimension} inputs; network has {first_input}")
    return encoding


def _attack_maps(board):
    """Geometric attacks, including defended occupied squares; pinned pieces attack.

    This is a representation of the position, not legal-move generation or a
    state classifier. Sliding attacks stop after the first occupied square.
    """
    maps = np.zeros((8, 8, 2), dtype=np.float32)
    for r, row in enumerate(board):
        for c, piece in enumerate(row):
            if piece == ".":
                continue
            color = 0 if piece.isupper() else 1
            kind = piece.upper()
            if kind == "P":
                dr = -1 if color == 0 else 1
                directions = ((dr, -1), (dr, 1))
            elif kind == "N":
                directions = ((-2, -1), (-2, 1), (-1, -2), (-1, 2),
                              (1, -2), (1, 2), (2, -1), (2, 1))
            elif kind == "K":
                directions = DIRECTIONS
            else:
                directions = tuple((dr, dc) for dr, dc in DIRECTIONS
                                   if kind == "Q" or (kind == "B" and dr and dc)
                                   or (kind == "R" and (dr == 0 or dc == 0)))
            sliding = kind in ("B", "R", "Q")
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                while 0 <= nr < 8 and 0 <= nc < 8:
                    maps[nr, nc, color] = 1
                    if not sliding or board[nr][nc] != ".":
                        break
                    nr, nc = nr + dr, nc + dc
    return maps


def _encode_tactical(fen_data, vector):
    board = fen_data["board"]
    en_passant = fen_data["en_passant"]
    if en_passant != "-":
        if len(en_passant) != 2 or en_passant[0] not in "abcdefgh" or en_passant[1] not in "36":
            raise ValueError(f"Invalid en passant square: {en_passant}")
        vector[EN_PASSANT.start + (8 - int(en_passant[1])) * 8 + ord(en_passant[0]) - ord("a")] = 1
    vector[ATTACK_MAPS] = _attack_maps(board).ravel()
    rays = vector[KING_RAYS].reshape(2, 8, 19)
    context = vector[KING_CONTEXT].reshape(2, 8, 4)
    for color, king in enumerate(("K", "k")):
        kr, kc = next((r, c) for r in range(8) for c in range(8) if board[r][c] == king)
        # Removing the king exposes lines that its current square obstructs.
        without_king = [row[:] for row in board]
        without_king[kr][kc] = "."
        enemy_attacks = _attack_maps(without_king)[:, :, 1 - color]
        for direction, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = kr + dr, kc + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                piece = board[nr][nc]
                occupied = piece != "."
                friendly = occupied and piece.isupper() == (color == 0)
                context[color, direction] = (1, friendly, occupied and not friendly, enemy_attacks[nr, nc])
            for distance in range(1, 8):
                nr, nc = kr + dr * distance, kc + dc * distance
                if not (0 <= nr < 8 and 0 <= nc < 8):
                    break
                piece = board[nr][nc]
                if piece != ".":
                    rays[color, direction, PIECE_TO_INDEX[piece]] = 1
                    rays[color, direction, 12 + distance - 1] = 1
                    break


def encode_board(fen_data: Union[str, Dict[str, Any]], encoding="basic") -> np.ndarray:
    """
    Encode chess position into 773 basic or 1333 tactical features.
    - 768 elements: 12 binary channels for 64 board squares
    - 1 element: active turn (1.0 for White, 0.0 for Black)
    - 4 elements: castling availability (K, Q, k, q)
    Tactical v1 appends en passant, geometric attacks and king-relative context.
    """
    if isinstance(fen_data, str):
        fen_data = parse_fen(fen_data)

    board = fen_data["board"]
    vector = np.zeros(input_dimension(encoding), dtype=np.float32)

    # 1. Piece positions
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece in PIECE_TO_INDEX:
                channel = PIECE_TO_INDEX[piece]
                square_idx = r * 8 + c
                vector[square_idx * 12 + channel] = 1.0

    offset = 64 * 12

    # 2. Active player
    vector[offset] = 1.0 if fen_data["active_color"] == "w" else 0.0
    offset += 1

    # 3. Castling rights
    castling = fen_data["castling"]
    vector[offset + 0] = 1.0 if "K" in castling else 0.0
    vector[offset + 1] = 1.0 if "Q" in castling else 0.0
    vector[offset + 2] = 1.0 if "k" in castling else 0.0
    vector[offset + 3] = 1.0 if "q" in castling else 0.0
    if encoding == "tactical":
        _encode_tactical(fen_data, vector)
    return vector


def encode_boards_parallel(fen_list: list, max_workers: int = None, encoding="basic") -> np.ndarray:
    """
    Encode multiple FEN chessboards into a 2D feature matrix (N, input_dimension)
    using multi-threaded parallel execution across available CPU cores.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    dimension = input_dimension(encoding)
    encoder = partial(encode_board, encoding=encoding)
    if not fen_list:
        return np.empty((0, dimension), dtype=np.float32)

    if len(fen_list) < 50:
        return np.array([encoder(f) for f in fen_list], dtype=np.float32)

    workers = max_workers or min(16, max(2, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        vectors = list(executor.map(encoder, fen_list))
    return np.array(vectors, dtype=np.float32)

