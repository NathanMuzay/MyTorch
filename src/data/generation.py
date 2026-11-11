#!/usr/bin/env python3
"""
Generate a diverse and balanced dataset of chessboard positions in FEN format
with labels ('Nothing', 'Check', 'Checkmate').
"""

import sys
import os
import random
from typing import List, Tuple

from src.chess_engine.rules import evaluate_game_state


def board_to_fen(board: List[List[str]], active_color: str, castling: str = "-", en_passant: str = "-", half: int = 0, full: int = 1) -> str:
    ranks = []
    for row in board:
        rank_str = ""
        empty_count = 0
        for cell in row:
            if cell == ".":
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += cell
        if empty_count > 0:
            rank_str += str(empty_count)
        ranks.append(rank_str)
    return f"{'/'.join(ranks)} {active_color} {castling} {en_passant} {half} {full}"


def empty_board() -> List[List[str]]:
    return [["." for _ in range(8)] for _ in range(8)]


def add_random_benign_pieces(board: List[List[str]], count: int = 2):
    """Add a few random pieces far from the active action."""
    pool = ["P", "N", "B", "p", "n", "b"]
    for _ in range(count):
        p = random.choice(pool)
        for _ in range(10):
            r, c = random.randint(3, 5), random.randint(0, 7)
            if board[r][c] == ".":
                board[r][c] = p
                break


def generate_backrank_mate(turn: str = "b") -> str:
    board = empty_board()
    col = random.randint(1, 6)
    if turn == "b":
        board[0][col] = "k"
        for dc in (-1, 0, 1):
            board[1][col + dc] = "p"
        rook_col = random.choice([0, 7]) if col not in (0, 7) else (7 if col == 0 else 0)
        board[0][rook_col] = "R"
        board[7][col] = "K"
    else:
        board[7][col] = "K"
        for dc in (-1, 0, 1):
            board[6][col + dc] = "P"
        rook_col = random.choice([0, 7]) if col not in (0, 7) else (7 if col == 0 else 0)
        board[7][rook_col] = "r"
        board[0][col] = "k"
    add_random_benign_pieces(board, random.randint(0, 3))
    return board_to_fen(board, turn)


def generate_queen_king_mate(turn: str = "b") -> str:
    board = empty_board()
    edge_col = random.choice([0, 7])
    edge_row = random.randint(0, 7)
    if turn == "b":
        board[edge_row][edge_col] = "k"
        q_col = 1 if edge_col == 0 else 6
        board[edge_row][q_col] = "Q"
        k_col = 2 if edge_col == 0 else 5
        board[edge_row][k_col] = "K"
    else:
        board[edge_row][edge_col] = "K"
        q_col = 1 if edge_col == 0 else 6
        board[edge_row][q_col] = "q"
        k_col = 2 if edge_col == 0 else 5
        board[edge_row][k_col] = "k"
    add_random_benign_pieces(board, random.randint(0, 3))
    return board_to_fen(board, turn)


def generate_rook_king_mate(turn: str = "b") -> str:
    board = empty_board()
    edge_col = random.randint(1, 6)
    if turn == "b":
        board[0][edge_col] = "k"
        board[2][edge_col] = "K"
        r_col = 0 if edge_col > 3 else 7
        board[0][r_col] = "R"
    else:
        board[7][edge_col] = "K"
        board[5][edge_col] = "k"
        r_col = 0 if edge_col > 3 else 7
        board[7][r_col] = "r"
    add_random_benign_pieces(board, random.randint(0, 3))
    return board_to_fen(board, turn)


def generate_check_position(turn: str = "b") -> str:
    """Generate a position where active player's King is in check but has an escape or block."""
    board = empty_board()
    kr, kc = random.randint(1, 6), random.randint(1, 6)
    k_char = "k" if turn == "b" else "K"
    board[kr][kc] = k_char

    # Other king far away
    okr, okc = (0, 0) if kr > 3 else (7, 7)
    board[okr][okc] = "K" if turn == "b" else "k"

    # Attacking rook or queen on the same rank, leaving open escape squares
    attacker = "R" if turn == "b" else "r"
    att_col = 0 if kc > 3 else 7
    board[kr][att_col] = attacker
    add_random_benign_pieces(board, random.randint(0, 4))
    return board_to_fen(board, turn)


def generate_normal_position(turn: str = "b") -> str:
    """Generate quiet standard middle/endgame positions."""
    board = empty_board()
    kr1, kc1 = random.randint(0, 2), random.randint(0, 7)
    board[kr1][kc1] = "k"
    kr2, kc2 = random.randint(5, 7), random.randint(0, 7)
    board[kr2][kc2] = "K"

    pieces_pool = ["R", "B", "N", "P", "r", "b", "n", "p"]
    for _ in range(random.randint(4, 10)):
        p = random.choice(pieces_pool)
        for _ in range(10):
            r, c = random.randint(2, 5), random.randint(0, 7)
            if board[r][c] == ".":
                board[r][c] = p
                break
    return board_to_fen(board, turn)


def build_dataset(total_samples: int = 1200) -> List[Tuple[str, str]]:
    results = []
    seen = set()

    seeds = [
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
        "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3",
        "8/8/8/8/8/8/8/k1K5 w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/4R1K1 b - - 0 1",
        "4r1k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/1r4K1 w - - 0 1",
        "1r4k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",
        "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
    ]
    for s in seeds:
        try:
            lbl = evaluate_game_state(s)
            results.append((s, lbl))
            seen.add(s)
        except Exception:
            pass

    target_per_class = total_samples // 3
    counts = {"Nothing": 0, "Check": 0, "Checkmate": 0}
    for _, l in results:
        counts[l] = counts.get(l, 0) + 1

    mate_funcs = [generate_backrank_mate, generate_queen_king_mate, generate_rook_king_mate]

    attempts = 0
    max_attempts = total_samples * 100

    while len(results) < total_samples and attempts < max_attempts:
        attempts += 1
        turn = random.choice(["w", "b"])

        # Decide which type to try generating based on current quota
        min_class = min(counts, key=counts.get)
        if min_class == "Checkmate":
            fen = random.choice(mate_funcs)(turn)
        elif min_class == "Check":
            fen = generate_check_position(turn)
        else:
            fen = generate_normal_position(turn)

        if fen in seen:
            continue

        try:
            state = evaluate_game_state(fen)
        except Exception:
            continue

        if counts[state] < target_per_class + 20:
            counts[state] += 1
            results.append((fen, state))
            seen.add(fen)

    random.shuffle(results)
    return results


def main():
    out_file = sys.argv[1] if len(sys.argv) > 1 else "dataset/examples/train_chessboards.txt"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 900

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    dataset = build_dataset(count)

    with open(out_file, "w", encoding="utf-8") as f:
        for fen, label in dataset:
            f.write(f"{fen} {label}\n")

    counts = {}
    for _, lbl in dataset:
        counts[lbl] = counts.get(lbl, 0) + 1

    print(f"[OK] Generated {len(dataset)} samples saved to {out_file}:")
    for k, v in counts.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
