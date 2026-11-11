#!/usr/bin/env python3
"""
Playable Chess AI using Minimax with Alpha-Beta Pruning,
guided by MyTorch neural network state evaluation and piece-square heuristics.
Bonus feature for MY_TORCH_ < A CLASH OF KINGS />
"""

import sys
import os
import argparse
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chess_engine.rules import (
    generate_legal_moves,
    is_king_in_check,
    evaluate_game_state,
)
from src.chess_engine.encoders import encode_board, model_encoding
from src.neural_network.model import Sequential

PIECE_SYMBOLS = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
    ".": "·",
}

PIECE_VALUES = {
    "P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000,
    "p": -100, "n": -320, "b": -330, "r": -500, "q": -900, "k": -20000,
    ".": 0,
}


def create_initial_board() -> List[List[str]]:
    return [
        ["r", "n", "b", "q", "k", "b", "n", "r"],
        ["p", "p", "p", "p", "p", "p", "p", "p"],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"],
    ]


def board_to_fen(board: List[List[str]], active_color: str, halfmove: int = 0, fullmove: int = 1) -> str:
    ranks = []
    for row in board:
        empty = 0
        rank_str = ""
        for cell in row:
            if cell == ".":
                empty += 1
            else:
                if empty > 0:
                    rank_str += str(empty)
                    empty = 0
                rank_str += cell
        if empty > 0:
            rank_str += str(empty)
        ranks.append(rank_str)
    board_part = "/".join(ranks)
    return f"{board_part} {active_color} KQkq - {halfmove} {fullmove}"


def display_board(board: List[List[str]], use_unicode: bool = False):
    print("    a   b   c   d   e   f   g   h")
    print("  +---+---+---+---+---+---+---+---+")
    for r in range(8):
        row_str = f"{8 - r} |"
        for c in range(8):
            cell = board[r][c]
            symbol = PIECE_SYMBOLS[cell] if use_unicode else cell
            row_str += f" {symbol} |"
        row_str += f" {8 - r}"
        print(row_str)
        print("  +---+---+---+---+---+---+---+---+")
    print("    a   b   c   d   e   f   g   h\n")


def parse_uci_move(uci: str) -> Optional[Tuple[int, int, int, int]]:
    """Convert UCI like 'e2e4' to (r1, c1, r2, c2)."""
    uci = uci.strip().lower()
    if len(uci) < 4:
        return None
    c1 = ord(uci[0]) - ord("a")
    r1 = 8 - int(uci[1])
    c2 = ord(uci[2]) - ord("a")
    r2 = 8 - int(uci[3])
    if not (0 <= r1 < 8 and 0 <= c1 < 8 and 0 <= r2 < 8 and 0 <= c2 < 8):
        return None
    return (r1, c1, r2, c2)


def move_to_uci(move: Tuple[int, int, int, int]) -> str:
    r1, c1, r2, c2 = move
    return f"{chr(c1 + ord('a'))}{8 - r1}{chr(c2 + ord('a'))}{8 - r2}"


def apply_move(board: List[List[str]], move: Tuple[int, int, int, int]) -> List[List[str]]:
    r1, c1, r2, c2 = move
    new_board = [row[:] for row in board]
    piece = new_board[r1][c1]
    new_board[r1][c1] = "."

    # Handle basic pawn promotion to Queen
    if piece == "P" and r2 == 0:
        piece = "Q"
    elif piece == "p" and r2 == 7:
        piece = "q"

    new_board[r2][c2] = piece
    return new_board


class ChessAI:
    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[Sequential] = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = Sequential.load(model_path)
            except Exception:
                self.model = None

    def evaluate_position(self, board: List[List[str]], active_color: str) -> float:
        """
        Evaluate board state using material balance and neural network tactical scoring.
        Positive score favors White, negative favors Black.
        """
        # 1. Material score
        material = sum(PIECE_VALUES[board[r][c]] for r in range(8) for c in range(8))

        # 2. Tactical state bonus
        fen = board_to_fen(board, active_color)
        nn_bonus = 0.0

        if self.model is not None:
            try:
                vec = encode_board(fen, encoding=model_encoding(self.model)).reshape(1, -1)
                preds = self.model.predict(vec)[0]
                # If model detects Checkmate, huge value
                if len(preds) == 3:
                    # [Nothing, Check, Checkmate]
                    if active_color == "b":
                        nn_bonus += preds[2] * 5000.0 + preds[1] * 200.0
                    else:
                        nn_bonus -= preds[2] * 5000.0 + preds[1] * 200.0
                elif len(preds) == 5:
                    # 5-class: [Nothing, Check White, Check Black, Checkmate White, Checkmate Black]
                    nn_bonus += (preds[3] - preds[4]) * 5000.0 + (preds[1] - preds[2]) * 200.0
            except Exception:
                pass
        else:
            state = evaluate_game_state(fen)
            if state == "Checkmate":
                nn_bonus = -10000.0 if active_color == "w" else 10000.0
            elif state == "Check":
                nn_bonus = -300.0 if active_color == "w" else 300.0

        return float(material + nn_bonus)

    def minimax(
        self,
        board: List[List[str]],
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
    ) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        color = "w" if is_maximizing else "b"
        legal_moves = generate_legal_moves(board, color)

        if not legal_moves:
            fen = board_to_fen(board, color)
            if is_king_in_check(board, color):
                # Checkmate
                return (-50000.0 if is_maximizing else 50000.0), None
            # Stalemate
            return 0.0, None

        if depth == 0:
            return self.evaluate_position(board, color), None

        best_move = legal_moves[0]

        if is_maximizing:
            max_eval = -float("inf")
            for move in legal_moves:
                new_board = apply_move(board, move)
                eval_val, _ = self.minimax(new_board, depth - 1, alpha, beta, False)
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = move
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float("inf")
            for move in legal_moves:
                new_board = apply_move(board, move)
                eval_val, _ = self.minimax(new_board, depth - 1, alpha, beta, True)
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = move
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def choose_move(self, board: List[List[str]], color: str, depth: int = 2) -> Optional[Tuple[int, int, int, int]]:
        is_max = (color == "w")
        _, move = self.minimax(board, depth, -float("inf"), float("inf"), is_max)
        return move


def main():
    parser = argparse.ArgumentParser(description="MyTorch Interactive Chess AI")
    parser.add_argument("--model", type=str, default="my_torch_network_full.nn", help="Path to pre-trained .nn model")
    parser.add_argument("--depth", type=int, default=2, help="Search depth (default: 2)")
    parser.add_argument("--color", type=str, choices=["w", "b"], default="w", help="Human color (w/b)")
    parser.add_argument("--auto", type=int, default=0, help="Run N automated turns for testing")
    parser.add_argument("--unicode", action="store_true", help="Display unicode chess glyphs")

    args = parser.parse_args()

    board = create_initial_board()
    ai = ChessAI(args.model if os.path.exists(args.model) else "my_torch_network_basic.nn")

    print("==================================================")
    print("   MY_TORCH_ CHESS AI - < A CLASH OF KINGS />   ")
    print(f"   Model: {args.model if os.path.exists(args.model) else 'Material Heuristic'} (Depth {args.depth})")
    print("==================================================\n")

    current_color = "w"
    human_color = args.color
    turn = 1

    # Automated test mode
    if args.auto > 0:
        print(f"Running automated demonstration for {args.auto} moves...")
        for step in range(args.auto):
            move = ai.choose_move(board, current_color, depth=args.depth)
            if not move:
                print(f"Game finished at turn {turn}.")
                break
            print(f"Turn {turn} ({'White' if current_color == 'w' else 'Black'}): {move_to_uci(move)}")
            board = apply_move(board, move)
            current_color = "b" if current_color == "w" else "w"
            if current_color == "w":
                turn += 1
        print("Automated demonstration finished successfully.")
        sys.exit(0)

    # Interactive loop
    display_board(board, use_unicode=args.unicode)

    while True:
        legal_moves = generate_legal_moves(board, current_color)
        if not legal_moves:
            fen = board_to_fen(board, current_color)
            if is_king_in_check(board, current_color):
                winner = "Black" if current_color == "w" else "White"
                print(f"Checkmate! {winner} wins.")
            else:
                print("Stalemate (Pat)! The game is a draw.")
            break

        if current_color == human_color:
            prompt = f"[{'White' if current_color == 'w' else 'Black'}] Enter move (e.g. e2e4, or 'quit'): "
            try:
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGame aborted by user.")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                print("Exiting game.")
                break

            move = parse_uci_move(user_input)
            if move is None or move not in legal_moves:
                print("Invalid or illegal move! Please try again.")
                continue

            board = apply_move(board, move)
        else:
            print(f"AI ({'White' if current_color == 'w' else 'Black'}) is thinking...")
            move = ai.choose_move(board, current_color, depth=args.depth)
            if not move:
                print("AI has no legal moves.")
                break
            print(f"AI plays: {move_to_uci(move)}\n")
            board = apply_move(board, move)

        display_board(board, use_unicode=args.unicode)
        current_color = "b" if current_color == "w" else "w"
        if current_color == "w":
            turn += 1


if __name__ == "__main__":
    main()
