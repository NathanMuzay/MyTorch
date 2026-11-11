from typing import List, Tuple, Dict, Any, Union
from src.chess_engine.fen_parser import parse_fen


def is_in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def find_king(board: List[List[str]], king_char: str) -> Tuple[int, int]:
    for r in range(8):
        for c in range(8):
            if board[r][c] == king_char:
                return (r, c)
    raise ValueError(f"King {king_char} not found on board")


def is_square_attacked(board: List[List[str]], target_r: int, target_c: int, by_color: str) -> bool:
    """
    Check if (target_r, target_c) is under attack by pieces of `by_color` ('w' or 'b').
    """
    pawn_char = "P" if by_color == "w" else "p"
    knight_char = "N" if by_color == "w" else "n"
    bishop_char = "B" if by_color == "w" else "b"
    rook_char = "R" if by_color == "w" else "r"
    queen_char = "Q" if by_color == "w" else "q"
    king_char = "K" if by_color == "w" else "k"

    # 1. Pawn attacks
    # White pawns move up (decreasing r), so they attack from (target_r + 1, target_c ± 1)
    # Black pawns move down (increasing r), so they attack from (target_r - 1, target_c ± 1)
    pawn_row = target_r + 1 if by_color == "w" else target_r - 1
    for dc in (-1, 1):
        pc = target_c + dc
        if is_in_bounds(pawn_row, pc) and board[pawn_row][pc] == pawn_char:
            return True

    # 2. Knight attacks
    knight_offsets = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1),
    ]
    for dr, dc in knight_offsets:
        nr, nc = target_r + dr, target_c + dc
        if is_in_bounds(nr, nc) and board[nr][nc] == knight_char:
            return True

    # 3. King attacks (adjacent squares)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = target_r + dr, target_c + dc
            if is_in_bounds(nr, nc) and board[nr][nc] == king_char:
                return True

    # 4. Ray attacks (Rook & Queen)
    straight_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in straight_dirs:
        step = 1
        while True:
            nr, nc = target_r + dr * step, target_c + dc * step
            if not is_in_bounds(nr, nc):
                break
            piece = board[nr][nc]
            if piece != ".":
                if piece == rook_char or piece == queen_char:
                    return True
                break
            step += 1

    # 5. Diagonal attacks (Bishop & Queen)
    diag_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in diag_dirs:
        step = 1
        while True:
            nr, nc = target_r + dr * step, target_c + dc * step
            if not is_in_bounds(nr, nc):
                break
            piece = board[nr][nc]
            if piece != ".":
                if piece == bishop_char or piece == queen_char:
                    return True
                break
            step += 1

    return False


def is_king_in_check(board: List[List[str]], color: str) -> bool:
    king_char = "K" if color == "w" else "k"
    enemy_color = "b" if color == "w" else "w"
    kr, kc = find_king(board, king_char)
    return is_square_attacked(board, kr, kc, enemy_color)


def generate_pseudo_legal_moves(board: List[List[str]], color: str) -> List[Tuple[int, int, int, int]]:
    """Generate all candidate moves (r1, c1, r2, c2) for the player."""
    moves = []
    enemy_color = "b" if color == "w" else "w"
    pawn_dir = -1 if color == "w" else 1
    start_row = 6 if color == "w" else 1

    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece == ".":
                continue
            if (color == "w" and not piece.isupper()) or (color == "b" and not piece.islower()):
                continue

            piece_type = piece.upper()

            if piece_type == "P":
                # Forward 1
                fr = r + pawn_dir
                if is_in_bounds(fr, c) and board[fr][c] == ".":
                    moves.append((r, c, fr, c))
                    # Forward 2 from initial rank
                    if r == start_row:
                        f2r = r + 2 * pawn_dir
                        if board[f2r][c] == ".":
                            moves.append((r, c, f2r, c))
                # Captures
                for dc in (-1, 1):
                    cr, cc = r + pawn_dir, c + dc
                    if is_in_bounds(cr, cc):
                        target = board[cr][cc]
                        if target != "." and ((color == "w" and target.islower()) or (color == "b" and target.isupper())):
                            moves.append((r, c, cr, cc))

            elif piece_type == "N":
                offsets = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if is_in_bounds(nr, nc):
                        target = board[nr][nc]
                        if target == "." or (color == "w" and target.islower()) or (color == "b" and target.isupper()):
                            moves.append((r, c, nr, nc))

            elif piece_type in ("B", "R", "Q"):
                dirs = []
                if piece_type in ("R", "Q"):
                    dirs.extend([(-1, 0), (1, 0), (0, -1), (0, 1)])
                if piece_type in ("B", "Q"):
                    dirs.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
                for dr, dc in dirs:
                    step = 1
                    while True:
                        nr, nc = r + dr * step, c + dc * step
                        if not is_in_bounds(nr, nc):
                            break
                        target = board[nr][nc]
                        if target == ".":
                            moves.append((r, c, nr, nc))
                        else:
                            if (color == "w" and target.islower()) or (color == "b" and target.isupper()):
                                moves.append((r, c, nr, nc))
                            break
                        step += 1

            elif piece_type == "K":
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if is_in_bounds(nr, nc):
                            target = board[nr][nc]
                            if target == "." or (color == "w" and target.islower()) or (color == "b" and target.isupper()):
                                moves.append((r, c, nr, nc))

    return moves


def generate_legal_moves(board: List[List[str]], color: str) -> List[Tuple[int, int, int, int]]:
    """Return all valid legal moves (r1, c1, r2, c2) that leave the player's king safe."""
    legal_moves = []
    for r1, c1, r2, c2 in generate_pseudo_legal_moves(board, color):
        saved_target = board[r2][c2]
        saved_source = board[r1][c1]
        board[r2][c2] = saved_source
        board[r1][c1] = "."

        in_check = is_king_in_check(board, color)

        board[r1][c1] = saved_source
        board[r2][c2] = saved_target

        if not in_check:
            legal_moves.append((r1, c1, r2, c2))
    return legal_moves


def has_any_legal_move(board: List[List[str]], color: str) -> bool:
    """Check if the player has at least one move that leaves their king not in check."""
    return len(generate_legal_moves(board, color)) > 0


def is_stalemate(fen_input: Union[str, Dict[str, Any]]) -> bool:
    """Check if the board state is a Stalemate (Pat): not in check and no legal moves."""
    fen_data = parse_fen(fen_input) if isinstance(fen_input, str) else fen_input
    board = fen_data["board"]
    active_color = fen_data["active_color"]

    in_check = is_king_in_check(board, active_color)
    has_moves = has_any_legal_move(board, active_color)
    return (not in_check) and (not has_moves)


def evaluate_game_state(fen_input: Union[str, Dict[str, Any]], include_stalemate: bool = False) -> str:
    """
    Evaluates the chessboard state according to EPITECH specifications:
    - 'Checkmate'
    - 'Check'
    - 'Stalemate' (if include_stalemate is True)
    - 'Nothing'
    """
    fen_data = parse_fen(fen_input) if isinstance(fen_input, str) else fen_input
    board = fen_data["board"]
    active_color = fen_data["active_color"]

    in_check = is_king_in_check(board, active_color)
    has_moves = has_any_legal_move(board, active_color)

    if in_check:
        if not has_moves:
            return "Checkmate"
        return "Check"
    else:
        if include_stalemate and not has_moves:
            return "Stalemate"
        return "Nothing"


def evaluate_game_state_detailed(fen_input: Union[str, Dict[str, Any]], include_stalemate: bool = True) -> str:
    """
    Detailed evaluation including colors and optional stalemate:
    - 'Checkmate Black' / 'Checkmate White'
    - 'Check Black' / 'Check White'
    - 'Stalemate'
    - 'Nothing'
    """
    fen_data = parse_fen(fen_input) if isinstance(fen_input, str) else fen_input
    board = fen_data["board"]
    active_color = fen_data["active_color"]
    giving_color = "Black" if active_color == "w" else "White"

    in_check = is_king_in_check(board, active_color)
    has_moves = has_any_legal_move(board, active_color)

    if in_check:
        if not has_moves:
            return f"Checkmate {giving_color}"
        return f"Check {giving_color}"
    elif include_stalemate and not has_moves:
        return "Stalemate"
    return "Nothing"
