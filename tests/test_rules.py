import unittest
from src.chess_engine.rules import (
    evaluate_game_state,
    evaluate_game_state_detailed,
    is_stalemate,
    generate_legal_moves,
)


class TestChessRules(unittest.TestCase):
    def test_fools_mate(self):
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        self.assertEqual(evaluate_game_state(fen), "Checkmate")
        self.assertEqual(evaluate_game_state_detailed(fen), "Checkmate Black")

    def test_scholars_mate(self):
        fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        self.assertEqual(evaluate_game_state(fen), "Checkmate")
        self.assertEqual(evaluate_game_state_detailed(fen), "Checkmate White")

    def test_check_not_mate(self):
        fen = "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3"
        self.assertEqual(evaluate_game_state(fen), "Check")
        self.assertEqual(evaluate_game_state_detailed(fen), "Check White")

    def test_quiet_starting_positions(self):
        fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.assertEqual(evaluate_game_state(fen1), "Nothing")

        fen2 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"
        self.assertEqual(evaluate_game_state(fen2), "Nothing")

    def test_stalemate_pat(self):
        # King in corner blocked by queen, not in check
        stalemate_fen = "k7/8/1Q6/8/8/8/8/K7 b - - 0 1"
        self.assertTrue(is_stalemate(stalemate_fen))
        self.assertEqual(evaluate_game_state(stalemate_fen, include_stalemate=True), "Stalemate")
        self.assertEqual(evaluate_game_state(stalemate_fen, include_stalemate=False), "Nothing")
        self.assertEqual(evaluate_game_state_detailed(stalemate_fen), "Stalemate")

    def test_legal_moves_generation(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        moves = generate_legal_moves([["."]*8 for _ in range(8)], "w")
        self.assertIsInstance(moves, list)


if __name__ == "__main__":
    unittest.main()
