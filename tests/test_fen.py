import unittest
from src.chess_engine.fen_parser import parse_fen
from src.chess_engine.encoders import encode_board, INPUT_DIM


class TestFENAndEncoders(unittest.TestCase):
    def test_valid_starting_fen(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        res = parse_fen(fen)
        self.assertEqual(res["active_color"], "w")
        self.assertEqual(res["castling"], "KQkq")
        self.assertEqual(len(res["board"]), 8)
        self.assertEqual(len(res["board"][0]), 8)

    def test_invalid_fens(self):
        # Missing king
        with self.assertRaises(ValueError):
            parse_fen("rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

        # Wrong number of ranks
        with self.assertRaises(ValueError):
            parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1")

        # Invalid digit
        with self.assertRaises(ValueError):
            parse_fen("rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def test_encoder_dimension(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        vec = encode_board(fen)
        self.assertEqual(vec.shape, (INPUT_DIM,))
        self.assertEqual(vec.dtype.name, "float32")

    def test_parallel_encoder(self):
        from src.chess_engine.encoders import encode_boards_parallel
        fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * 60
        mat = encode_boards_parallel(fens, max_workers=4)
        self.assertEqual(mat.shape, (60, INPUT_DIM))
        self.assertEqual(mat.dtype.name, "float32")


if __name__ == "__main__":
    unittest.main()

