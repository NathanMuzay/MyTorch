"""Validate geometry independently of labels and preserve physical symmetry."""

import unittest

import numpy as np

from src.chess_engine.encoders import (encode_board, encode_boards_parallel, model_encoding,
                                       INPUT_DIM, TACTICAL_INPUT_DIM, EN_PASSANT, ATTACK_MAPS,
                                       KING_RAYS, KING_CONTEXT, DIRECTIONS, PIECE_TO_INDEX)
from src.chess_engine.fen_parser import parse_fen
from src.chess_engine.rules import is_square_attacked
from src.data.augmentation import mirror_colors, augment_color_symmetry
from src.neural_network.layers import Dense
from src.neural_network.model import Sequential


FEN = "r3k2r/ppp2ppp/2n5/3pP3/8/2N5/PPP2PPP/R3K2R w Kq d6 0 1"
MIRRORED_FEN = "r3k2r/ppp2ppp/2n5/8/3Pp3/2N5/PPP2PPP/R3K2R b Qk d3 0 1"


class TacticalEncodingTests(unittest.TestCase):
    def test_layout_preserves_basic_features_and_en_passant(self):
        parsed = parse_fen(FEN)
        original = [row[:] for row in parsed["board"]]
        vector = encode_board(parsed, encoding="tactical")
        self.assertEqual(vector.shape, (TACTICAL_INPUT_DIM,))
        self.assertEqual(TACTICAL_INPUT_DIM, 1333)
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(np.isin(vector, [0, 1]).all())
        np.testing.assert_array_equal(vector[:INPUT_DIM], encode_board(FEN))
        self.assertEqual(vector[EN_PASSANT].sum(), 1)
        self.assertEqual(vector[EN_PASSANT].reshape(8, 8)[2, 3], 1)
        self.assertEqual(parsed["board"], original)
        without_ep = encode_board(FEN.replace("d6", "-"), encoding="tactical")
        changed = np.flatnonzero(vector != without_ep)
        np.testing.assert_array_equal(changed, [EN_PASSANT.start + 2 * 8 + 3])
        np.testing.assert_array_equal(vector, encode_board(FEN.replace("0 1", "42 99"), encoding="tactical"))
        for square in ("a4", "i6", "d9", "d66"):
            with self.subTest(square=square), self.assertRaisesRegex(ValueError, "en passant"):
                encode_board(FEN.replace("d6", square), encoding="tactical")

    def test_attack_maps_match_independent_square_attack_queries(self):
        fens = [FEN, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "4k3/2n5/3b4/3p4/3K1R2/3P4/8/4Q3 w - - 0 1"]
        for fen in fens:
            board = parse_fen(fen)["board"]
            vector = encode_board(fen, encoding="tactical")
            attacks = vector[ATTACK_MAPS].reshape(8, 8, 2)
            expected = [[[is_square_attacked(board, r, c, color) for color in ("w", "b")]
                         for c in range(8)] for r in range(8)]
            np.testing.assert_array_equal(attacks, expected)

    def test_first_blocker_distance_and_king_vacancy_are_encoded(self):
        fen = "k7/8/8/3p4/r2K1R2/8/8/8 w - - 0 1"
        vector = encode_board(fen, encoding="tactical")
        rays = vector[KING_RAYS].reshape(2, 8, 19)
        up = DIRECTIONS.index((-1, 0))
        right = DIRECTIONS.index((0, 1))
        self.assertEqual(rays[0, up, PIECE_TO_INDEX["p"]], 1)
        self.assertEqual(rays[0, up, 12], 1)  # one square away
        self.assertEqual(rays[0, right, PIECE_TO_INDEX["R"]], 1)
        self.assertEqual(rays[0, right, 13], 1)  # two squares away
        context = vector[KING_CONTEXT].reshape(2, 8, 4)
        # e4 is attacked by a4's rook once the king vacates d4.
        np.testing.assert_array_equal(context[0, right], [1, 0, 0, 1])
        np.testing.assert_array_equal(context[0, up], [1, 0, 1, 0])
        # The black king's off-board neighbors have four zero flags.
        np.testing.assert_array_equal(context[1, up], [0, 0, 0, 0])

    def test_tactical_mirror_matches_fen_and_rejects_validation_overlap(self):
        inputs = encode_board(FEN, encoding="tactical")[None]
        expected = encode_board(MIRRORED_FEN, encoding="tactical")[None]
        np.testing.assert_array_equal(mirror_colors(inputs), expected)
        np.testing.assert_array_equal(mirror_colors(mirror_colors(inputs)), inputs)
        # Exclusion also applies to the same board with a different EP field.
        validation = encode_board(MIRRORED_FEN.replace("d3", "-"), encoding="tactical")[None]
        augmented, labels, stats = augment_color_symmetry(inputs, np.array([1]), validation, 5)
        np.testing.assert_array_equal(augmented, inputs)
        np.testing.assert_array_equal(labels, [1])
        self.assertEqual(stats["excluded_validation_matches"], 1)

    def test_mirror_matches_independent_random_board_encodings(self):
        generator = np.random.default_rng(42)
        for _ in range(20):
            board = np.full((8, 8), ".", dtype="<U1")
            positions = generator.choice(64, size=18, replace=False)
            pieces = ["K", "k"] + generator.choice(list("PNBRQpnbrq"), size=16).tolist()
            board.ravel()[positions] = pieces
            data = {"board": board.tolist(), "active_color": "w", "castling": "Kq", "en_passant": "c6"}
            mirrored_board = [[p.swapcase() for p in row] for row in data["board"][::-1]]
            expected_data = {"board": mirrored_board, "active_color": "b", "castling": "Qk", "en_passant": "c3"}
            actual = mirror_colors(encode_board(data, encoding="tactical")[None])[0]
            np.testing.assert_array_equal(actual, encode_board(expected_data, encoding="tactical"))

    def test_batch_empty_and_model_schema_validation(self):
        for encoding, dimension in (("basic", INPUT_DIM), ("tactical", TACTICAL_INPUT_DIM)):
            empty = encode_boards_parallel([], encoding=encoding)
            self.assertEqual(empty.shape, (0, dimension))
            self.assertEqual(mirror_colors(empty).shape, empty.shape)
            batch = encode_boards_parallel([FEN] * 50, max_workers=2, encoding=encoding)
            np.testing.assert_array_equal(batch, np.repeat(encode_board(FEN, encoding=encoding)[None], 50, axis=0))
        old = Sequential([Dense(INPUT_DIM, 3)])
        self.assertEqual(model_encoding(old), "basic")
        old.metadata["encoding"] = "tactical"
        with self.assertRaisesRegex(ValueError, "expects"):
            model_encoding(old)
        old.metadata.update(encoding="basic", encoding_version=2)
        with self.assertRaisesRegex(ValueError, "version"):
            model_encoding(old)
        with self.assertRaisesRegex(ValueError, "Unknown chess encoding"):
            encode_boards_parallel([], encoding="unknown")


if __name__ == "__main__":
    unittest.main()
