"""Check the physical chess symmetry and separation from validation."""

import unittest

import numpy as np

from src.chess_engine.encoders import encode_board, INPUT_DIM
from src.data.augmentation import augment_color_symmetry, mirror_colors


FEN = "r3k2r/ppp2ppp/2n5/3pP3/8/2N5/PPP2PPP/R3K2R w Kq d6 0 1"
MIRRORED_FEN = "r3k2r/ppp2ppp/2n5/8/3Pp3/2N5/PPP2PPP/R3K2R b Qk d3 0 1"


class AugmentationTests(unittest.TestCase):
    def test_mirror_matches_independently_encoded_fen_and_is_an_involution(self):
        inputs = encode_board(FEN)[None, :]
        expected = encode_board(MIRRORED_FEN)[None, :]
        mirrored = mirror_colors(inputs)
        np.testing.assert_array_equal(mirrored, expected)
        np.testing.assert_array_equal(mirror_colors(mirrored), inputs)
        self.assertEqual(mirrored.dtype, np.float32)
        self.assertFalse(np.shares_memory(inputs, mirrored))

    def test_five_class_targets_swap_colors_and_three_classes_keep_states(self):
        empty_validation = np.empty((0, INPUT_DIM), dtype=np.float32)
        for num_classes, expected in ((5, [0, 2, 1, 4, 3]), (3, [0, 1, 2])):
            inputs = np.repeat(encode_board(FEN)[None, :], num_classes, axis=0)
            labels = np.arange(num_classes)
            augmented, targets, stats = augment_color_symmetry(inputs, labels, empty_validation, num_classes)
            np.testing.assert_array_equal(targets, list(range(num_classes)) + expected)
            self.assertEqual(augmented.shape, (num_classes * 2, INPUT_DIM))
            self.assertEqual(stats["mirrored_samples"], num_classes)
            self.assertEqual(stats["excluded_validation_matches"], 0)

    def test_generated_validation_matches_are_excluded_and_inputs_unchanged(self):
        inputs = np.asarray([encode_board(FEN), encode_board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")])
        labels = np.array([1, 4])
        validation = encode_board(MIRRORED_FEN)[None, :]
        original_inputs, original_labels, original_validation = inputs.copy(), labels.copy(), validation.copy()
        augmented, targets, stats = augment_color_symmetry(inputs, labels, validation, 5)
        self.assertEqual(stats, {"original_samples": 2, "mirrored_samples": 1, "excluded_validation_matches": 1})
        np.testing.assert_array_equal(augmented[:2], inputs)
        np.testing.assert_array_equal(targets, [1, 4, 3])
        self.assertFalse(np.any(np.all(augmented[2:] == validation[0], axis=1)))
        np.testing.assert_array_equal(inputs, original_inputs)
        np.testing.assert_array_equal(labels, original_labels)
        np.testing.assert_array_equal(validation, original_validation)

    def test_invalid_features_and_labels_are_rejected(self):
        inputs = encode_board(FEN)[None, :]
        validation = np.empty((0, INPUT_DIM), dtype=np.float32)
        with self.assertRaises(ValueError):
            mirror_colors(inputs[:, :-1])
        for targets, classes in (([0], 4), ([5], 5), ([-1], 5), ([[1, 0, 0]], 3), ([1.5], 3)):
            with self.subTest(targets=targets, classes=classes), self.assertRaises(ValueError):
                augment_color_symmetry(inputs, targets, validation, classes)
        invalid = inputs.copy()
        invalid[0, 0] = 0.5
        with self.assertRaises(ValueError):
            augment_color_symmetry(invalid, [0], validation, 5)


if __name__ == "__main__":
    unittest.main()
