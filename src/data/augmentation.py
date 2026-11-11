"""Color symmetry for encoded chessboards, applied only after the data split."""

import numpy as np

from src.chess_engine.encoders import (INPUT_DIM, TACTICAL_INPUT_DIM, EN_PASSANT,
                                       ATTACK_MAPS, KING_RAYS, KING_CONTEXT, DIRECTIONS)


def mirror_colors(inputs):
    """Flip ranks and exchange colors, including optional tactical features."""
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[1] not in (INPUT_DIM, TACTICAL_INPUT_DIM):
        raise ValueError("Color symmetry requires basic or tactical chess features")
    mirrored = np.empty_like(inputs)
    boards = inputs[:, :768].reshape(-1, 8, 8, 12)[:, ::-1]
    mirrored[:, :768] = np.concatenate((boards[..., 6:], boards[..., :6]), axis=-1).reshape(-1, 768)
    mirrored[:, 768] = 1.0 - inputs[:, 768]
    mirrored[:, 769:773] = inputs[:, [771, 772, 769, 770]]
    if inputs.shape[1] == TACTICAL_INPUT_DIM:
        mirrored[:, EN_PASSANT] = inputs[:, EN_PASSANT].reshape(-1, 8, 8)[:, ::-1].reshape(-1, 64)
        mirrored[:, ATTACK_MAPS] = inputs[:, ATTACK_MAPS].reshape(-1, 8, 8, 2)[:, ::-1, :, ::-1].reshape(-1, 128)
        direction_map = [DIRECTIONS.index((-dr, dc)) for dr, dc in DIRECTIONS]
        rays = inputs[:, KING_RAYS].reshape(-1, 2, 8, 19)[:, ::-1]
        rays = np.take(rays, direction_map, axis=2)
        rays = np.take(rays, list(range(6, 12)) + list(range(6)) + list(range(12, 19)), axis=3)
        mirrored[:, KING_RAYS] = rays.reshape(-1, KING_RAYS.stop - KING_RAYS.start)
        context = inputs[:, KING_CONTEXT].reshape(-1, 2, 8, 4)[:, ::-1]
        mirrored[:, KING_CONTEXT] = np.take(context, direction_map, axis=2).reshape(-1, 64)
    return mirrored


def augment_color_symmetry(inputs, labels, validation_inputs, num_classes):
    """Append mirrored training examples except exact matches in validation.

    Canonical labels: Nothing, Check White/Black, Checkmate White/Black;
    three-class models keep their state labels unchanged.
    """
    inputs = np.asarray(inputs, dtype=np.float32)
    validation_inputs = np.asarray(validation_inputs, dtype=np.float32)
    labels = np.asarray(labels)
    if num_classes not in (3, 5):
        raise ValueError("Color symmetry supports 3 or 5 classes")
    mirrored = mirror_colors(inputs)
    if validation_inputs.ndim != 2 or validation_inputs.shape[1] != inputs.shape[1]:
        raise ValueError("Invalid validation features for color symmetry")
    if labels.shape != (len(inputs),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Color symmetry requires one integer label per board")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Unknown class index for color symmetry")
    # Pack binary feature planes to keep the validation membership set small.
    for features in (inputs, validation_inputs):
        if not np.all((features == 0) | (features == 1)):
            raise ValueError("Color symmetry expects binary chessboard features")
    # Use the same conservative board identity in both encodings. En passant
    # differences do not allow a mirrored validation board back into training.
    validation_keys = {row.tobytes() for row in np.packbits(validation_inputs[:, :INPUT_DIM].astype(np.uint8), axis=1)}
    mirrored_keys = np.packbits(mirrored[:, :INPUT_DIM].astype(np.uint8), axis=1)
    keep = np.array([row.tobytes() not in validation_keys for row in mirrored_keys], dtype=bool)
    permutation = np.array([0, 2, 1, 4, 3] if num_classes == 5 else [0, 1, 2], dtype=labels.dtype)
    selected = np.flatnonzero(keep)
    augmented_inputs = np.empty((len(inputs) + len(selected), inputs.shape[1]), dtype=inputs.dtype)
    augmented_inputs[:len(inputs)] = inputs
    # Avoid allocating a second full filtered copy of the mirrored matrix.
    for start in range(0, len(selected), 4096):
        block = selected[start:start + 4096]
        augmented_inputs[len(inputs) + start:len(inputs) + start + len(block)] = mirrored[block]
    augmented_labels = np.concatenate((labels, permutation[labels[keep]]))
    stats = {"original_samples": len(inputs), "mirrored_samples": int(keep.sum()),
             "excluded_validation_matches": int((~keep).sum())}
    return augmented_inputs, augmented_labels, stats
