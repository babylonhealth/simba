import numpy as np

from simba.similarities import (
    dynamax_dice, dynamax_otsuka, dynamax_jaccard, max_jaccard
)


def test_max_jaccard():
    x = np.array([
        [1, 0, 0],
        [2, 1, 0]
    ])
    y1 = np.array([
        [2, 1, 0],
        [5, 0, 5],
        [1, 4, 1]
    ])
    y2 = np.array([
        [0, 1, 1],
        [0, 0, 2],
        [0, 0, 3],
        [0, 0, 4]
    ])

    # Sequences that share similar embeddings should have higher scores.
    assert max_jaccard(x, y1) > max_jaccard(x, y2)


def test_dynamax_jaccard():
    x = np.array([
        [1, 0, 0],
        [2, 1, 0]
    ])
    y1 = np.array([
        [2, 1, 0],
        [5, 0, 5],
        [1, 4, 1]
    ])
    y2 = np.array([
        [0, 1, 1],
        [0, 0, 2],
        [0, 0, 3],
        [0, 0, 4]
    ])

    # Sequences that share similar embeddings should have higher scores.
    assert dynamax_jaccard(x, y1) > dynamax_jaccard(x, y2)


def test_dynamax_dice():
    x = np.array([
        [1, 0, 0],
        [2, 1, 0]
    ])
    y1 = np.array([
        [2, 1, 0],
        [5, 0, 5],
        [1, 4, 1]
    ])
    y2 = np.array([
        [0, 1, 1],
        [0, 0, 2],
        [0, 0, 3],
        [0, 0, 4]
    ])

    # Sequences that share similar embeddings should have higher scores.
    assert dynamax_dice(x, y1) > dynamax_dice(x, y2)


def test_dynamax_otsuka():
    x = np.array([
        [1, 0, 0],
        [2, 1, 0]
    ])
    y1 = np.array([
        [2, 1, 0],
        [5, 0, 5],
        [1, 4, 1]
    ])
    y2 = np.array([
        [0, 1, 1],
        [0, 0, 2],
        [0, 0, 3],
        [0, 0, 4]
    ])

    # Sequences that share similar embeddings should have higher scores.
    assert dynamax_otsuka(x, y1) > dynamax_otsuka(x, y2)


# TODO tests for fbow factory
