import numpy as np

from simba.similarities import avg_cosine


def test_avg_cosine():
    x = np.random.random((10, 100))
    y = np.random.random((15, 100))
    assert np.isfinite(avg_cosine(x, y))


def test_avg_cosine_ordering():
    x = np.array([[1, -1], [1, 1]])
    y1 = np.array([[5, 3], [5, -3]])
    y2 = np.array([[-5, 4], [5, 4]])

    # Cosine similarity of parallel average vectors should be greater than for
    # orthogonal average vectors.
    assert avg_cosine(x, y1) > avg_cosine(x, y2)
