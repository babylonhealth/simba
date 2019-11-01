import numpy as np
import pytest

from simba.similarities.sif import _avg_pca, batch_avg_pca


def test_avg_pca():
    x = np.array([1, 1])
    y = np.array([2, 2])
    pc = np.array([1, 1])
    assert np.isfinite(_avg_pca(x, y, pc))


def test_avg_pca_orthogonal():
    x = np.array([1, 0])
    y = np.array([0, 1])
    pc = np.array([1, 0])
    with pytest.warns(RuntimeWarning):
        assert np.isnan(_avg_pca(x, y, pc))


def test_batch_avg_pca():
    xs = [
        np.random.random((3, 3))
        for _ in range(5)
    ]
    ys = [
        np.random.random((3, 3))
        for _ in range(5)
    ]

    scores = batch_avg_pca(xs, ys)
    assert len(scores) == 5
    assert np.all(np.isfinite(scores))


def test_batch_avg_pca_ordering():
    # All x_means will be [1, 0]
    xs = [
        np.array([[1, 1], [1, -1]])
        for _ in range(5)
    ]
    # All y_means will be almost orthogonal to x_means,
    # except the last which is parallel
    ys = [
        np.array([[-1, -1], [1.1, -1]])
        for _ in range(4)
    ]
    ys.append(np.array([[1, 1], [1, -1]]))

    # Hence the last score should be the highest
    scores = batch_avg_pca(xs, ys)
    assert np.all(scores[4] > scores[i] for i in range(4))


def test_batch_avg_pca_single_pair():
    # If only a single pair of sequences, similarity will always be -1
    xs = [np.random.random((20, 200)) * 10]
    ys = [np.random.random((20, 200))]
    np.testing.assert_allclose(batch_avg_pca(xs, ys), -1)
