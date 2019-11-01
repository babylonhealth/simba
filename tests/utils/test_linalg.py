import numpy as np
import pytest

from simba.utils.linalg import cosine, compute_pc, normalise_rows


@pytest.mark.parametrize(
    'x, y, expected',
    [
        ([1, 2, 3], [2, 4, 6], 1),
        ([1, 1, 1, 1], [-1, -1, -1, -1], -1),
        ([1, 1], [1, -1], 0),
        ([1, 1], [1, 0], np.cos(np.pi / 4)),
    ]
)
def test_cosine(x, y, expected):
    result = cosine(x, y)
    np.testing.assert_allclose(result, expected)


def test_cosine_zero_vector():
    x = np.random.random(3)
    y = np.zeros_like(x)
    with pytest.warns(RuntimeWarning):
        result = cosine(x, y)
        assert np.isnan(result)


def test_compute_pc():
    X = np.random.random((4, 5))
    result = compute_pc(X, npc=2)
    assert result.shape == (2, 5)


def test_compute_pc_too_many_components():
    X = np.random.random((4, 5))
    with pytest.raises(ValueError):
        compute_pc(X, npc=11)


def test_normalise_rows():
    X = np.random.random((4, 5))
    result = normalise_rows(X)
    for row in result:
        np.testing.assert_allclose(np.linalg.norm(row), 1)


def test_normalise_rows_zero_matrix():
    X = np.random.random((4, 5))
    X[0, :] = np.zeros(5)
    with pytest.warns(RuntimeWarning):
        result = normalise_rows(X)
        assert np.all(np.isnan(result[0]))
