import numpy as np
import pytest

from simba.utils.linalg import cosine, compute_pc


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
