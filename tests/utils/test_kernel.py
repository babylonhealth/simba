import numpy as np
import pytest

from simba.utils.kernel import gaussian_kernel


good_xs = [
    np.array([[10, 20, 30], [50, 25, 0]]),
    np.array([[1, 1, 2], [2, 2, 4], [3, 3, 6]]),
    np.random.random((120, 100)),
]

bad_xs = [
    np.array([[1, 1], [2, 2]]),
    np.zeros((4, 2)),
    np.array([[1], [2], [3], [5]]),
]


@pytest.mark.parametrize('X', good_xs)
def test_gaussian_kernel(X):
    # Result should be square and symmetric with 1's on diagonal
    result = gaussian_kernel(X)
    d = X.shape[1]
    assert result.shape == (d, d)
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(np.diag(result), 1.)


@pytest.mark.parametrize('X', good_xs)
def test_gaussian_kernel_with_sigma(X):
    # Smaller sigma means off-diagonal entries should be smaller
    result_with_small_sigma = gaussian_kernel(X, sigma=0.1)
    result_with_large_sigma = gaussian_kernel(X, sigma=10.)
    assert np.all(
        np.logical_or(
            result_with_small_sigma < result_with_large_sigma,
            np.isclose(result_with_small_sigma, result_with_large_sigma)
        )
    )


def test_gaussian_kernel_zero_sigma():
    X = np.random.random((3, 4))
    with pytest.warns(RuntimeWarning):
        result = gaussian_kernel(X, sigma=0)
        assert np.any(np.isnan(result))


@pytest.mark.parametrize('X', bad_xs)
def test_gaussian_kernel_degenerate_matrix(X):
    # Computing sigma when all columns of X are the same is problematic
    with pytest.warns(RuntimeWarning):
        result = gaussian_kernel(X)
        assert np.all(np.isnan(result))


@pytest.mark.parametrize('X', bad_xs)
def test_gaussian_kernel_degenerate_matrix_with_sigma(X):
    # When sigma is provided, all columns of X being identical is OK
    result = gaussian_kernel(X, sigma=5.)
    np.testing.assert_allclose(result, 1.)
