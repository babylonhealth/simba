import numpy as np
import pytest

from simba.utils.linalg import normalise_rows
from simba.utils.vmf import vmf_aic, vmf_tic


# TODO why are these okay?
good_xs = [
    np.array([[1, 2, 3, 4, 5]]),
    np.array([[1, 2, 3], [4, 5, 6]]),
]

bad_xs = [
    np.zeros((3, 3)),
    np.ones((3, 100)),
    np.ones((5, 100)) / np.sqrt(100),
]


@pytest.mark.parametrize('X', good_xs)
def test_vmf_aic(X):
    assert np.isfinite(vmf_aic(X))


@pytest.mark.parametrize('X', good_xs)
def test_vmf_tic(X):
    assert np.isfinite(vmf_tic(X))


@pytest.mark.parametrize('X', bad_xs)
def test_vmf_aic_zero_matrix(X):
    with pytest.warns(RuntimeWarning):
        assert np.isnan(vmf_aic(X))


def test_vmf_aic_unnormalised_matrix():
    # VMF AIC is nan for unnormalised rows
    X = np.array([[0.9, 0.2], [0.3, 2], [1, 1]])
    with pytest.warns(RuntimeWarning):
        assert np.isnan(vmf_aic(X))

    # Normalising fixes this
    norm_X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
    assert np.isfinite(vmf_aic(norm_X))


def test_vmf_tic_unnormalised_matrix():
    # VMF TIC is nan for unnormalised rows
    X = np.array([[0.9, 0.2], [0.3, 2], [1, 1]])
    with pytest.warns(RuntimeWarning):
        assert np.isnan(vmf_tic(X))

    # Normalising fixes this
    norm_X = normalise_rows(X)
    assert np.isfinite(vmf_tic(norm_X))


def test_vmf_aic_different_distributions():
    # VMF AIC should be higher for VMF than another distribution
    X = np.random.vonmises(mu=np.zeros(3), kappa=np.ones(3), size=(4, 3))
    X = normalise_rows(X)
    Y = np.random.dirichlet(alpha=np.random.random(3), size=4)
    Y = normalise_rows(Y)
    assert vmf_aic(X) > vmf_aic(Y)


def test_vmf_tic_different_distributions():
    # VMF TIC should be higher for VMF than another distribution
    X = np.random.vonmises(mu=np.zeros(3), kappa=np.ones(3), size=(4, 3))
    X = normalise_rows(X)
    Y = np.random.dirichlet(alpha=np.random.random(3), size=4)
    Y = normalise_rows(Y)
    assert vmf_tic(X) > vmf_tic(Y)


# TODO test for log_VMF_gradient, log_likelihood?
