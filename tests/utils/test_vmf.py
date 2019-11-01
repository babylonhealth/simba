import numpy as np
import pytest

from simba.utils.linalg import normalise_rows
from simba.utils.vmf import to_cartesian, vmf_aic, vmf_tic
from tests.conftest import INPUTS_


good_xs = [
    np.array([[1, 0], [1 / np.sqrt(2), 1 / np.sqrt(2)]]),
    np.array([
        np.divide([1, 2, 3, 4, 5], np.sqrt(55)),
        np.divide([6, 7, 8, 9, 0], np.sqrt(230))
    ])
]

bad_xs = [
    np.array([[1, 2, 3, 4, 5]]) / np.sqrt(55),
    np.zeros((3, 3)),
    np.ones((3, 100)),
    np.ones((5, 100)) / np.sqrt(100),
]

cart_outputs_ = [
    "E", np.array([0.54030231, 1.]),
    "E", np.array([1., 1.]),
    np.array([1.660455, 1.660455]), "E"
]

vmf_aic_outputs_ = [
    3.26136446, 4.58170763,
    "NaN", "NaN",
    "NaN", 6.64525188
]

vmf_tic_outputs_ = [
    0.29796992, 0.38251421,
    "NaN", "NaN",
    "NaN", 8.32262594
]


@pytest.mark.parametrize('X,Y', zip(INPUTS_, cart_outputs_))
def test_to_cartesian(X, Y):
    if str(Y) == "E":
        with pytest.raises(ValueError):
            to_cartesian(X)
    else:
        assert np.allclose(to_cartesian(X), Y, atol=1e-8)


@pytest.mark.parametrize('X', good_xs)
def test_vmf_aic(X):
    assert np.isfinite(vmf_aic(X))


@pytest.mark.parametrize('X,Y', zip(INPUTS_, vmf_aic_outputs_))
def test_vmf_aic_num(X, Y):
    if str(Y) == "NaN":
        assert np.isnan(vmf_aic(X))
    else:
        assert np.allclose(vmf_aic(X), Y, atol=1e-8)


@pytest.mark.parametrize('X', good_xs)
def test_vmf_tic(X):
    assert np.isfinite(vmf_tic(X))


@pytest.mark.parametrize('X,Y', zip(INPUTS_, vmf_tic_outputs_))
def test_vmf_tic_num(X, Y):
    if str(Y) == "NaN":
        assert np.isnan(vmf_tic(X))
    else:
        assert np.allclose(vmf_tic(X), Y, atol=1e-8)


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
