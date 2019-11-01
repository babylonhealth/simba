import numpy as np
import pytest

from simba.utils.gaussian import aic, aic_spherical, tic, tic_spherical
from tests.conftest import INPUTS_


xs = [
    np.random.random((120, 100)),
    np.zeros((3, 3)),
    np.array([[1, 2, 3, 4, 5]]),
]


aic_outputs_ = [
    24.00419702, 16.00279802, -178.69176477, -119.12784318,
    -119.12784318, -178.69176477, 119.25944763
]

tic_outputs_ = [
    19.50419554, 13.00279702, -193.69176477, -129.12784318,
    -129.12784318, -193.69176477, 111.74772509
]

aic_spherical_outputs_ = [
    20.00419702, 14.00279801, -99.798701423, -65.865800949,
    -65.865800949, -99.798701423, 111.89780686
]

tic_spherical_outputs_ = [
    18.00417002, 12.16944618, -116.79870142, -75.865800949,
    -75.865800949, -116.79870142, 111.11225760
]


@pytest.mark.parametrize('X', xs)
def test_aic(X):
    assert np.isfinite(aic(X))


@pytest.mark.parametrize('X,Y', zip(INPUTS_, aic_outputs_))
def test_aic_num(X, Y):
    assert np.isclose(aic(X), Y, atol=1e-8)


@pytest.mark.parametrize('X', xs)
def test_tic(X):
    assert np.isfinite(tic(X))


@pytest.mark.parametrize('X,Y', zip(INPUTS_, tic_outputs_))
def test_tic_num(X, Y):
    assert np.isclose(tic(X), Y, atol=1e-8)


def test_aic_different_distributions():
    # Gaussian AIC should be higher for multivariate normal
    # than another distribution
    cov = np.random.random((3, 3))
    cov = np.dot(cov, cov.T)
    X = np.random.multivariate_normal(mean=np.zeros(3), cov=cov, size=4)
    Y = np.random.dirichlet(alpha=np.random.random(3), size=4)
    assert aic(X) > aic(Y)


def test_tic_different_distributions():
    # Gaussian TIC should be higher for multivariate normal
    # than another distribution
    cov = np.random.random((3, 3))
    cov = np.dot(cov, cov.T)
    X = np.random.multivariate_normal(mean=np.zeros(3), cov=cov, size=4)
    Y = np.random.dirichlet(alpha=np.random.random(3), size=4)
    assert tic(X) > tic(Y)


@pytest.mark.parametrize('X', xs)
def test_aic_spherical(X):
    assert np.isfinite(aic_spherical(X))


@pytest.mark.parametrize('X,Y', zip(INPUTS_, aic_spherical_outputs_))
def test_aic_spherical_num(X, Y):
    assert np.isclose(aic_spherical(X), Y, atol=1e-8)


@pytest.mark.parametrize('X', xs)
def test_tic_spherical(X):
    assert np.isfinite(tic_spherical(X))


@pytest.mark.parametrize('X,Y', zip(INPUTS_, tic_spherical_outputs_))
def test_tic_spherical_num(X, Y):
    assert np.isclose(tic_spherical(X), Y, atol=1e-8)


def test_aic_spherical_different_distributions():
    # Spherical gaussian AIC should be smaller for non-spherical gaussians
    cov = np.random.random((3, 3))
    cov = np.dot(cov, cov.T)
    X = np.random.multivariate_normal(mean=np.zeros(3), cov=np.eye(3), size=4)
    Y = np.random.multivariate_normal(mean=np.zeros(3), cov=cov, size=4)
    assert aic_spherical(X) > aic_spherical(Y)


def test_tic_spherical_different_distributions():
    # Spherical gaussian TIC should be smaller for non-spherical gaussians
    cov = np.random.random((3, 3))
    cov = np.dot(cov, cov.T)
    X = np.random.multivariate_normal(mean=np.zeros(3), cov=np.eye(3), size=4)
    Y = np.random.multivariate_normal(mean=np.zeros(3), cov=cov, size=4)
    assert tic_spherical(X) > tic_spherical(Y)
