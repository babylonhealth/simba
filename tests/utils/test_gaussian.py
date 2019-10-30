import numpy as np
import pytest
from simba.utils.gaussian import aic, tic, aic_spherical, tic_spherical


xs = [
    np.random.random((120, 100)),
    np.zeros((3, 3)),
    np.array([[1, 2, 3, 4, 5]]),
]


@pytest.mark.parametrize('X', xs)
def test_aic(X):
    assert np.isfinite(aic(X))


@pytest.mark.parametrize('X', xs)
def test_tic(X):
    assert np.isfinite(tic(X))


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


@pytest.mark.parametrize('X', xs)
def test_tic_spherical(X):
    assert np.isfinite(tic_spherical(X))


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
