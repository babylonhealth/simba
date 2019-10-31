import numpy as np

from simba.similarities import (
    gaussian_correction_aic, gaussian_correction_tic,
    spherical_gaussian_correction_aic, spherical_gaussian_correction_tic,
    von_mises_correction_aic, von_mises_correction_tic,
)
from simba.utils.linalg import normalise_rows


def test_gaussian_correction_aic():
    cov1 = np.array([[-1, 2, -2], [2, 3, 1], [-3, 1, 1]])
    cov1 = np.dot(cov1, cov1.T)
    x = np.random.multivariate_normal(mean=np.zeros(3), cov=cov1, size=5)
    y1 = np.random.multivariate_normal(mean=np.zeros(3), cov=cov1, size=3)

    cov2 = np.eye(3)
    y2 = np.random.multivariate_normal(mean=np.ones(3), cov=cov2, size=4)

    # Score should be higher for vectors sampled from same distribution.
    assert gaussian_correction_aic(x, y1) > gaussian_correction_aic(x, y2)


def test_gaussian_correction_tic():
    cov1 = np.array([[-1, 2, -2], [2, 3, 1], [-3, 1, 1]])
    cov1 = np.dot(cov1, cov1.T)
    x = np.random.multivariate_normal(mean=np.zeros(3), cov=cov1, size=5)
    y1 = np.random.multivariate_normal(mean=np.zeros(3), cov=cov1, size=3)

    cov2 = np.eye(3)
    y2 = np.random.multivariate_normal(mean=np.ones(3), cov=cov2, size=4)

    # Score should be higher for vectors sampled from same distribution.
    assert gaussian_correction_tic(x, y1) > gaussian_correction_tic(x, y2)


def test_spherical_gaussian_correction_aic():
    cov1 = np.array([[-1, 2, -2], [2, 3, 1], [-3, 1, 1]])
    cov1 = np.dot(cov1, cov1.T)
    x = np.random.multivariate_normal(mean=np.zeros(3), cov=cov1, size=5)
    y1 = np.random.multivariate_normal(mean=np.zeros(3), cov=cov1, size=3)

    cov2 = np.eye(3)
    y2 = np.random.multivariate_normal(mean=np.ones(3), cov=cov2, size=4)

    # Score should be higher for vectors sampled from same distribution.
    assert (
        spherical_gaussian_correction_aic(x, y1)
        > spherical_gaussian_correction_aic(x, y2)
    )


def test_spherical_gaussian_correction_tic():
    cov1 = np.array([[-1, 2, -2], [2, 3, 1], [-3, 1, 1]])
    cov1 = np.dot(cov1, cov1.T)
    x = np.random.multivariate_normal(mean=np.zeros(3), cov=cov1, size=5)
    y1 = np.random.multivariate_normal(mean=np.zeros(3), cov=cov1, size=3)

    cov2 = np.eye(3)
    y2 = np.random.multivariate_normal(mean=np.ones(3), cov=cov2, size=4)

    # Score should be higher for vectors sampled from same distribution.
    assert (
        spherical_gaussian_correction_tic(x, y1)
        > spherical_gaussian_correction_tic(x, y2)
    )


def test_von_mises_correction_aic():
    cov1 = np.array([[-1, 2, -2], [2, 3, 1], [-3, 1, 1]])
    cov1 = np.dot(cov1, cov1.T)
    x = np.random.multivariate_normal(mean=np.zeros(3), cov=cov1, size=5)
    y1 = np.random.multivariate_normal(mean=np.zeros(3), cov=cov1, size=3)

    cov2 = np.eye(3)
    y2 = np.random.multivariate_normal(mean=np.ones(3), cov=cov2, size=4)

    # VMF needs normalised rows
    x = normalise_rows(x)
    y1 = normalise_rows(y1)
    y2 = normalise_rows(y2)

    # Score should be higher for vectors sampled from same distribution.
    assert von_mises_correction_aic(x, y1) > von_mises_correction_aic(x, y2)


def test_von_mises_correction_tic():
    cov1 = np.array([[-1, 2, -2], [2, 3, 1], [-3, 1, 1]])
    cov1 = np.dot(cov1, cov1.T)
    x = np.random.multivariate_normal(mean=np.zeros(3), cov=cov1, size=5)
    y1 = np.random.multivariate_normal(mean=np.zeros(3), cov=cov1, size=3)

    cov2 = np.eye(3)
    y2 = np.random.multivariate_normal(mean=np.ones(3), cov=cov2, size=4)

    # VMF needs normalised rows
    x = normalise_rows(x)
    y1 = normalise_rows(y1)
    y2 = normalise_rows(y2)

    # Score should be higher for vectors sampled from same distribution.
    assert von_mises_correction_tic(x, y1) > von_mises_correction_tic(x, y2)
