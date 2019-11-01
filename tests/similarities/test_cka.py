import numpy as np

from simba.similarities import dcorr, cka_linear, cka_gaussian


def test_dcorr():
    # Generate correlated and uncorrelated multivariate samples.
    samples = np.random.multivariate_normal(
        mean=np.zeros(7),
        cov=np.ones((7, 7)),
        size=10
    )
    x = samples[:, :5].T
    y1 = samples[:, 5:].T
    y2 = np.random.multivariate_normal(
        mean=np.zeros(6),
        cov=np.ones((6, 6)),
        size=10
    ).T

    # Score for correlated vectors should be higher.
    assert dcorr(x, y1) > dcorr(x, y2)


def test_cka_linear():
    # Generate correlated and uncorrelated multivariate samples.
    samples = np.random.multivariate_normal(
        mean=np.zeros(7),
        cov=np.ones((7, 7)),
        size=10
    )
    x = samples[:, :5].T
    y1 = samples[:, 5:].T
    y2 = np.random.multivariate_normal(
        mean=np.zeros(6),
        cov=np.ones((6, 6)),
        size=10
    ).T

    # Score for correlated vectors should be higher.
    assert cka_linear(x, y1) > cka_linear(x, y2)


def test_cka_gaussian():
    # Generate correlated and uncorrelated multivariate samples.
    samples = np.random.multivariate_normal(
        mean=np.zeros(7),
        cov=np.ones((7, 7)),
        size=10
    )
    x = samples[:, :5].T
    y1 = samples[:, 5:].T
    y2 = np.random.multivariate_normal(
        mean=np.zeros(6),
        cov=np.ones((6, 6)),
        size=10
    ).T

    # Score for correlated vectors should be higher.
    assert cka_gaussian(x, y1) > cka_gaussian(x, y2)
