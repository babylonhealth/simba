import numpy as np

from simba.similarities import (
    avg_pearson, avg_spearman, avg_kendall, max_spearman
)


def test_avg_pearson():
    # Generate correlated and uncorrelated univariate samples.
    x = []
    y1 = []
    y2 = []
    for _ in range(5):
        samples = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=np.ones((2, 2)),
            size=10
        )
        x.append(samples[:, 0])
        y1.append(samples[:, 1])
        y2.append(np.random.normal(size=10))

    # Score for correlated vectors should be higher.
    assert avg_pearson(x, y1) > avg_pearson(x, y2)


def test_avg_spearman():
    # Generate correlated and uncorrelated univariate samples.
    x = []
    y1 = []
    y2 = []
    for _ in range(5):
        samples = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=np.ones((2, 2)),
            size=10
        )
        x.append(samples[:, 0])
        y1.append(samples[:, 1])
        y2.append(np.random.normal(size=10))

    # Score for correlated vectors should be higher.
    assert avg_spearman(x, y1) > avg_spearman(x, y2)


def test_avg_kendall():
    # Generate correlated and uncorrelated univariate samples.
    x = []
    y1 = []
    y2 = []
    for _ in range(5):
        samples = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=np.ones((2, 2)),
            size=10
        )
        x.append(samples[:, 0])
        y1.append(samples[:, 1])
        y2.append(np.random.normal(size=10))

    # Score for correlated vectors should be higher.
    assert avg_kendall(x, y1) > avg_kendall(x, y2)


def test_max_spearman():
    # Generate correlated and uncorrelated univariate samples.
    x = []
    y1 = []
    y2 = []
    for _ in range(5):
        samples = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=np.ones((2, 2)),
            size=10
        )
        x.append(samples[:, 0])
        y1.append(samples[:, 1])
        y2.append(np.random.normal(size=10))

    # Score for correlated vectors should be higher.
    assert max_spearman(x, y1) > max_spearman(x, y2)
