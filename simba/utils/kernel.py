import numpy as np


def linear_kernel(X):
    """
    Computes a linear kernel for X
    :param X: word embedding matrix X with shape (k x D)
    :return: linear kernel for X
    """
    return np.dot(X.T, X)


def gaussian_kernel(X, sigma=None):
    """
    Computes a Gaussian kernel for X
    :param X: word embedding matrix X with shape (k x D)
    :param sigma: standard deviation of the kernel
    :return: Gaussian kernel for X
    """
    G = linear_kernel(X)
    SN = np.diag(G)
    SD = -2 * G + SN[:, np.newaxis] + SN[np.newaxis, :]
    if sigma is None:
        sigma = np.sqrt(np.median(SD[SD != 0]))
    K = np.exp(-SD / (2 * sigma ** 2))
    return K


def centering_matrix(d):
    """
    Returns a centering matrix of dimension d
    :param d: dimension of the matrix
    :return: centering matrix of dimension d
    """
    return np.eye(d) - np.ones((d, d)) / d
