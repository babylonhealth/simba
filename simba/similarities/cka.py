import numpy as np
import dcor

from simba.utils.kernel import centering_matrix, linear_kernel, gaussian_kernel


def cka_factory(kernel=None):
    """
    Builds a Centered Kernel Alignment (CKA) similarity function
    with the specified kernel
    :param kernel: kernel function for CKA
    :return: CKA similarity function
    """
    def hsic(X, Y):
        """
        Computes Hilbert-Schmidt independence criterion (HSIC)
        between word embedding matrices X and Y
        :param X: word embedding matrix X with shape (k x D)
        :param Y: word embedding matrix Y with shape (l x D)
        :return: HSIC (unnormalised) between X and Y
        """
        X = np.array(X)
        Y = np.array(Y)
        assert X.shape[1] == Y.shape[1]
        d = X.shape[1]
        H = centering_matrix(d)
        KX = kernel(X)
        KY = kernel(Y)
        return np.trace(KX @ H @ KY @ H)

    def cka(X, Y):
        """
        Computes Centered Kernel Alignment (CKA)
        between word embedding matrices X and Y
        :param X: word embedding matrix X with shape (k x D)
        :param Y: word embedding matrix Y with shape (l x D)
        :return: CKA between X and Y
        """
        return hsic(X, Y) / np.sqrt(hsic(X, X) * hsic(Y, Y))

    return cka


def cka_linear(X, Y):
    return cka_factory(linear_kernel)(X, Y)


def cka_gaussian(X, Y):
    return cka_factory(gaussian_kernel)(X, Y)


def dcorr(X, Y):
    """
    Computes Distance Correlation (dCorr)
    between word embedding matrices X and Y
    :param x: X: word embedding matrix X with shape (k x D)
    :param y: Y: word embedding matrix Y with shape (l x D)
    :return: distance correlation between X and Y
    """
    X = np.array(X)
    Y = np.array(Y)
    return dcor.distance_correlation(X.T, Y.T)
