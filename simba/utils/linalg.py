import numpy as np
from sklearn.decomposition import TruncatedSVD


def cosine(x, y):
    """Cosine similarity between x and y."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    Function taken from https://github.com/PrincetonML/SIF
    Copyright (c) 2017 PrincetonML
    This function is licensed under the MIT license found at
    https://github.com/PrincetonML/SIF/blob/master/LICENSE.
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def normalise_rows(X):
    """Normalise the rows of the matrix X, using L2 norm."""
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]
