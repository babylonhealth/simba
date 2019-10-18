import numpy as np

from ..utils.linalg import cosine


def avg_cosine(x, y):
    """
    Cosine similarity between two avg. word vectors.
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between two sentences
    """
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    return cosine(x_mean, y_mean)
