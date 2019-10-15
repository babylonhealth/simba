import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


def avg_pearson(x, y):
    """
    Pearson correlation coefficient between two sentences
    represented as averaged word vectors
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity measure between the two sentences
    """
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    return pearsonr(x_mean, y_mean)[0]


def avg_spearman(x, y):
    """
    Spearman correlation coefficient between two sentences
    represented as averaged word vectors
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity measure between the two sentences
    """
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    return spearmanr(x_mean, y_mean)[0]


def avg_kendall(x, y):
    """
    Kendall correlation coefficient between two sentences
    represented as averaged word vectors
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity measure between the two sentences
    """
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    return kendalltau(x_mean, y_mean, method='asymptotic')[0]


def max_spearman(x, y):
    """
    Spearman correlation coefficient between two sentences
    represented as max-pooled word vectors
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity measure between the two sentences
    """
    x_max = np.max(x, axis=0)
    y_max = np.max(y, axis=0)
    return spearmanr(x_max, y_max)[0]
