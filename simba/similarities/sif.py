import numpy as np

from ..utils.linalg import compute_pc, cosine


def _avg_pca(x_mean, y_mean, pc):
    """
    Helper for the batched method.
    :param x_mean: mean word vector for first sentence
    :param y_mean: mean word vector for second sentence
    :param pc: 1st principle component of the embedding matrix
    :return: similarity measure between the two sentences
    """
    x_mean = (x_mean - x_mean.dot(pc.T) * pc).reshape(-1)
    y_mean = (y_mean - y_mean.dot(pc.T) * pc).reshape(-1)
    return cosine(x_mean, y_mean)


def batch_avg_pca(xs, ys):
    """
    Mean vector with removal of first principle component, as proposed in
    Arora et al. 2017. Note that this is an offline method due to PCA.
    :param xs: first list of sequences of embeddings
    :param ys: second list of sequences of embeddings
    :return: similarity scores for each pair of sequencies
    """
    x_means = [np.mean(x, axis=0) for x in xs]
    y_means = [np.mean(y, axis=0) for y in ys]
    embs = np.vstack((x_means, y_means))
    pc = compute_pc(embs)
    return [
        _avg_pca(x, y, pc) for x, y in zip(x_means, y_means)
    ]
