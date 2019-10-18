import numpy as np

from ..utils.linalg import compute_pc, cosine


def _sif(x_mean, y_mean, pc):
    """

    :param x_mean: mean word vector for first sentence
    :param y_mean: mean word vector for second sentence
    :param pc: 1st principle component of the embedding matrix
    :return: similarity measure between the two sentences
    """
    x_mean = (x_mean - x_mean.dot(pc.T) * pc).reshape(-1)
    y_mean = (y_mean - y_mean.dot(pc.T) * pc).reshape(-1)
    return cosine(x_mean, y_mean)


def batch_avg_sif(xs, ys):
    """

    :param xs:
    :param ys:
    :return:
    """
    x_means = [np.mean(x, axis=0) for x in xs]
    y_means = [np.mean(y, axis=0) for y in ys]
    embs = np.vstack((x_means, y_means))
    pc = compute_pc(embs)
    return [
        _sif(x, y, pc) for x, y in zip(x_means, y_means)
    ]


