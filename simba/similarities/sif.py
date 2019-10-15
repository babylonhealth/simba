import numpy as np

from simba.utils.linalg import compute_pc, cosine


def avg_sif(x, y, pc):
    """

    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :param pc: 1st principle component of the embedding matrix
    :return: similarity measure between the two sentences
    """
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    x_mean = (x_mean - x_mean.dot(pc.T) * pc).reshape(-1)
    y_mean = (y_mean - y_mean.dot(pc.T) * pc).reshape(-1)
    return cosine(x_mean, y_mean)


def batch_sif(xs, ys):
    """

    :param xs:
    :param ys:
    :return:
    """
    embs = np.vstack(np.hstack((xs, ys)))
    pc = compute_pc(embs)
    return [
        avg_sif(x, y, pc) for x, y in zip(xs, ys)
    ]


