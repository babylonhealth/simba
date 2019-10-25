import numpy as np


def _fuzzify(s, u):
    """
    Sentence fuzzifier.
    Computes membership vector for the sentence S with respect to the
    universe U
    :param s: list of word embeddings for the sentence
    :param u: the universe matrix U with shape (K, d)
    :return: membership vectors for the sentence
    """
    f_s = np.dot(s, u.T)
    m_s = np.max(f_s, axis=0)
    m_s = np.maximum(m_s, 0, m_s)
    return m_s


def dynamax_jaccard(x, y):
    """
    DynaMax-Jaccard similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    u = np.vstack((x, y))
    m_x = _fuzzify(x, u)
    m_y = _fuzzify(y, u)

    m_inter = np.sum(np.minimum(m_x, m_y))
    m_union = np.sum(np.maximum(m_x, m_y))
    return m_inter / m_union


def dynamax_otsuka(x, y):
    """
    DynaMax-Otsuka similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    u = np.vstack((x, y))
    m_x = _fuzzify(x, u)
    m_y = _fuzzify(y, u)

    m_inter = np.sum(np.minimum(m_x, m_y))
    m_x_card = np.sum(m_x)
    m_y_card = np.sum(m_y)
    return m_inter / np.sqrt(m_x_card * m_y_card)


def dynamax_dice(x, y):
    """
    DynaMax-Dice similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    u = np.vstack((x, y))
    m_x = _fuzzify(x, u)
    m_y = _fuzzify(y, u)

    f_inter = np.sum(np.minimum(m_x, m_y))
    m_x_card = np.sum(m_x)
    m_y_card = np.sum(m_y)
    return 2 * f_inter / (m_x_card + m_y_card)


def max_jaccard(x, y):
    """
    MaxPool-Jaccard similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    m_x = np.max(x, axis=0)
    m_x = np.maximum(m_x, 0, m_x)
    m_y = np.max(y, axis=0)
    m_y = np.maximum(m_y, 0, m_y)
    m_inter = np.sum(np.minimum(m_x, m_y))
    m_union = np.sum(np.maximum(m_x, m_y))
    return m_inter / m_union


def fbow_jaccard_factory(u):
    """
    Factory for building FBoW-Jaccard similarity measures
    with the custom universe matrix U
    :param u: the universe matrix U
    :return: similarity function
    """
    def u_jaccard(x, y):
        m_x = _fuzzify(x, u)
        m_y = _fuzzify(y, u)

        m_inter = np.sum(np.minimum(m_x, m_y))
        m_union = np.sum(np.maximum(m_x, m_y))
        return m_inter / m_union
    return u_jaccard
