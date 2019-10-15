import io

import numpy as np

from simba.config import EMB_MAP


def _create_dictionary(sequences, threshold=0):
    tokens = {}
    for s in sequences:
        for token in s:
            tokens[token] = tokens.get(token, 0) + 1

    if threshold > 0:
        new_tokens = {}
        for token in tokens:
            if tokens[token] >= threshold:
                new_tokens[token] = tokens[token]
        tokens = new_tokens
    # TODO where are these numbers coming from?
    tokens['<s>'] = 1e9 + 4
    tokens['</s>'] = 1e9 + 3
    tokens['<p>'] = 1e9 + 2

    sorted_tokens = sorted(tokens.items(), key=lambda x: -x[1])  # inverse sort
    id2token = []
    token2id = {}
    for i, (t, _) in enumerate(sorted_tokens):
        id2token.append(t)
        token2id[t] = i

    return id2token, token2id


def _get_embedding_map(embedding_path, sequences, path_to_counts=None):
    embedding_map = {}
    token_freq_map = None
    if path_to_counts:
        token_freq_map = _get_token_freq_map(path_to_counts)
    _, token2id = _create_dictionary(sequences)

    with io.open(embedding_path, 'r', encoding='utf-8', errors='ignore') as f:
        _, dim = next(f).split()
        dim = int(dim)
        for line in f:
            token, vec = line.split(' ', 1)
            if token in token2id:
                np_vector = np.fromstring(vec, sep=' ')
                if token_freq_map:
                    np_vector = _get_token_weight(token, token_freq_map) * np_vector
                embedding_map[token] = np_vector

    return embedding_map, dim


def _get_token_freq_map(path_to_counts):
    """
    Loads word counts and calculates word frequencies
    :param path_to_counts: path to word frequency file
    :return: dict containing word: word frequency
    """
    token_count_list = []

    total_count = 0.0
    with io.open(path_to_counts, 'r') as f:
        for line in f:
            token_count = line.split(' ')
            token = token_count[0]
            count = float(token_count[1])
            total_count += count
            token_count_list.append((token, count))

    token_freq_map = {}
    for token_count in token_count_list:
        token_freq_map[token_count[0]] = token_count[1] / total_count

    return token_freq_map


def _get_token_weight(token, token_freq_map, a=1e-3):
    """
    Computes SIF weight (Arora et al. 2017)
    :param token: input word
    :param token_freq_map: dict containing word: word freq.
    :param a: weight parameter
    :return: SIF weight for the word
    """
    token_freq = token_freq_map.get(token, 0.0)
    return a / (a + token_freq)


def get_embeddings(sequences, embedding, counts_path=None):
    embeddings = []
    embs_path = EMB_MAP[embedding]
    word_vec, dim = _get_embedding_map(embs_path, sequences, counts_path)

    for seq in sequences:
        seq_vec = []
        for token in seq:
            if token in word_vec:
                seq_vec.append(word_vec[token])
        if not seq_vec:
            vec = np.zeros(dim)
            seq_vec.append(vec)
        embeddings.append(seq_vec)

    return embeddings
