import os

import numpy as np

from .config import (
    EMB_MAP, EMB_MAP_FILE, FREQ_MAP, FREQ_MAP_FILE,
    logger, save_config, get_path
)
from .utils.embedding import get_embedding_map, get_token_freq_map


def register_embeddings(name, path):
    """
    Register embeddings file.
    :param name: name to refer to embeddings
    :param path: path to embeddings file
    """
    path = os.path.expanduser(path)
    logger.info(f'Registering embeddings {name} from {path}')
    if name in EMB_MAP:
        logger.warning(f'Overwriting old value: {EMB_MAP[name]}')

    # Load the file to check the format.
    try:
        get_embedding_map(path, ['Be prepared'])
    except Exception:
        logger.error(f'Error loading embeddings file, not updating.')
        return

    EMB_MAP[name] = path


def register_frequencies(name, path):
    """
    Register frequencies file.
    :param name: name to refer to frequencies
    :param path: path to frequencies file
    """
    path = os.path.expanduser(path)
    logger.info(f'Registering frequencies {name} from {path}')
    if name in FREQ_MAP:
        logger.warning(f'Overwriting old value: {FREQ_MAP[name]}')

    # Load the file to check the format.
    try:
        get_token_freq_map(path)
    except Exception:
        logger.error(f'Error loading frequencies file, not updating.')
        return

    FREQ_MAP[name] = path


def save_embeddings_config():
    save_config(EMB_MAP, EMB_MAP_FILE)


def save_frequencies_config():
    save_config(FREQ_MAP, FREQ_MAP_FILE)


def embed(sequences, embedding, norm=False, frequencies=None, pad_token=None):
    """
    Converts token sequences to embedding sequences
    :param sequences: list of token sequences
    :param embedding: name of the embedding to use, or path to embedding file
    :param norm: Whether to normalise all embeddings or not
    :param frequencies: optional name of frequencies, or path to frequencies
        file, to use for weighting
    :param pad_token: If present, pads every sequence with a single instance
        of this token
    :return: embedded sequences
    """
    if pad_token is not None:
        sequences = [x + [pad_token] for x in sequences]

    embeddings = []
    embs_path = get_path(embedding, EMB_MAP)
    if embs_path is None:
        logger.error('Embedding name not found, '
                     'maybe you forgot to register it?')
        return None

    counts_path = None
    if frequencies is not None:
        counts_path = get_path(frequencies, FREQ_MAP)
        if counts_path is None:
            logger.error('Frequency name not found, '
                         'maybe you forgot to register it?')
            return None

    word_vec, dim = get_embedding_map(embs_path, sequences, norm, counts_path)

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
