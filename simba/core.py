import numpy as np

from .config import EMB_MAP, FREQ_MAP, save_config, FREQ_MAP_FILE, logger, EMB_MAP_FILE
from .utils.embedding import get_embedding_map, get_token_freq_map


def register_embeddings(name, path):
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
    save_config(EMB_MAP, EMB_MAP_FILE)


def register_frequencies(name, path):
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
    save_config(FREQ_MAP, FREQ_MAP_FILE)


def embed(sequences, embedding, norm=False, frequencies=None):
    """

    :param sequences:
    :param embedding:
    :param norm:
    :param frequencies:
    :return:
    """
    embeddings = []
    try:
        embs_path = EMB_MAP[embedding]
    except KeyError:
        logger.error('Embedding name not found, maybe you forgot to register it?')
        return None

    try:
        counts_path = FREQ_MAP[frequencies] if frequencies else None
    except KeyError:
        logger.error('Frequency name not found, maybe you forgot to register it?')
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
