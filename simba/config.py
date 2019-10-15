import json
import logging
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).parents[1]
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
EMB_MAP_FILE = os.path.join(DATA_DIR, 'emb_map.json')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_embedding_map():
    logger.debug(f'Loading embedding map from {EMB_MAP_FILE}')
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(EMB_MAP_FILE):
        return {}

    with open(EMB_MAP_FILE, 'r') as f:
        emb_map = json.load(f)
    return emb_map


def save_embedding_map(emb_map):
    logger.debug(f'Saving embedding map to {EMB_MAP_FILE}')
    with open(EMB_MAP_FILE, 'w+') as f:
        json.dump(emb_map, f)


EMB_MAP = load_embedding_map()
