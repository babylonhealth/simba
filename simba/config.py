import json
import logging
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).parents[1]
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
EMB_MAP_FILE = os.path.join(DATA_DIR, 'emb_map.json')
FREQ_MAP_FILE = os.path.join(DATA_DIR, 'freq_map.json')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(filename):
    logger.debug(f'Loading config map from {filename}')
    if not os.path.exists(filename):
        return {}
    with open(filename, 'r') as f:
        config_map = json.load(f)
    return config_map


def print_config(config_map):
    if len(config_map) == 0:
        print('Empty.')
        return
    print()
    max_len = max(len(n) for n in config_map)
    for name, path in config_map.items():
        print(f'{name:{max_len}} => {path}')
    print()


def save_config(config_map, filename):
    logger.debug(f'Saving config map to {filename}')
    with open(filename, 'w+') as f:
        json.dump(config_map, f)


EMB_MAP = load_config(EMB_MAP_FILE)
FREQ_MAP = load_config(FREQ_MAP_FILE)
