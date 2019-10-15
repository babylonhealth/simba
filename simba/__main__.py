import click
from .config import EMB_MAP, logger, save_embedding_map


@click.group()
def simba():
    pass


@click.command()
@click.option('--name', required=True, help='Name of embedding')
@click.option('--path', required=True, help='Path to embedding file in w2v txt format')
def register(name, path):
    """Register a new embeddings file to use."""
    logger.info(f'Registering {name} from {path}')
    if name in EMB_MAP:
        logger.warn(f'Overwriting old value: {EMB_MAP[name]}')
    EMB_MAP[name] = path
    save_embedding_map(EMB_MAP)


@click.command()
def embeddings():
    """Print embeddings that have been registered."""
    max_len = max(len(n) for n in EMB_MAP)
    for name, path in EMB_MAP.items():
        print(f'{name:{max_len}} => {path}')


simba.add_command(register)
simba.add_command(embeddings)
