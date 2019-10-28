import click

from .core import (
    register_embeddings, register_frequencies,
    save_embeddings_config, save_frequencies_config
)
from .config import EMB_MAP, FREQ_MAP, print_config


@click.group()
def simba():
    pass


@click.group()
def embs():
    pass


@click.group()
def freqs():
    pass


@click.command()
def list_embs():
    """Print embeddings that have been registered."""
    print_config(EMB_MAP)


@click.command()
def list_freqs():
    """Print frequencies that have been registered."""
    print_config(FREQ_MAP)


@click.command()
@click.option('--name', required=True, help='Name of embedding')
@click.option('--path', required=True, help='Path to embedding file')
def register_emb(name, path):
    """Register a new embeddings file to use."""
    register_embeddings(name, path)
    save_embeddings_config()


@click.command()
@click.option('--name', required=True, help='Name of frequencies')
@click.option('--path', required=True, help='Path to frequency file')
def register_freq(name, path):
    """Register a new frequencies file to use."""
    register_frequencies(name, path)
    save_frequencies_config()


simba.add_command(embs)
embs.add_command(list_embs, name='list')
embs.add_command(register_emb, name='register')

simba.add_command(freqs)
freqs.add_command(list_freqs, name='list')
freqs.add_command(register_freq, name='register')
