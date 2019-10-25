# Demonstrate fuzzy bag-of-words with different universe matrices.
import numpy as np

from simba.evaluation import evaluate_multiple
from simba.similarities import fbow_jaccard_factory
from simba.core import embed
from simba.utils.embedding import load_embedding_matrix

# A very useful dataset.
sentences1 = [
    "Remember who you are",
    "Any story worth telling is worth telling twice",
    "Being brave doesn’t mean you go looking for trouble",
]
sentences2 = [
    "Remember that those kings will always be there to guide you",
    "Any story worth telling is worth telling twice",
    "I'm surrounded by idiots"
]
gold_scores = [1, 2, 0]

# Get similarity functions corresponding to various universe matrices
dim = 300
universes = [
    load_embedding_matrix('fasttext'),
    np.eye(dim),
    np.random.normal(size=(dim, dim))
]
sim_fns = [fbow_jaccard_factory(u) for u in universes]
names = ['vocab', 'id', 'random']

# Get word embeddings.
sentences1 = [s.split() for s in sentences1]
sentences2 = [s.split() for s in sentences2]
embeddings1 = embed(sentences1, embedding='fasttext')
embeddings2 = embed(sentences2, embedding='fasttext')

# Compute scores for all similarity functions.
all_scores = evaluate_multiple(
    embeddings1,
    embeddings2,
    sim_fns,
    gold_scores,
    names,
)
print(all_scores)
