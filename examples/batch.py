from scipy.stats import pearsonr

from simba.similarities import batch_sif
from simba.utils.embedding import get_embeddings

# A very useful dataset.
sentences1 = [
    'she likes cats',
    'she likes airplanes',
    'frogs are good',
]
sentences2 = [
    'he loves dogs',
    'airplanes are cool',
    'i have no more milk'
]
gold_scores = [1, 2, 0]

# Get word embeddings.
embeddings1 = get_embeddings(sentences1, embedding='fasttext')
embeddings2 = get_embeddings(sentences2, embedding='fasttext')

# Evaluate batch methods.
sif_scores = batch_sif(embeddings1, embeddings2)
print(sif_scores)
print(pearsonr(sif_scores, gold_scores)[0])
