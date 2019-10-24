# Demonstrate usage of batch STS methods.
from scipy.stats import pearsonr

from simba.similarities import batch_avg_pca
from simba.core import embed

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

# Get word embeddings.
sentences1 = [s.split() for s in sentences1]
sentences2 = [s.split() for s in sentences2]
embeddings1 = embed(sentences1, embedding='fasttext', frequencies='arora')
embeddings2 = embed(sentences2, embedding='fasttext', frequencies='arora')

# Evaluate batch methods.
sif_scores = batch_avg_pca(embeddings1, embeddings2)
print(sif_scores)
print(pearsonr(sif_scores, gold_scores)[0])
