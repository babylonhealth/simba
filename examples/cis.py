from simba.evaluation import evaluate_multiple, confidence_intervals
from simba.similarities import dynamax_jaccard, avg_cosine
from simba.core import embed

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
sentences1 = [s.split() for s in sentences1]
sentences2 = [s.split() for s in sentences2]
embeddings1 = embed(sentences1, embedding='fasttext')
embeddings2 = embed(sentences2, embedding='fasttext')

# Compute confidence intervals for dynamax compared to cossim.
all_scores = evaluate_multiple(
    embeddings1,
    embeddings2,
    [dynamax_jaccard, avg_cosine]
)
print(all_scores)

dm_scores = all_scores['dynamax_jaccard']
ac_scores = all_scores['avg_cosine']
cis = confidence_intervals(dm_scores, ac_scores, gold_scores)
print(cis)
