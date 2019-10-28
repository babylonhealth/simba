# Demonstrate computing confidence intervals.
from simba.evaluation import evaluate_multiple, confidence_intervals
from simba.similarities import dynamax_jaccard, avg_cosine
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
embeddings1 = embed(sentences1, embedding='fasttext')
embeddings2 = embed(sentences2, embedding='fasttext')

# Compute confidence intervals for dynamax compared to cossim.
all_scores = evaluate_multiple(
    embeddings1,
    embeddings2,
    [dynamax_jaccard, avg_cosine]
)
print(all_scores)

dm_scores = all_scores['dynamax_jaccard'][0]
ac_scores = all_scores['avg_cosine'][0]
cis = confidence_intervals(dm_scores, ac_scores, gold_scores)
print(cis)
