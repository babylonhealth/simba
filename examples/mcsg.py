# Demonstrate usage of model comparison methods.
from simba.evaluation import evaluate_multiple
from simba.similarities import (
    von_mises_correction_aic, von_mises_correction_tic,
    gaussian_correction_aic, gaussian_correction_tic,
    spherical_gaussian_correction_aic, spherical_gaussian_correction_tic
)
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
# Note that the VMF methods require more than one embedding per sequence,
# so it's advised to pad with an arbitrary (common) token, e.g. '.' or 'the'
# in the case of text.
# VMF also requires normalised word vectors.
sentences1 = [s.split() for s in sentences1]
sentences2 = [s.split() for s in sentences2]
embeddings1 = embed(sentences1, embedding='fasttext', norm=True, pad_token='.')
embeddings2 = embed(sentences2, embedding='fasttext', norm=True, pad_token='.')

# Compute results for all AIC/TIC methods.
all_scores = evaluate_multiple(
    embeddings1,
    embeddings2,
    [
        von_mises_correction_aic,
        von_mises_correction_tic,
        gaussian_correction_aic,
        gaussian_correction_tic,
        spherical_gaussian_correction_aic,
        spherical_gaussian_correction_tic,
    ],
    gold_scores,
)
for method, result in all_scores.items():
    print(f'{method}: {result[0]}')
