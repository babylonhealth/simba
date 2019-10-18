from simba.evaluation import evaluate_multiple
from simba.similarities import von_mises_correction_aic, von_mises_correction_tic, gaussian_correction_aic, gaussian_correction_tic, spherical_gaussian_correction_aic, spherical_gaussian_correction_tic
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
    'i have no more milk',
]
gold_scores = [1, 2, 0]

# Get word embeddings.
# Note that the VMF methods require more than one embedding per sequence,
# so it's advised to pad with an arbitrary (common) word embedding, e.g.
# '.' or 'the' in the case of text.
# VMF also requires normalised word vectors.
sentences1 = [s.split() + ['.'] for s in sentences1]
sentences2 = [s.split() + ['.'] for s in sentences2]
embeddings1 = embed(sentences1, embedding='fasttext', norm=True)
embeddings2 = embed(sentences2, embedding='fasttext', norm=True)

# Compute confidence intervals for dynamax compared to cossim.
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
