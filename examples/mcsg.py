from simba.evaluation import evaluate_multiple
from simba.similarities import von_mises_correction_aic, von_mises_correction_tic, gaussian_correction_aic, gaussian_correction_tic, spherical_gaussian_correction_aic, spherical_gaussian_correction_tic
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
    'i have no more milk',
]
gold_scores = [1, 2, 0]

# Get word embeddings.
# Note that the VMF methods require more than one embedding per sequence,
# so it's advised to pad with an arbitrary (common) word embedding, e.g.
# '.' or 'the' in the case of text.
sentences1 = [s.split() + ['.'] for s in sentences1]
sentences2 = [s.split() + ['.'] for s in sentences2]
embeddings1 = get_embeddings(sentences1, embedding='fasttext')
embeddings2 = get_embeddings(sentences2, embedding='fasttext')

# Compute confidence intervals for dynamax compared to cossim.
all_scores = evaluate_multiple(
    embeddings1,
    embeddings2,
    [
        # TODO VMF still nan's
        # von_mises_correction_aic,
        # von_mises_correction_tic,
        gaussian_correction_aic,
        gaussian_correction_tic,
        spherical_gaussian_correction_aic,
        spherical_gaussian_correction_tic,
    ],
    gold_scores,
)
print(all_scores)
