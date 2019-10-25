# flake8: noqa
from .cka import (
    cka_factory,
    cka_linear,
    cka_gaussian,
    dcorr,
)
from .correlation import (
    avg_pearson,
    avg_spearman,
    avg_kendall,
    max_spearman,
)
from .fuzzy import (
    dynamax_jaccard,
    dynamax_otsuka,
    dynamax_dice,
    max_jaccard,
    fbow_jaccard_factory,
)
from .geometry import avg_cosine
from .mcsg import (
    gaussian_correction_aic,
    gaussian_correction_tic,
    spherical_gaussian_correction_aic,
    spherical_gaussian_correction_tic,
    von_mises_correction_aic,
    von_mises_correction_tic,
)
from .sif import batch_avg_pca
