import numpy as np

from ..utils.gaussian import aic_spherical, tic, tic_spherical
from ..utils.vmf import vmf_aic, vmf_tic


def von_mises_correction_aic(Dnew, Dc):
    """
    :param Dnew[nxd matrix]: set 1
    :param Dc[mxd matrix]: set 2
    :return [float]: semantic similarity measure
        (approximation of the bayes factors)
    """
    D = np.concatenate((Dnew, Dc), axis=0)

    aic_x = -vmf_aic(Dnew)
    aic_y = -vmf_aic(Dc)
    aic_xy = -vmf_aic(D)
    similarity = aic_xy - (aic_x + aic_y)

    return similarity


def von_mises_correction_tic(Dnew, Dc):
    """
    :param Dnew[nxd matrix]: set 1
    :param Dc[mxd matrix]: set 2
    :return [float]: semantic similarity measure
        (approximation of the bayes factors)
    """
    D = np.concatenate((Dnew, Dc), axis=0)

    tic_x = -vmf_tic(Dnew)
    tic_y = -vmf_tic(Dc)
    tic_xy = -vmf_tic(D)
    similarity = tic_xy - (tic_x + tic_y)

    return similarity


def gaussian_correction_aic(Dnew, Dc):
    """
    :param Dnew[nxd matrix]: set 1
    :param Dc[mxd matrix]: set 2
    :return [float]: semantic similarity measure
        (approximation of the bayes factors)
    """
    Dnew = np.array(Dnew)
    Dc = np.array(Dc)
    K, D = Dnew.shape
    L, D = Dc.shape

    mu_1 = np.mean(Dnew, axis=0)
    mu_2 = np.mean(Dc, axis=0)
    mu_1_sq = np.mean(Dnew ** 2, axis=0)
    mu_2_sq = np.mean(Dc ** 2, axis=0)
    p1 = K * 1.0 / (K + L)
    mu_3 = p1 * mu_1 + (1 - p1) * mu_2
    mu_3_sq = p1 * mu_1_sq + (1 - p1) * mu_2_sq

    reg = 1e-5
    v_1 = mu_1_sq - mu_1 ** 2 + reg
    v_2 = mu_2_sq - mu_2 ** 2 + reg
    v_3 = mu_3_sq - mu_3 ** 2 + reg

    ll_fast_x = K * np.sum(np.log(v_1))
    ll_fast_y = L * np.sum(np.log(v_2))
    ll_fast_xy = (K + L) * np.sum(np.log(v_3))
    similarity_fast = - ll_fast_xy + ll_fast_x + ll_fast_y

    return similarity_fast + 4 * D


def spherical_gaussian_correction_aic(Dnew, Dc):
    """
    :param Dnew[nxd matrix]: set 1
    :param Dc[mxd matrix]: set 2
    :return [float]: semantic similarity measure
        (approximation of the bayes factors)
    """
    D = np.concatenate((Dnew, Dc), axis=0)

    aic_x = -aic_spherical(Dnew)
    aic_y = -aic_spherical(Dc)
    aic_xy = -aic_spherical(D)

    return aic_xy - (aic_x + aic_y)


def gaussian_correction_tic(Dnew, Dc):
    """
    :param Dnew[nxd matrix]: set 1
    :param Dc[mxd matrix]: set 2
    :return [float]: semantic similarity measure
        (approximation of the bayes factors)
    """
    D = np.concatenate((Dnew, Dc), axis=0)

    tic_x = -tic(Dnew)
    tic_y = -tic(Dc)
    tic_xy = -tic(D)

    similarity = tic_xy - (tic_x + tic_y)

    return similarity


def spherical_gaussian_correction_tic(Dnew, Dc):
    """
    :param Dnew[nxd matrix]: set 1
    :param Dc[mxd matrix]: set 2
    :return [float]: semantic similarity measure
        (approximation of the bayes factors)
    """
    D = np.concatenate((Dnew, Dc), axis=0)

    tic_x = -tic_spherical(Dnew)
    tic_y = -tic_spherical(Dc)
    tic_xy = -tic_spherical(D)

    similarity = tic_xy - (tic_x + tic_y)

    return similarity
