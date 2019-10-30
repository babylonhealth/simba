from scipy.special import ive
import numpy as np
import numpy.linalg as la


def to_cartesian(phi):
    """
    Converts from spherical hyperpolars
    :params Phi [nxd ndarray]: array vectors of angles
    :return X [nx(d+1) ndarray]: Dataset in cartesian coordinates
    """
    d = len(phi)
    X = np.cos(phi[0])

    # can probably be vectorised with a cumulative product
    for i in range(1, d):
        X += np.prod(np.sin(phi[:i]), axis=1) * np.cos(phi[i])

    X += np.prod(np.sin(phi))
    return X


def fit_mean_direction(X):
    return X.sum(axis=0) / la.norm(X.sum(axis=0))


def fit_concentration(X):
    """
    Computes the vMF MLE sol for the concentration parameter
    NOTE: This is an approximate solution to a transcendental eq
    :param X [nxd ndarray]: Design matrix of normalised word vectors
    :return [float]: MLE concentration parameter solution
    """
    X = np.array(X)
    n, d = X.shape
    R = la.norm(X.sum(axis=0)) / n
    Rs = R**2
    return R * (d - Rs) / (1.0 - Rs)


def to_hypersphericalPolars(mu):
    """
    Return the d-1 angles describing mu
    :param mu [dx1 ndarray]: unit norm d dimensional vector
    """
    mu_sq = mu ** 2
    rev_cumsum = np.cumsum(mu_sq[::-1])[::-1]
    rev_cumnorms = np.sqrt(rev_cumsum)

    thetas = np.arccos(mu[:-1] / rev_cumnorms[:-1])

    norm = la.norm(mu[-2:])
    thetas[-1] = np.sign(mu[-1]) * np.arccos(mu[-2] / norm)
    return thetas


def log_vMF_gradient(opt_mu, k, x):
    """
    :params opt_mu[dx1 ndarray]: optimum direction
    :params k[float]: optimum concentration
    :params x[d, ndarray]: A datapoint to differentiate about
    """

    def analytic_grad(k_phi_mu):
        kappa = k_phi_mu[0]
        phi = k_phi_mu[1:]

        elementwise = x * opt_mu
        cdots = np.cumsum(elementwise[::-1])[::-1]

        tans = np.tan(phi)

        comps = -elementwise[:-1] * tans
        invs = cdots[1:] / tans

        D = x.shape[0]
        grad = invs + comps
        grad *= kappa

        v = D * 1.0 / 2 - 1
        grad_kappa = np.dot(opt_mu, x) - ive(v+1, k) / ive(v, k)

        return np.concatenate([[grad_kappa], grad])

    phi_mu = to_hypersphericalPolars(opt_mu)

    k_phi_mu = np.concatenate(([k], phi_mu), axis=0)

    a_gradient = analytic_grad(k_phi_mu)

    return a_gradient


def log_likelihood(X):
    X = np.array(X)
    N, D = X.shape
    V = 0.5 * D - 1  # Order of the bessel function
    R = la.norm(X.sum(axis=0)) / N
    kappa = fit_concentration(X)
    # These tests should really be regarded as having no knowledge self.
    # All of them coincide with one word in each sentence that is equal
    # This maps the score to infinity.
    if np.isinf(kappa) or kappa > 1e+17:
        return np.nan
    besseli = -N * (np.log(ive(V, kappa)) + kappa - V * np.log(kappa))
    exponent = kappa * R * N
    return exponent + besseli


def vmf_aic(X):
    X = np.array(X)
    N, D = X.shape
    K = D + 1
    return - 2 * log_likelihood(X) + 2 * K


def vmf_tic(X):
    X = np.array(X)
    ll = log_likelihood(X)

    if np.isnan(ll):
        return np.nan

    N, D = X.shape
    mu = fit_mean_direction(X)
    kappa = fit_concentration(X)

    total = np.zeros(D)
    hess_diagonal = np.zeros(D - 1)
    for i, x in enumerate(X):
        grad_i = log_vMF_gradient(mu, kappa, x)
        total += grad_i ** 2

        elementwise = x * mu
        cdots = np.cumsum(elementwise[::-1])[::-1]

        hess_diagonal += -cdots[:-1] * kappa
    total /= N
    hess_diagonal /= N

    v = (D / 2 - 1)
    num = ive(v + 1, kappa) * (ive(v - 1, kappa) + ive(v + 1, kappa)) - ive(v, kappa) * (ive(v, kappa) + ive(v + 2, kappa))  # noqa
    denom = 2 * ive(v, kappa) ** 2

    k_diag = num / denom

    fisher_diagonal = - np.concatenate([[k_diag], hess_diagonal])

    correction = np.sum(total / fisher_diagonal)

    return - (ll - correction)
