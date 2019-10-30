import numpy as np


def fit_covariance(X, reg_cov=1e-10):
    variances = np.var(X, axis=0, ddof=0) + reg_cov
    return variances


def fit_mean(X):
    mean = np.mean(X, axis=0)
    return mean


def get_score(X, mu, var_diag):
    X = np.array(X)
    N, D = X.shape
    log_prob = np.sum((X - mu) ** 2 / var_diag)
    log_det = np.sum(np.log(var_diag))
    return -.5 * (log_prob + N * D * np.log(2 * np.pi) + N * log_det)


def aic(X):
    X = np.array(X)
    N, D = X.shape
    K = D * 2
    mu = fit_mean(X)
    var_diag = fit_covariance(X)
    return - 2 * get_score(X, mu, var_diag) + 2 * K


def tic(X):
    X = np.array(X)
    N, D = X.shape
    mu = fit_mean(X)
    var_diag = fit_covariance(X)

    jacob = np.zeros((2 * D,))
    hess = np.zeros((2 * D,))

    grad_mu = lambda x: (x - mu) / var_diag  # noqa
    grad_var = lambda x: .5 * (((x - mu) / var_diag) ** 2 - 1 / var_diag)  # noqa

    # Jacob of mus
    for n in range(N):
        jacob[:D] += grad_mu(X[n]) ** 2

    # Jacob of sigmas
    for n in range(N):
        jacob[D:] += grad_var(X[n]) ** 2
    jacob /= N

    # Hess of mus
    hess[:D] = - N / var_diag

    # Hess of sigmas
    for n in range(N):
        hess[D:] -= (X[n] - mu) ** 2 / var_diag ** 3
    hess[D:] += N / (var_diag ** 2) / 2
    hess /= N

    # Fisher is negative of Hessian
    fisher = -hess

    penalty = np.sum(jacob / (fisher + 1e-06))

    score = get_score(X, mu, var_diag)

    return - 2 * (score - penalty)


def fit_covariance_spherical(X, reg_cov=1e-6):
    variances = np.var(X, axis=0, ddof=0) + reg_cov
    spherical = np.mean(variances)
    return spherical


def get_score_spherical(X, mu, var):
    X = np.array(X)
    N, D = X.shape
    log_prob = np.sum((X - mu) ** 2) / var
    log_det = D * np.log(var)
    return -.5 * (log_prob + N * D * np.log(2 * np.pi) + N * log_det)


def aic_spherical(X):
    X = np.array(X)
    N, D = X.shape
    K = D + 1
    mu = fit_mean(X)
    var = fit_covariance_spherical(X)
    return - 2 * get_score_spherical(X, mu, var) + 2 * K


def tic_spherical(X):
    X = np.array(X)
    N, D = X.shape
    mu = fit_mean(X)
    var = fit_covariance_spherical(X)

    jacob = np.zeros((D + 1,))
    hess = np.zeros((D + 1,))

    grad_mu = lambda x: (x - mu) / var  # noqa
    grad_var = lambda x: .5 * np.sum(((x - mu) / var) ** 2 - 1 / var)  # noqa

    # Jacob of mus
    for n in range(N):
        jacob[:D] += grad_mu(X[n]) ** 2

    # Jacob of sigmas
    for n in range(N):
        jacob[D] += grad_var(X[n]) ** 2
    jacob /= N

    # Hess of mus
    hess[:D] = - N / var

    # Hess of sigma
    for n in range(N):
        hess[D] -= np.sum((X[n] - mu) ** 2) / var ** 3
    hess[D] += N / (var ** 2) / 2
    hess /= N

    # Fisher is negative of Hessian
    fisher = -hess

    penalty = np.sum(jacob / fisher)

    score = get_score_spherical(X, mu, var)

    return - 2 * (score - penalty)
