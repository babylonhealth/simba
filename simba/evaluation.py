import numpy as np
from scipy.stats import pearsonr
import scikits.bootstrap as bootstrap


def evaluate(xs, ys, sim_fn, gold_scores=None):
    """
    Evaluate a similarity measure on a dataset of pairs.
    :param xs: first list of sequences of embeddings
    :param ys: second list of sequences of embeddings
    :param sim_fn: similarity measure to evaluate
    :param gold_scores: optional list of gold scores
    :return: similarity score for each pair, and Pearson correlation
        (if gold scores were passed)
    """
    scores = [
        np.nan_to_num(sim_fn(np.nan_to_num(x), np.nan_to_num(y)))
        for x, y in zip(xs, ys)
    ]
    if gold_scores is not None:
        prs = pearsonr(gold_scores, scores)[0]
        return scores, prs
    return scores, None


def evaluate_multiple(xs, ys, sim_fns, gold_scores=None, names=None):
    """
    Compare multiple similarity measures on a dataset of pairs.
    :param xs: first list of sequences of embeddings
    :param ys: second list of sequences of embeddings
    :param sim_fns: similarity measures to evaluate
    :param gold_scores: optional list of gold scores
    :param names: optional list of names for the similarity functions, which
        will be keys in the returned dict. If omitted it will default to the
        __name__ of he function
    :return: dict containing scores and Pearson correlation (if gold scores
        were passed) for each method
    """
    if names is None:
        names = [sim_fn.__name__ for sim_fn in sim_fns]
    return {
        name: evaluate(xs, ys, sim_fn, gold_scores)
        for name, sim_fn in zip(names, sim_fns)
    }


def confidence_intervals(system_scores, baseline_scores, gold_scores):
    """
    Compute BCa confidence intervals for a system compared to a baseline.
    :param system_scores: list of system's scores
    :param baseline_scores: list of baseline method's scores
    :param gold_scores: list of gold scores
    :return: dict containing system and baseline Pearson correlation,
        delta between them, and confidence interval
    """
    system_prs = pearsonr(gold_scores, system_scores)[0]
    baseline_prs = pearsonr(gold_scores, baseline_scores)[0]

    data = list(zip(gold_scores, system_scores, baseline_scores))

    def statistic(data):
        gs = data[:, 0]
        sys = data[:, 1]
        base = data[:, 2]
        r1 = pearsonr(gs, sys)[0]
        r2 = pearsonr(gs, base)[0]
        return r1 - r2

    conf_int = bootstrap.ci(data, statfunction=statistic, method='bca')
    return {
        'system': system_prs,
        'baseline': baseline_prs,
        'delta': system_prs - baseline_prs,
        'conf_int': list(conf_int),
    }
