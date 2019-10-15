from scipy.stats import pearsonr
import scikits.bootstrap as bootstrap


def evaluate(xs, ys, sim_fn, gold_scores=None):
    """
    Evaluate a similarity measure on a dataset of pairs.
    :param xs:
    :param ys:
    :param sim_fn:
    :param gold_scores:
    :return:
    """
    scores = [sim_fn(x, y) for x, y in zip(xs, ys)]
    if gold_scores is not None:
        prs = pearsonr(gold_scores, scores)[0]
        return prs, scores
    return scores


def evaluate_multiple(xs, ys, sim_fns, gold_scores=None):
    """
    Compare multiple similarity measures on a dataset of pairs.
    :param xs:
    :param ys:
    :param sim_fns:
    :param gold_scores:
    :return:
    """
    return {
        sim_fn.__name__: evaluate(xs, ys, sim_fn, gold_scores)
        for sim_fn in sim_fns
    }


def confidence_intervals(system_scores, baseline_scores, gold_scores):
    """
    Compute BCa confidence intervals for a system compared to a baseline.
    :param system_scores:
    :param baseline_scores:
    :param gold_scores:
    :return:
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
