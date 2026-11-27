"""PRIORITEST-STAR — effort-aware metrics and statistical tests (roadmap Phase 1).

* popt / ifa / recall_at_loc : effort-aware ranking metrics (Kamei et al. 2013 definition of Popt,
  same oracle/worst curves as analyse_star.py: optimal = defective first then smallest LOC).
* bootstrap_ci        : class-level bootstrap 95 % CI of any metric on ONE test release.
* paired_bootstrap    : CI and two-sided p-value of the difference between two models on the same classes.
* wilcoxon_seeds      : Wilcoxon signed-rank over seed-paired results (consecutive ablation steps).
* cliffs_delta        : non-parametric effect size, with the usual magnitude labels.
All functions are pure numpy/scipy and never look at anything but the arrays they receive.
"""
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


def _curve(order, y, loc):
    x = np.concatenate([[0], np.cumsum(loc[order])]) / loc.sum()
    r = np.concatenate([[0], np.cumsum(y[order])]) / max(1, y.sum())
    return x, r


def _area(x, r):
    return float(np.trapezoid(r, x)) if hasattr(np, 'trapezoid') else float(np.trapz(r, x))


def popt(scores, y, loc):
    """Normalised effort-aware area (Popt). Classes are inspected in decreasing score order.
    Pass score/LOC as `scores` for density ordering. 1 = oracle, 0 = worst, ~0.5 = random."""
    y = np.asarray(y).astype(int); loc = np.maximum(np.asarray(loc, dtype=float), 1)
    if y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    model = _curve(np.argsort(-np.asarray(scores, dtype=float), kind='stable'), y, loc)
    opt = _curve(np.lexsort((loc, -y)), y, loc)
    worst = _curve(np.lexsort((-loc, y)), y, loc)
    return 1 - (_area(*opt) - _area(*model)) / (_area(*opt) - _area(*worst))


def ifa(scores, y):
    """Initial False Alarms: clean classes inspected before the first defective one."""
    y = np.asarray(y).astype(int)[np.argsort(-np.asarray(scores, dtype=float), kind='stable')]
    hit = np.flatnonzero(y)
    return int(hit[0]) if len(hit) else len(y)


def recall_at_loc(scores, y, loc, budget=.2):
    """Share of defective classes found when inspecting in score order until `budget` of total LOC
    (prefix rule: stop at the first class that would exceed the budget — the effort curve read at `budget`)."""
    y = np.asarray(y).astype(int); loc = np.maximum(np.asarray(loc, dtype=float), 1)
    x, r = _curve(np.argsort(-np.asarray(scores, dtype=float), kind='stable'), y, loc)
    k = np.searchsorted(x, budget, side='right') - 1
    return float(r[k])


def metric_fn(name):
    """name -> f(scores, y, loc). Density variants ('*_density') rank by score / LOC."""
    base = name.replace('_density', '')
    dens = name.endswith('_density')

    def f(s, y, loc):
        s = np.asarray(s, dtype=float)
        if dens:
            s = s / np.maximum(np.asarray(loc, dtype=float), 1)
        if base == 'AP':
            return average_precision_score(y, s)
        if base == 'ROC_AUC':
            return roc_auc_score(y, s)
        if base == 'Popt':
            return popt(s, y, loc)
        if base == 'R@20LOC':
            return recall_at_loc(s, y, loc, .2)
        if base == 'IFA':
            return ifa(s, y)
        raise KeyError(name)
    return f


def bootstrap_ci(scores, y, loc, metric='AP', n=2000, seed=0, alpha=.05):
    """Percentile bootstrap over classes. Resamples lacking both labels are skipped."""
    f = metric_fn(metric); rng = np.random.default_rng(seed)
    s, y, loc = map(np.asarray, (scores, y, loc)); N = len(y); vals = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        if 0 < y[i].sum() < N:
            vals.append(f(s[i], y[i], loc[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(f(s, y, loc)), float(lo), float(hi)


def paired_bootstrap(scores_a, scores_b, y, loc, metric='AP', n=2000, seed=0, alpha=.05):
    """Difference metric(A) - metric(B) on the same resampled classes.
    Returns (diff, ci_low, ci_high, p_two_sided). p = 2·min(P(d<=0), P(d>=0))."""
    f = metric_fn(metric); rng = np.random.default_rng(seed)
    a, b, y, loc = map(np.asarray, (scores_a, scores_b, y, loc)); N = len(y); d = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        if 0 < y[i].sum() < N:
            d.append(f(a[i], y[i], loc[i]) - f(b[i], y[i], loc[i]))
    d = np.asarray(d); lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    p = min(1., 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return float(f(a, y, loc) - f(b, y, loc)), float(lo), float(hi), float(p)


def wilcoxon_seeds(a, b):
    """Two-sided Wilcoxon signed-rank on seed-paired values. Returns p (1.0 when all differences are 0)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b, zero_method='zsplit').pvalue)


def cliffs_delta(a, b):
    """Cliff's δ = P(a>b) - P(a<b) and its magnitude (Romano et al. 2006 thresholds)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    gt = (a[:, None] > b[None, :]).mean(); lt = (a[:, None] < b[None, :]).mean(); d = float(gt - lt)
    m = abs(d)
    mag = 'negligible' if m < .147 else 'small' if m < .33 else 'medium' if m < .474 else 'large'
    return d, mag
