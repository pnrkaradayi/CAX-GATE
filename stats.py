"""Statistical helpers."""
from math import sqrt

import numpy as np
from scipy import stats as st

Z = 1.959964


def wilson(p: float, n: int):
    """95% Wilson score interval for a proportion p observed on n items, in percent."""
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z * sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return 100 * (c - h), 100 * (c + h)


def bootstrap_ci(values, resamples=2000, seed=0, clusters=None):
    """95% percentile bootstrap interval of the mean; resamples whole clusters when given."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(values), size=(resamples, len(values)))
        means = values[idx].mean(axis=1)
    else:
        clusters = np.asarray(clusters)
        ids = np.unique(clusters)
        members = [np.nonzero(clusters == c)[0] for c in ids]
        means = np.empty(resamples)
        for b in range(resamples):
            pick = rng.integers(0, len(ids), size=len(ids))
            means[b] = values[np.concatenate([members[k] for k in pick])].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_t(a, b):
    t, p = st.ttest_rel(np.asarray(a, float), np.asarray(b, float))
    return float(t), float(p)


def mcnemar_exact(a_bin, b_bin):
    """Exact McNemar test on two paired binary outcomes."""
    a, b = np.asarray(a_bin, bool), np.asarray(b_bin, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(st.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def cohens_d_paired(a, b):
    """Cohen's d_z for paired samples: mean difference / SD of the differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def bonferroni(p: float, m: int) -> float:
    return min(1.0, p * m)


def stars(p_adj: float) -> str:
    return "**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "")


def spearman(x, y):
    r, p = st.spearmanr(x, y)
    return float(r), float(p)
