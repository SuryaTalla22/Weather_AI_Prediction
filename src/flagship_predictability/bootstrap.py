from __future__ import annotations
import numpy as np


def block_bootstrap_indices(n: int, block_length: int, n_samples: int, rng: np.random.Generator):
    starts = np.arange(n)
    out = []
    for _ in range(n_samples):
        idx = []
        while len(idx) < n:
            s = int(rng.choice(starts))
            idx.extend(range(s, min(s + block_length, n)))
        out.append(np.array(idx[:n], dtype=int))
    return out


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 400, block_length: int = 5, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    boots = []
    for idx in block_bootstrap_indices(len(x), block_length, n_boot, rng):
        boots.append(float(np.nanmean(x[idx])))
    boots = np.asarray(boots, dtype=float)
    return {
        "mean": float(np.nanmean(x)),
        "lo": float(np.nanquantile(boots, alpha / 2)),
        "hi": float(np.nanquantile(boots, 1 - alpha / 2)),
        "n": int(len(x)),
    }


def paired_block_bootstrap_metric(a: np.ndarray, b: np.ndarray, metric_fn, n_boot: int = 500, block_length: int = 5, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    b = np.asarray(b)
    n = min(len(a), len(b))
    if n == 0:
        return {"delta": np.nan, "lo": np.nan, "hi": np.nan, "samples": np.array([])}
    a = a[:n]
    b = b[:n]
    boots = []
    for idx in block_bootstrap_indices(n, block_length, n_boot, rng):
        boots.append(metric_fn(a[idx]) - metric_fn(b[idx]))
    boots = np.asarray(boots)
    return {
        "delta": float(metric_fn(a) - metric_fn(b)),
        "lo": float(np.nanquantile(boots, alpha / 2)),
        "hi": float(np.nanquantile(boots, 1 - alpha / 2)),
        "samples": boots,
    }
