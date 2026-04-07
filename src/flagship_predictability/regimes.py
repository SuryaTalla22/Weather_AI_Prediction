from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from .datasets import infer_dims
from .bootstrap import bootstrap_mean_ci, paired_block_bootstrap_metric


def _flatten_space(da: xr.DataArray):
    dims = infer_dims(da)
    time_dim = "valid_time" if "valid_time" in da.dims else dims["time"]
    lat_name, lon_name = dims["lat"], dims["lon"]
    X = da.transpose(time_dim, lat_name, lon_name).fillna(0.0).values
    nt, ny, nx = X.shape
    return X.reshape(nt, ny * nx), time_dim, lat_name, lon_name


def build_regime_labels(truth_da: xr.DataArray, n_components: int = 10, n_regimes: int = 4, random_state: int = 0):
    X, time_dim, lat_name, lon_name = _flatten_space(truth_da)
    X = X - X.mean(axis=0, keepdims=True)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    pcs = svd.fit_transform(X)
    km = KMeans(n_clusters=n_regimes, n_init=20, random_state=random_state)
    labels = km.fit_predict(pcs)
    label_da = xr.DataArray(labels, dims=[time_dim], coords={time_dim: truth_da[time_dim]}, name="regime")
    return {"labels": label_da, "svd": svd, "kmeans": km, "pcs": pcs}


def regime_metric_dataframe(model_results: dict, regime_labels: xr.DataArray, metric_name: str):
    rows = []
    time_name = regime_labels.dims[0]
    for model_name, lead_map in model_results.items():
        for lead, metric_da in lead_map.items():
            arr = metric_da
            if arr.dims[0] != time_name:
                arr = arr.rename({arr.dims[0]: time_name})
            common = np.intersect1d(arr[time_name].values, regime_labels[time_name].values)
            a = arr.sel({time_name: common})
            r = regime_labels.sel({time_name: common})
            for regime in np.unique(r.values):
                vals = a.where(r == regime, drop=True).values
                rows.append({
                    "model": model_name,
                    "lead": pd.to_timedelta(lead),
                    "regime": int(regime),
                    "metric": metric_name,
                    "n": int(np.isfinite(vals).sum()),
                    "mean": float(np.nanmean(vals)),
                    "median": float(np.nanmedian(vals)),
                    "std": float(np.nanstd(vals)),
                })
    return pd.DataFrame(rows)


def regime_metric_dataframe_with_ci(model_results: dict, regime_labels: xr.DataArray, metric_name: str, n_boot: int = 400, block_length: int = 5, seed: int = 0):
    rows = []
    time_name = regime_labels.dims[0]
    for model_name, lead_map in model_results.items():
        for lead, metric_da in lead_map.items():
            arr = metric_da
            if arr.dims[0] != time_name:
                arr = arr.rename({arr.dims[0]: time_name})
            common = np.intersect1d(arr[time_name].values, regime_labels[time_name].values)
            a = arr.sel({time_name: common})
            r = regime_labels.sel({time_name: common})
            for regime in np.unique(r.values):
                vals = a.where(r == regime, drop=True).values
                ci = bootstrap_mean_ci(vals, n_boot=n_boot, block_length=block_length, seed=seed)
                rows.append({
                    "model": model_name,
                    "lead": pd.to_timedelta(lead),
                    "regime": int(regime),
                    "metric": metric_name,
                    "n": ci["n"],
                    "mean": ci["mean"],
                    "ci_lo": ci["lo"],
                    "ci_hi": ci["hi"],
                    "median": float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan,
                    "std": float(np.nanstd(vals)) if np.isfinite(vals).any() else np.nan,
                })
    return pd.DataFrame(rows)


def regime_sensitivity_table(truth_anom: xr.DataArray, component_grid: list[int], regime_grid: list[int], random_state: int = 0) -> pd.DataFrame:
    rows = []
    for n_comp in component_grid:
        for n_reg in regime_grid:
            bundle = build_regime_labels(truth_anom, n_components=n_comp, n_regimes=n_reg, random_state=random_state)
            labels = bundle["labels"].values
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                rows.append({
                    "n_components": n_comp,
                    "n_regimes": n_reg,
                    "regime": int(u),
                    "count": int(c),
                    "fraction": float(c / counts.sum()),
                    "explained_variance_sum": float(bundle["svd"].explained_variance_ratio_.sum()),
                })
    return pd.DataFrame(rows)

__all__ = [
    "build_regime_labels",
    "regime_metric_dataframe",
    "regime_metric_dataframe_with_ci",
    "regime_sensitivity_table",
    "paired_block_bootstrap_metric",
]
