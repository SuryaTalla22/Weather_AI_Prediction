from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from .datasets import infer_dims


def latitude_weights(da: xr.DataArray) -> xr.DataArray:
    dims = infer_dims(da)
    lat_name = dims["lat"]
    if lat_name is None:
        raise ValueError("No latitude coordinate found.")
    lat = da[lat_name]
    w = np.cos(np.deg2rad(lat.astype(float)))
    return w / w.mean()


def _spatial_dims(da: xr.DataArray) -> list[str]:
    dims = infer_dims(da)
    out = [dims["lat"], dims["lon"]]
    return [d for d in out if d is not None]


def weighted_rmse(forecast: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
    w = latitude_weights(truth)
    diff2 = (forecast - truth) ** 2
    return np.sqrt(diff2.weighted(w).mean(dim=_spatial_dims(truth)))


def weighted_mae(forecast: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
    w = latitude_weights(truth)
    return abs(forecast - truth).weighted(w).mean(dim=_spatial_dims(truth))


def weighted_bias(forecast: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
    w = latitude_weights(truth)
    return (forecast - truth).weighted(w).mean(dim=_spatial_dims(truth))


def anomaly_correlation(forecast: xr.DataArray, truth: xr.DataArray, climatology: xr.DataArray | None = None) -> xr.DataArray:
    dims = infer_dims(truth)
    if climatology is None:
        if "valid_time" in truth.dims:
            climatology = truth.mean("valid_time")
        else:
            climatology = truth.mean(dims["time"])
    fa = forecast - climatology
    ta = truth - climatology
    w = latitude_weights(truth)
    spatial = _spatial_dims(truth)
    num = (fa * ta).weighted(w).mean(dim=spatial)
    den = np.sqrt((fa ** 2).weighted(w).mean(dim=spatial) * (ta ** 2).weighted(w).mean(dim=spatial))
    return num / den


def crps_ensemble(ensemble_forecast: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
    dims = infer_dims(ensemble_forecast)
    member = dims["member"]
    if member is None:
        raise ValueError("No ensemble/member dimension found for CRPS.")
    x = ensemble_forecast
    m1 = abs(x - truth).mean(dim=member)
    x1 = x.rename({member: f"{member}_1"})
    x2 = x.rename({member: f"{member}_2"})
    pair = abs(x1 - x2).mean(dim=[f"{member}_1", f"{member}_2"])
    w = latitude_weights(truth)
    crps_field = m1 - 0.5 * pair
    return crps_field.weighted(w).mean(dim=_spatial_dims(truth))


def spread_skill_ratio(ensemble_forecast: xr.DataArray, truth: xr.DataArray) -> xr.DataArray:
    dims = infer_dims(ensemble_forecast)
    member = dims["member"]
    if member is None:
        raise ValueError("No ensemble/member dimension found.")
    ens_mean = ensemble_forecast.mean(member)
    ens_std = ensemble_forecast.std(member)
    w = latitude_weights(truth)
    spread = ens_std.weighted(w).mean(dim=_spatial_dims(truth))
    skill = weighted_rmse(ens_mean, truth)
    return spread / skill


def brier_score_exceedance(ensemble_forecast: xr.DataArray, truth: xr.DataArray, threshold: float) -> xr.DataArray:
    dims = infer_dims(ensemble_forecast)
    member = dims["member"]
    if member is None:
        raise ValueError("No ensemble/member dimension found.")
    p = (ensemble_forecast > threshold).mean(member)
    o = (truth > threshold).astype(float)
    bs_field = (p - o) ** 2
    w = latitude_weights(truth)
    return bs_field.weighted(w).mean(dim=_spatial_dims(truth))


def rank_histogram_counts(ensemble_forecast: xr.DataArray, truth: xr.DataArray) -> np.ndarray:
    dims = infer_dims(ensemble_forecast)
    member = dims["member"]
    if member is None:
        raise ValueError("No ensemble/member dimension found.")
    x = ensemble_forecast.transpose(..., member)
    m = x.sizes[member]
    flat_x = x.stack(sample=tuple([d for d in x.dims if d != member])).transpose("sample", member).values
    flat_t = truth.stack(sample=truth.dims).values
    mask = np.isfinite(flat_t) & np.all(np.isfinite(flat_x), axis=1)
    flat_x = flat_x[mask]
    flat_t = flat_t[mask]
    ranks = np.sum(flat_x < flat_t[:, None], axis=1)
    counts = np.bincount(ranks, minlength=m + 1)
    return counts


def threshold_reliability_table(ensemble_forecast: xr.DataArray, truth: xr.DataArray, threshold: float, n_bins: int = 10) -> pd.DataFrame:
    dims = infer_dims(ensemble_forecast)
    member = dims["member"]
    if member is None:
        raise ValueError("No ensemble/member dimension found.")
    p = (ensemble_forecast > threshold).mean(member)
    o = (truth > threshold).astype(float)
    flat_p = p.stack(sample=p.dims).values
    flat_o = o.stack(sample=o.dims).values
    mask = np.isfinite(flat_p) & np.isfinite(flat_o)
    flat_p = flat_p[mask]
    flat_o = flat_o[mask]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(flat_p, bins, right=True)
    rows = []
    for b in range(1, len(bins) + 1):
        sel = idx == b
        if not np.any(sel):
            continue
        rows.append({
            "bin": b,
            "forecast_prob_mean": float(np.mean(flat_p[sel])),
            "observed_freq": float(np.mean(flat_o[sel])),
            "count": int(np.sum(sel)),
            "threshold": float(threshold),
        })
    return pd.DataFrame(rows)
