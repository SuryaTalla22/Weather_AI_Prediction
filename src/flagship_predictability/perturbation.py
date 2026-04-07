from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from .datasets import infer_dims
from .metrics import latitude_weights


def _spatial_norm(da: xr.DataArray) -> xr.DataArray:
    dims = infer_dims(da)
    lat_name = dims["lat"]
    lon_name = dims["lon"]
    w = latitude_weights(da)
    return np.sqrt(((da) ** 2).weighted(w).mean(dim=[lat_name, lon_name]))


def lagged_growth_curve(forecast_da: xr.DataArray, lead, lag: str = "6h") -> xr.DataArray:
    dims = infer_dims(forecast_da)
    tdim = dims["time"]
    lead_dim = dims["lead"]
    if tdim is None or lead_dim is None:
        raise ValueError("Need time and lead dimensions for lagged growth.")
    lead_td = pd.to_timedelta(lead).to_timedelta64()
    f0 = forecast_da.sel({lead_dim: lead_td})
    coords = pd.to_datetime(f0[tdim].values)
    step_hours = np.median(np.diff(coords).astype("timedelta64[h]").astype(int)) if len(coords) > 1 else 6
    step = max(1, int(pd.to_timedelta(lag) / pd.to_timedelta(f"{step_hours}h")))
    f1 = f0.shift({tdim: step})
    delta = f0 - f1
    out = _spatial_norm(delta).rename("lagged_growth")
    return out


def forecast_error_curve(forecast_at_lead: xr.DataArray, truth_at_valid: xr.DataArray) -> xr.DataArray:
    return _spatial_norm(forecast_at_lead - truth_at_valid).rename("forecast_error")


def ensemble_growth_curve(ensemble_da: xr.DataArray, lead) -> xr.DataArray:
    dims = infer_dims(ensemble_da)
    member = dims["member"]
    lead_dim = dims["lead"]
    if member is None or lead_dim is None:
        raise ValueError("Need member and lead dimensions for ensemble growth.")
    lead_td = pd.to_timedelta(lead).to_timedelta64()
    x = ensemble_da.sel({lead_dim: lead_td})
    centered = x - x.mean(member)
    return _spatial_norm(centered).mean(member).rename("ensemble_growth")


def fsle_threshold_times(curve: xr.DataArray, thresholds=(1.5, 2.0, 3.0)):
    vals = np.asarray(curve.values, dtype=float)
    if len(vals) == 0 or not np.isfinite(vals[0]):
        return {f"x{thr}": np.nan for thr in thresholds}
    base = vals[0]
    out = {}
    for thr in thresholds:
        target = base * thr
        idx = np.where(vals >= target)[0]
        out[f"x{thr}"] = np.nan if len(idx) == 0 else float(idx[0])
    return out
