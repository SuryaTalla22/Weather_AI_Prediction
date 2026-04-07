from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flagship_predictability import (
    infer_dims,
    weighted_rmse,
    anomaly_correlation,
    build_regime_labels,
    lead_spectral_rmse,
    lagged_growth_curve,
    tibaldi_blocking_mask,
)


def make_truth_forecast():
    time = pd.date_range("2020-01-01", periods=8, freq="6h")
    lead = pd.to_timedelta([24, 72], unit="h")
    lat = np.linspace(-90, 90, 17)
    lon = np.linspace(0, 357.5, 32)
    tt, yy, xx = np.meshgrid(np.arange(len(time)), lat, lon, indexing="ij")
    truth = xr.DataArray(
        np.sin(np.deg2rad(yy)) + 0.1 * np.cos(np.deg2rad(xx)) + 0.01 * tt,
        dims=["time", "latitude", "longitude"],
        coords={"time": time, "latitude": lat, "longitude": lon},
        name="z",
    )
    fc = xr.concat([
        truth + 0.05,
        truth + 0.10,
    ], dim=xr.DataArray(lead.values, dims=["prediction_timedelta"], name="prediction_timedelta"))
    fc = fc.transpose("time", "prediction_timedelta", "latitude", "longitude")
    return truth, fc


if __name__ == "__main__":
    truth, fc = make_truth_forecast()
    assert infer_dims(truth)["lat"] == "latitude"
    from flagship_predictability.datasets import align_forecast_truth_at_lead
    a, t = align_forecast_truth_at_lead(fc, truth, np.timedelta64(24, "h"))
    rmse = weighted_rmse(a, t)
    acc = anomaly_correlation(a, t)
    assert np.isfinite(float(rmse.mean()))
    assert np.isfinite(float(acc.mean()))
    bundle = build_regime_labels(truth - truth.mean("time"), n_components=3, n_regimes=2)
    assert bundle["labels"].size == truth.sizes["time"]
    srmse = lead_spectral_rmse(a, t)
    assert srmse.size > 0
    lg = lagged_growth_curve(fc, np.timedelta64(24, "h"), lag="6h")
    assert lg.size == truth.sizes["time"]
    bm = tibaldi_blocking_mask(t)
    assert "valid_time" in bm.dims
    print("smoke_test passed")
