from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon")
TIME_NAMES = ("time", "init_time", "date")
LEAD_NAMES = ("prediction_timedelta", "lead_time", "step", "forecast_hour")
LEVEL_NAMES = ("level", "pressure_level", "isobaricInhPa")
MEMBER_NAMES = ("number", "member", "realization", "ensemble", "sample")


def open_local_zarr(path: str | Path, chunks="auto") -> xr.Dataset:
    path = str(path)
    errors = []
    for consolidated in (True, False):
        try:
            return xr.open_zarr(
                path,
                consolidated=consolidated,
                chunks=chunks,
                decode_timedelta=True,
            )
        except Exception as e:
            errors.append(f"consolidated={consolidated}: {e}")
    raise RuntimeError(f"Unable to open zarr store {path}\n" + "\n".join(errors))


def infer_dims(obj: xr.Dataset | xr.DataArray) -> dict[str, str | None]:
    dims = list(obj.dims)
    coords = set(obj.coords)

    def pick(cands):
        for c in cands:
            if c in dims or c in coords:
                return c
        return None

    return {
        "lat": pick(LAT_NAMES),
        "lon": pick(LON_NAMES),
        "time": pick(TIME_NAMES),
        "lead": pick(LEAD_NAMES),
        "level": pick(LEVEL_NAMES),
        "member": pick(MEMBER_NAMES),
    }


def maybe_pick_var(ds: xr.Dataset, candidates: list[str]) -> xr.DataArray | None:
    for c in candidates:
        if c in ds:
            return ds[c]
    return None


def pick_var(ds: xr.Dataset, candidates: list[str]) -> xr.DataArray:
    out = maybe_pick_var(ds, candidates)
    if out is None:
        raise KeyError(f"None of the variable candidates were found: {candidates}")
    return out


def maybe_select_level(da: xr.DataArray, level_value=None) -> xr.DataArray:
    dims = infer_dims(da)
    level_dim = dims["level"]
    if level_dim and level_value is not None and level_dim in da.coords:
        try:
            return da.sel({level_dim: level_value})
        except Exception:
            coord = da[level_dim]
            idx = int(np.argmin(np.abs(coord.astype(float) - float(level_value))))
            return da.isel({level_dim: idx})
    return da


def subset_date(da: xr.DataArray, date_window: tuple[str, str]) -> xr.DataArray:
    dims = infer_dims(da)
    tdim = dims["time"]
    if tdim is None:
        return da
    return da.sel({tdim: slice(date_window[0], date_window[1])})


def standardize_longitudes(da: xr.DataArray | xr.Dataset, to: str = "0_360"):
    dims = infer_dims(da)
    lon_name = dims["lon"]
    if lon_name is None:
        return da
    lon = xr.DataArray(da[lon_name].values, dims=(lon_name,))
    if to == "0_360":
        new_lon = (lon % 360.0)
    elif to == "-180_180":
        new_lon = ((lon + 180.0) % 360.0) - 180.0
    else:
        raise ValueError("to must be '0_360' or '-180_180'")
    out = da.assign_coords({lon_name: new_lon}).sortby(lon_name)
    return out


def _to_timedelta64(lead):
    if isinstance(lead, np.timedelta64):
        return lead
    if isinstance(lead, pd.Timedelta):
        return lead.to_timedelta64()
    return np.timedelta64(pd.to_timedelta(lead))


def available_leads(obj: xr.Dataset | xr.DataArray) -> list[np.timedelta64]:
    dims = infer_dims(obj)
    lead_dim = dims["lead"]
    if lead_dim is None:
        return []
    vals = obj[lead_dim].values
    out = []
    for v in vals:
        out.append(_to_timedelta64(v))
    return out


def has_member_dim(obj: xr.Dataset | xr.DataArray) -> bool:
    return infer_dims(obj)["member"] is not None


def compute_valid_time_coord(fc: xr.DataArray, init_dim: str, lead64: np.timedelta64) -> xr.DataArray:
    vals = pd.to_datetime(fc[init_dim].values) + pd.to_timedelta(lead64)
    return xr.DataArray(vals.values, dims=(init_dim,), coords={init_dim: fc[init_dim].values}, name="valid_time")


def align_forecast_truth_at_lead(
    forecast_da: xr.DataArray,
    truth_da: xr.DataArray,
    lead,
    join: str = "inner",
) -> tuple[xr.DataArray, xr.DataArray]:
    fdims = infer_dims(forecast_da)
    tdims = infer_dims(truth_da)
    time_dim_f = fdims["time"]
    time_dim_t = tdims["time"]
    lead_dim = fdims["lead"]
    if not (time_dim_f and time_dim_t and lead_dim):
        raise ValueError("Could not infer time/lead dimensions for alignment.")
    lead64 = _to_timedelta64(lead)
    fc = forecast_da.sel({lead_dim: lead64})
    valid_time = compute_valid_time_coord(fc, time_dim_f, lead64)
    fc = fc.assign_coords(valid_time=valid_time).swap_dims({time_dim_f: "valid_time"})
    if time_dim_f in fc.coords:
        try:
            fc = fc.drop_vars(time_dim_f)
        except Exception:
            pass
    truth = truth_da.rename({time_dim_t: "valid_time"})
    fc, truth = xr.align(fc, truth, join=join)
    return fc, truth


def dataset_summary(ds: xr.Dataset, dataset_name: str) -> pd.DataFrame:
    rows = []
    dims = infer_dims(ds)
    for var_name, da in ds.data_vars.items():
        row = {
            "dataset": dataset_name,
            "variable": var_name,
            "dtype": str(da.dtype),
            "dims": ",".join(map(str, da.dims)),
            "shape": "x".join(map(str, da.shape)),
            "lat_dim": dims["lat"],
            "lon_dim": dims["lon"],
            "time_dim": infer_dims(da)["time"],
            "lead_dim": infer_dims(da)["lead"],
            "level_dim": infer_dims(da)["level"],
            "member_dim": infer_dims(da)["member"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_summary(ds: xr.Dataset, dataset_name: str) -> pd.DataFrame:
    dims = infer_dims(ds)
    rows = []
    row = {
        "dataset": dataset_name,
        "n_vars": len(ds.data_vars),
        "lat_dim": dims["lat"],
        "lon_dim": dims["lon"],
        "time_dim": dims["time"],
        "lead_dim": dims["lead"],
        "level_dim": dims["level"],
        "member_dim": dims["member"],
    }
    if dims["time"]:
        t = pd.to_datetime(ds[dims["time"]].values)
        row["time_start"] = str(t.min()) if len(t) else None
        row["time_end"] = str(t.max()) if len(t) else None
        row["n_times"] = len(t)
    if dims["lead"]:
        row["n_leads"] = int(ds[dims["lead"]].size)
    if dims["level"]:
        row["n_levels"] = int(ds[dims["level"]].size)
    if dims["member"]:
        row["n_members"] = int(ds[dims["member"]].size)
    rows.append(row)
    return pd.DataFrame(rows)


def alignment_audit(forecast_da: xr.DataArray, truth_da: xr.DataArray, leads: list[np.timedelta64]) -> pd.DataFrame:
    rows = []
    for lead in leads:
        try:
            fc, tr = align_forecast_truth_at_lead(forecast_da, truth_da, lead)
            rows.append({
                "lead_hours": int(pd.to_timedelta(lead).total_seconds() / 3600),
                "n_valid_times": int(fc.sizes.get("valid_time", 0)),
                "forecast_shape": "x".join(map(str, fc.shape)),
                "truth_shape": "x".join(map(str, tr.shape)),
                "status": "ok",
            })
        except Exception as e:
            rows.append({
                "lead_hours": int(pd.to_timedelta(lead).total_seconds() / 3600),
                "n_valid_times": 0,
                "forecast_shape": None,
                "truth_shape": None,
                "status": f"error: {e}",
            })
    return pd.DataFrame(rows)
