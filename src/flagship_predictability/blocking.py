from __future__ import annotations
import pandas as pd
import xarray as xr
from .datasets import infer_dims, standardize_longitudes


def _to_height(z_or_phi: xr.DataArray, assume_geopotential: bool = True):
    return z_or_phi / 9.80665 if assume_geopotential else z_or_phi


def tibaldi_blocking_mask(z500: xr.DataArray, south=40.0, center=60.0, north=80.0, assume_geopotential: bool = True) -> xr.DataArray:
    dims = infer_dims(z500)
    lat_name = dims["lat"]
    z = _to_height(z500, assume_geopotential=assume_geopotential)
    z = z.sortby(lat_name)
    zs = z.interp({lat_name: south})
    zc = z.interp({lat_name: center})
    zn = z.interp({lat_name: north})
    ghgs = (zc - zs) / (center - south)
    ghgn = (zn - zc) / (north - center)
    blocked = (ghgs > 0.0) & (ghgn < -10.0)
    return blocked.rename("blocking_mask")


def _select_sector(block_mask: xr.DataArray, lo: float, hi: float) -> xr.DataArray:
    dims = infer_dims(block_mask)
    lon_name = dims["lon"]
    sub = standardize_longitudes(block_mask, to="0_360")
    lo = lo % 360.0
    hi = hi % 360.0
    if lo <= hi:
        return sub.sel({lon_name: slice(lo, hi)})
    left = sub.sel({lon_name: slice(lo, 360.0)})
    right = sub.sel({lon_name: slice(0.0, hi)})
    return xr.concat([left, right], dim=lon_name)


def sector_blocking_series(block_mask: xr.DataArray, sectors: dict[str, tuple[float, float]]):
    dims = infer_dims(block_mask)
    lon_name = dims["lon"]
    tdim = "valid_time" if "valid_time" in block_mask.dims else dims["time"]
    rows = []
    for name, (lo, hi) in sectors.items():
        sub = _select_sector(block_mask, lo, hi)
        frac = sub.mean(lon_name)
        rows.append(frac.to_pandas().rename(name))
    df = pd.concat(rows, axis=1)
    df.index.name = tdim
    return df


def event_table_from_binary_series(obs: pd.Series, fcst: pd.Series):
    obs = obs.astype(bool)
    fcst = fcst.astype(bool)
    hits = int(((obs) & (fcst)).sum())
    misses = int(((obs) & (~fcst)).sum())
    false_alarms = int(((~obs) & (fcst)).sum())
    correct_neg = int(((~obs) & (~fcst)).sum())
    pod = hits / max(hits + misses, 1)
    far = false_alarms / max(hits + false_alarms, 1)
    csi = hits / max(hits + misses + false_alarms, 1)
    bias = (hits + false_alarms) / max(hits + misses, 1)
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_neg,
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "frequency_bias": bias,
    }
