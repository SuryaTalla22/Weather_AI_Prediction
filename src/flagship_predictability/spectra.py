
from __future__ import annotations
import numpy as np
import xarray as xr
from .datasets import infer_dims

def _isotropic_spectrum_2d(field2d: np.ndarray):
    ny, nx = field2d.shape
    arr = np.asarray(field2d)
    arr = arr - np.nanmean(arr)
    arr = np.nan_to_num(arr, nan=0.0)
    spec2 = np.abs(np.fft.rfft2(arr)) ** 2
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.rfftfreq(nx) * nx
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX**2 + KY**2)
    bins = np.arange(0, int(np.floor(kr.max())) + 2)
    out = np.zeros(len(bins) - 1)
    kval = np.zeros(len(bins) - 1)
    for i in range(len(out)):
        mask = (kr >= bins[i]) & (kr < bins[i+1])
        if np.any(mask):
            out[i] = spec2[mask].mean()
            kval[i] = 0.5 * (bins[i] + bins[i+1])
        else:
            out[i] = np.nan
            kval[i] = 0.5 * (bins[i] + bins[i+1])
    return kval, out

def mean_isotropic_spectrum(da: xr.DataArray, sample_dim: str | None = None) -> xr.Dataset:
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"
    if sample_dim is None:
        sample_dim = "valid_time" if "valid_time" in da.dims else "time"
    spectra = []
    kval_ref = None
    for i in range(da.sizes[sample_dim]):
        fld = da.isel({sample_dim: i}).transpose(lat_name, lon_name).values
        k, p = _isotropic_spectrum_2d(fld)
        kval_ref = k
        spectra.append(p)
    spectra = np.asarray(spectra)
    return xr.Dataset(
        data_vars={
            "spectrum": (("sample", "wavenumber"), spectra),
            "mean_spectrum": (("wavenumber",), np.nanmean(spectra, axis=0)),
        },
        coords={"sample": np.arange(spectra.shape[0]), "wavenumber": kval_ref},
    )

def lead_spectral_rmse(forecast: xr.DataArray, truth: xr.DataArray, sample_dim: str = "valid_time") -> xr.DataArray:
    lat_name = "latitude" if "latitude" in truth.dims else "lat"
    lon_name = "longitude" if "longitude" in truth.dims else "lon"
    vals = []
    kval_ref = None
    for i in range(truth.sizes[sample_dim]):
        err = (forecast.isel({sample_dim: i}) - truth.isel({sample_dim: i})).transpose(lat_name, lon_name).values
        k, p = _isotropic_spectrum_2d(err)
        kval_ref = k
        vals.append(np.sqrt(p))
    vals = np.asarray(vals)
    return xr.DataArray(np.nanmean(vals, axis=0), dims=["wavenumber"], coords={"wavenumber": kval_ref}, name="spectral_rmse")

def spectral_retention(forecast: xr.DataArray, truth: xr.DataArray, sample_dim: str = "valid_time") -> xr.DataArray:
    fs = mean_isotropic_spectrum(forecast, sample_dim=sample_dim)["mean_spectrum"]
    ts = mean_isotropic_spectrum(truth, sample_dim=sample_dim)["mean_spectrum"]
    return (fs / ts).rename("spectral_retention")

def divergence_vorticity_fields(
    u: xr.DataArray,
    v: xr.DataArray,
    pole_cos_threshold: float = 1e-3,
    drop_edge_rows: int = 1,
):
    """
    Compute spherical horizontal divergence and relative vorticity robustly on a latitude-longitude grid.

    Fixes two issues that can spoil publication-quality diagnostics:
    1) the pole singularity from division by cos(phi) / tan(phi);
    2) inadvertent conversion of latitude coordinates from degrees to radians, which breaks later weighting.
    """
    dims = infer_dims(u)
    lat_name, lon_name = dims["lat"], dims["lon"]

    if lat_name is None or lon_name is None:
        raise ValueError("Could not infer latitude/longitude dimensions.")

    # Exact alignment and monotonic coordinates
    u, v = xr.align(u, v, join="exact")
    u = u.sortby(lat_name).sortby(lon_name)
    v = v.sortby(lat_name).sortby(lon_name)

    if u.sizes[lat_name] < 5 or u.sizes[lon_name] < 3:
        raise ValueError(
            f"Need at least 5 latitude points and 3 longitude points to compute robust derivatives. "
            f"Got {u.sizes[lat_name]} x {u.sizes[lon_name]}."
        )

    lat_deg = xr.DataArray(u[lat_name].astype(float).values, dims=(lat_name,), coords={lat_name: u[lat_name].values})
    lon_deg = xr.DataArray(u[lon_name].astype(float).values, dims=(lon_name,), coords={lon_name: u[lon_name].values})

    # Mask pole rows where spherical metric terms become singular.
    cosphi_deg = np.cos(np.deg2rad(lat_deg))
    keep = np.abs(cosphi_deg) > pole_cos_threshold
    if int(keep.sum()) < 5:
        raise ValueError("Too few interior latitude rows remain after masking poles.")

    u = u.sel({lat_name: lat_deg[keep].values})
    v = v.sel({lat_name: lat_deg[keep].values})
    lat_deg = lat_deg.sel({lat_name: lat_deg[keep].values})

    # Optionally drop one more row at each edge to stabilize edge-order-2 derivatives.
    if drop_edge_rows > 0:
        if u.sizes[lat_name] <= 2 * drop_edge_rows + 2:
            raise ValueError("Too few latitude rows remain after edge trimming.")
        interior_vals = lat_deg.values[drop_edge_rows:-drop_edge_rows]
        u = u.sel({lat_name: interior_vals})
        v = v.sel({lat_name: interior_vals})
        lat_deg = lat_deg.sel({lat_name: interior_vals})

    # Preserve degree coordinates for downstream weighting, but differentiate in radians.
    lat_rad_vals = np.deg2rad(lat_deg.values.astype(float))
    lon_rad_vals = np.deg2rad(lon_deg.values.astype(float))

    u_work = u.assign_coords({lat_name: lat_rad_vals, lon_name: lon_rad_vals})
    v_work = v.assign_coords({lat_name: lat_rad_vals, lon_name: lon_rad_vals})

    a = 6_371_000.0
    cosphi = np.cos(u_work[lat_name])
    tanphi = np.tan(u_work[lat_name])

    # Avoid tiny metric denominators in any residual near-polar rows.
    cosphi_safe = xr.where(np.abs(cosphi) < pole_cos_threshold, np.nan, cosphi)

    dudlam = u_work.differentiate(lon_name, edge_order=2)
    dudphi = u_work.differentiate(lat_name, edge_order=2)
    dvdlam = v_work.differentiate(lon_name, edge_order=2)
    dvdphi = v_work.differentiate(lat_name, edge_order=2)

    div = (
        dudlam / (a * cosphi_safe)
        + dvdphi / a
        - v_work * tanphi / a
    ).rename("divergence")

    vort = (
        dvdlam / (a * cosphi_safe)
        - dudphi / a
        + u_work * tanphi / a
    ).rename("vorticity")

    # Restore degree coordinates so later latitude weighting is correct.
    div = div.assign_coords({lat_name: lat_deg.values, lon_name: lon_deg.values})
    vort = vort.assign_coords({lat_name: lat_deg.values, lon_name: lon_deg.values})

    return div, vort
