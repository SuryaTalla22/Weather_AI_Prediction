from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from .config import AtlasConfig, VariableSpec
from .wb2_paths import resolve_dataset_path, resolve_output_root
from .datasets import (
    open_local_zarr, pick_var, maybe_pick_var, maybe_select_level, subset_date,
    align_forecast_truth_at_lead, dataset_summary, coverage_summary, alignment_audit,
    infer_dims,
)
from .metrics import (
    weighted_rmse, weighted_mae, weighted_bias, anomaly_correlation,
    crps_ensemble, spread_skill_ratio, rank_histogram_counts,
    threshold_reliability_table, brier_score_exceedance,
)
from .regimes import (
    build_regime_labels, regime_metric_dataframe_with_ci, regime_sensitivity_table,
)
from .spectra import mean_isotropic_spectrum, lead_spectral_rmse, spectral_retention, divergence_vorticity_fields
from .perturbation import lagged_growth_curve, forecast_error_curve, ensemble_growth_curve, fsle_threshold_times
from .blocking import tibaldi_blocking_mask, sector_blocking_series, event_table_from_binary_series


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_df(df: pd.DataFrame, path: Path):
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)


def _save_json(obj, path: Path):
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _window_tag(window: tuple[str, str]) -> str:
    return f"{window[0]}_{window[1]}".replace(":", "-")




@lru_cache(maxsize=16)
def _open_dataset_cached(dataset_name: str):
    return open_local_zarr(resolve_dataset_path(dataset_name))

def _load_dataarray(dataset_name: str, spec: VariableSpec, window: tuple[str, str]) -> xr.DataArray:
    ds = _open_dataset_cached(dataset_name)
    try:
        da = maybe_select_level(pick_var(ds, spec.candidates), spec.level)
    except KeyError as e:
        available = sorted(list(ds.data_vars))
        raise KeyError(
            f"Dataset '{dataset_name}' is missing variable candidates {spec.candidates} "
            f"for VariableSpec(name='{spec.name}', level={spec.level}). "
            f"Available variables: {available}"
        ) from e
    return subset_date(da, window)


def run_dataset_audit(cfg: AtlasConfig, output_root: Path | None = None) -> dict[str, Path]:
    outroot = _ensure_dir((resolve_output_root() if output_root is None else Path(output_root)) / "audit")
    all_names = {cfg.truth_dataset, *cfg.deterministic_models.values(), *cfg.ensemble_models.values()}
    ds_rows = []
    cov_rows = []
    align_rows = []
    for ds_name in sorted(all_names):
        ds = _open_dataset_cached(ds_name)
        ds_rows.append(dataset_summary(ds, ds_name))
        cov_rows.append(coverage_summary(ds, ds_name))
        for var_name, spec in cfg.variables.items():
            da = maybe_pick_var(ds, spec.candidates)
            if da is None:
                continue
            da = maybe_select_level(da, spec.level)
            if ds_name != cfg.truth_dataset:
                tr = _load_dataarray(cfg.truth_dataset, spec, cfg.date_windows[0])
                align = alignment_audit(subset_date(da, cfg.date_windows[0]), tr, cfg.leads)
                align.insert(0, "dataset", ds_name)
                align.insert(1, "variable_key", var_name)
                align_rows.append(align)
    ds_df = pd.concat(ds_rows, ignore_index=True) if ds_rows else pd.DataFrame()
    cov_df = pd.concat(cov_rows, ignore_index=True) if cov_rows else pd.DataFrame()
    align_df = pd.concat(align_rows, ignore_index=True) if align_rows else pd.DataFrame()
    _save_df(ds_df, outroot / "dataset_variables.csv")
    _save_df(cov_df, outroot / "coverage_summary.csv")
    _save_df(align_df, outroot / "alignment_audit.csv")
    return {
        "dataset_variables": outroot / "dataset_variables.csv",
        "coverage_summary": outroot / "coverage_summary.csv",
        "alignment_audit": outroot / "alignment_audit.csv",
    }


def run_truth_regimes(cfg: AtlasConfig, output_root: Path | None = None) -> dict[str, Path]:
    outroot = _ensure_dir((resolve_output_root() if output_root is None else Path(output_root)) / "regimes")
    zspec = cfg.variables.get("z500") or next(iter(cfg.variables.values()))
    frames = []
    for window in cfg.date_windows:
        truth = _load_dataarray(cfg.truth_dataset, zspec, window)
        truth_anom = truth - truth.mean(infer_dims(truth)["time"])
        bundle = build_regime_labels(truth_anom, n_components=cfg.n_eof_components, n_regimes=cfg.n_regimes, random_state=0)
        labels = bundle["labels"]
        tag = _window_tag(window)
        labels.to_netcdf(outroot / f"regime_labels_{tag}.nc")
        frame = pd.DataFrame({
            "time": pd.to_datetime(labels[labels.dims[0]].values),
            "regime": labels.values.astype(int),
            "window": tag,
        })
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    _save_df(df, outroot / "regime_labels.csv")
    return {"regime_labels_csv": outroot / "regime_labels.csv"}


def run_regime_sensitivity(cfg: AtlasConfig, output_root: Path | None = None, component_grid: list[int] | None = None, regime_grid: list[int] | None = None) -> dict[str, Path]:
    outroot = _ensure_dir((resolve_output_root() if output_root is None else Path(output_root)) / "regimes")
    component_grid = component_grid or [6, 8, 10, 12]
    regime_grid = regime_grid or [3, 4, 5, 6]
    zspec = cfg.variables.get("z500") or next(iter(cfg.variables.values()))
    frames = []
    for window in cfg.date_windows:
        truth = _load_dataarray(cfg.truth_dataset, zspec, window)
        truth_anom = truth - truth.mean(infer_dims(truth)["time"])
        df = regime_sensitivity_table(truth_anom, component_grid=component_grid, regime_grid=regime_grid, random_state=0)
        df.insert(0, "window", _window_tag(window))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    _save_df(out, outroot / "regime_sensitivity.csv")
    return {"regime_sensitivity": outroot / "regime_sensitivity.csv"}


def run_deterministic_atlas(cfg: AtlasConfig, output_root: Path | None = None) -> dict[str, Path]:
    outroot = _ensure_dir((resolve_output_root() if output_root is None else Path(output_root)) / "deterministic")
    summary_rows = []
    regime_rows = []
    spectral_rows = []
    balance_rows = []
    error_rows = []
    availability_rows = []
    for window in cfg.date_windows:
        tag = _window_tag(window)
        zspec = cfg.variables.get("z500") or next(iter(cfg.variables.values()))
        truth_z = _load_dataarray(cfg.truth_dataset, zspec, window)
        truth_anom = truth_z - truth_z.mean(infer_dims(truth_z)["time"])
        regime_labels = build_regime_labels(truth_anom, n_components=cfg.n_eof_components, n_regimes=cfg.n_regimes, random_state=0)["labels"]
        rlabels = regime_labels.rename({regime_labels.dims[0]: "valid_time"})
        for vkey, spec in cfg.variables.items():
            try:
                truth = _load_dataarray(cfg.truth_dataset, spec, window)
            except Exception as e:
                error_rows.append({
                    "workflow": "deterministic",
                    "window": tag,
                    "variable": vkey,
                    "model": "TRUTH",
                    "lead": pd.NaT,
                    "error": str(e),
                })
                continue
            rmse_ts = {}
            acc_ts = {}
            mae_ts = {}
            bias_ts = {}
            for model_name, ds_name in cfg.deterministic_models.items():
                try:
                    da = _load_dataarray(ds_name, spec, window)
                    availability_rows.append({
                        "window": tag,
                        "variable": vkey,
                        "model": model_name,
                        "dataset": ds_name,
                        "status": "ok",
                        "message": "",
                    })
                except Exception as e:
                    availability_rows.append({
                        "window": tag,
                        "variable": vkey,
                        "model": model_name,
                        "dataset": ds_name,
                        "status": "missing_variable",
                        "message": str(e),
                    })
                    error_rows.append({
                        "workflow": "deterministic",
                        "window": tag,
                        "variable": vkey,
                        "model": model_name,
                        "lead": pd.NaT,
                        "error": str(e),
                    })
                    continue
                rmse_ts[model_name] = {}
                acc_ts[model_name] = {}
                mae_ts[model_name] = {}
                bias_ts[model_name] = {}
                for lead in cfg.leads:
                    try:
                        fc, tr = align_forecast_truth_at_lead(da, truth, lead)
                    except Exception as e:
                        error_rows.append({
                            "workflow": "deterministic",
                            "window": tag,
                            "variable": vkey,
                            "model": model_name,
                            "lead": pd.to_timedelta(lead),
                            "error": str(e),
                        })
                        continue
                    clim = tr.mean("valid_time")
                    rm = weighted_rmse(fc, tr)
                    ac = anomaly_correlation(fc, tr, climatology=clim)
                    ma = weighted_mae(fc, tr)
                    bi = weighted_bias(fc, tr)
                    rmse_ts[model_name][lead] = rm
                    acc_ts[model_name][lead] = ac
                    mae_ts[model_name][lead] = ma
                    bias_ts[model_name][lead] = bi
                    summary_rows.append({
                        "window": tag,
                        "variable": vkey,
                        "model": model_name,
                        "lead": pd.to_timedelta(lead),
                        "rmse_mean": float(rm.mean().compute()),
                        "acc_mean": float(ac.mean().compute()),
                        "mae_mean": float(ma.mean().compute()),
                        "bias_mean": float(bi.mean().compute()),
                        "n_valid_times": int(fc.sizes.get("valid_time", 0)),
                    })
                    if vkey == "z500":
                        try:
                            ret = spectral_retention(fc, tr)
                            srmse = lead_spectral_rmse(fc, tr)
                            for k, rv, ev in zip(ret["wavenumber"].values, ret.values, srmse.values):
                                spectral_rows.append({
                                    "window": tag,
                                    "variable": vkey,
                                    "model": model_name,
                                    "lead": pd.to_timedelta(lead),
                                    "wavenumber": float(k),
                                    "spectral_retention": float(rv),
                                    "spectral_rmse": float(ev),
                                })
                        except Exception as e:
                            error_rows.append({
                                "workflow": "deterministic_spectral",
                                "window": tag,
                                "variable": vkey,
                                "model": model_name,
                                "lead": pd.to_timedelta(lead),
                                "error": str(e),
                            })
            for metric_name, ts in {
                "RMSE": rmse_ts,
                "ACC": acc_ts,
                "MAE": mae_ts,
                "BIAS": bias_ts,
            }.items():
                nonempty_ts = {k: v for k, v in ts.items() if v}
                if nonempty_ts:
                    try:
                        rdf = regime_metric_dataframe_with_ci(nonempty_ts, rlabels, metric_name, n_boot=cfg.bootstrap_n, block_length=cfg.bootstrap_block)
                        rdf.insert(0, "window", tag)
                        rdf.insert(1, "variable", vkey)
                        regime_rows.append(rdf)
                    except Exception as e:
                        error_rows.append({
                            "workflow": "deterministic_regime_metrics",
                            "window": tag,
                            "variable": vkey,
                            "model": "ALL",
                            "lead": pd.NaT,
                            "error": str(e),
                        })
        if {"u850", "v850"}.issubset(set(cfg.variables)):
            try:
                u_truth = _load_dataarray(cfg.truth_dataset, cfg.variables["u850"], window)
                v_truth = _load_dataarray(cfg.truth_dataset, cfg.variables["v850"], window)
            except Exception as e:
                error_rows.append({
                    "workflow": "deterministic_balance",
                    "window": tag,
                    "variable": "u850,v850",
                    "model": "TRUTH",
                    "lead": pd.NaT,
                    "error": str(e),
                })
                u_truth = None
                v_truth = None
            if u_truth is not None and v_truth is not None:
                for model_name, ds_name in cfg.deterministic_models.items():
                    try:
                        u_fc_all = _load_dataarray(ds_name, cfg.variables["u850"], window)
                        v_fc_all = _load_dataarray(ds_name, cfg.variables["v850"], window)
                    except Exception as e:
                        error_rows.append({
                            "workflow": "deterministic_balance",
                            "window": tag,
                            "variable": "u850,v850",
                            "model": model_name,
                            "lead": pd.NaT,
                            "error": str(e),
                        })
                        continue
                    for lead in cfg.leads:
                        try:
                            uf, ut = align_forecast_truth_at_lead(u_fc_all, u_truth, lead)
                            vf, vt = align_forecast_truth_at_lead(v_fc_all, v_truth, lead)
                            fdiv, fvort = divergence_vorticity_fields(uf, vf)
                            tdiv, tvort = divergence_vorticity_fields(ut, vt)
                            balance_rows.append({
                                "window": tag,
                                "model": model_name,
                                "lead": pd.to_timedelta(lead),
                                "div_rmse_mean": float(weighted_rmse(fdiv, tdiv).mean().compute()),
                                "vort_rmse_mean": float(weighted_rmse(fvort, tvort).mean().compute()),
                            })
                        except Exception as e:
                            error_rows.append({
                                "workflow": "deterministic_balance",
                                "window": tag,
                                "variable": "u850,v850",
                                "model": model_name,
                                "lead": pd.to_timedelta(lead),
                                "error": str(e),
                            })
                            continue
    summary_df = pd.DataFrame(summary_rows)
    regime_df = pd.concat(regime_rows, ignore_index=True) if regime_rows else pd.DataFrame()
    spectral_df = pd.DataFrame(spectral_rows)
    balance_df = pd.DataFrame(balance_rows)
    errors_df = pd.DataFrame(error_rows)
    availability_df = pd.DataFrame(availability_rows)
    _save_df(summary_df, outroot / "deterministic_summary.csv")
    _save_df(regime_df, outroot / "regime_conditioned_metrics.csv")
    _save_df(spectral_df, outroot / "spectral_diagnostics.csv")
    _save_df(balance_df, outroot / "balance_diagnostics.csv")
    _save_df(errors_df, outroot / "deterministic_errors.csv")
    _save_df(availability_df, outroot / "variable_availability.csv")
    return {
        "deterministic_summary": outroot / "deterministic_summary.csv",
        "regime_conditioned_metrics": outroot / "regime_conditioned_metrics.csv",
        "spectral_diagnostics": outroot / "spectral_diagnostics.csv",
        "balance_diagnostics": outroot / "balance_diagnostics.csv",
        "deterministic_errors": outroot / "deterministic_errors.csv",
        "variable_availability": outroot / "variable_availability.csv",
    }


def run_growth_diagnostics(cfg: AtlasConfig, output_root: Path | None = None, thresholds=(1.25, 1.5, 2.0)) -> dict[str, Path]:
    outroot = _ensure_dir((resolve_output_root() if output_root is None else Path(output_root)) / "growth")
    rows = []
    fsle_rows = []
    error_rows = []
    zspec = cfg.variables.get("z500") or next(iter(cfg.variables.values()))
    for window in cfg.date_windows:
        tag = _window_tag(window)
        truth = _load_dataarray(cfg.truth_dataset, zspec, window)
        truth_anom = truth - truth.mean(infer_dims(truth)["time"])
        regime_labels = build_regime_labels(truth_anom, n_components=cfg.n_eof_components, n_regimes=cfg.n_regimes, random_state=0)["labels"]
        rlabels = regime_labels.rename({regime_labels.dims[0]: "valid_time"})
        for model_name, ds_name in cfg.deterministic_models.items():
            da = _load_dataarray(ds_name, zspec, window)
            lagged_vals = []
            error_vals = []
            for lead in cfg.leads:
                try:
                    lg = lagged_growth_curve(da, lead=lead, lag=cfg.lag)
                    lg_time_dim = infer_dims(lg)["time"]
                    if "valid_time" not in lg.dims and lg_time_dim is not None:
                        lead_td = pd.to_timedelta(lead)
                        valid_time = xr.DataArray(
                            (pd.to_datetime(lg[lg_time_dim].values) + lead_td).values,
                            dims=(lg_time_dim,),
                            coords={lg_time_dim: lg[lg_time_dim].values},
                            name="valid_time",
                        )
                        lg = lg.assign_coords(valid_time=valid_time).swap_dims({lg_time_dim: "valid_time"})
                        if lg_time_dim in lg.coords:
                            try:
                                lg = lg.drop_vars(lg_time_dim)
                            except Exception:
                                pass
                    fc, tr = align_forecast_truth_at_lead(da, truth, lead)
                    err = forecast_error_curve(fc, tr)
                    common = np.intersect1d(lg["valid_time"].values, err["valid_time"].values)
                    common = np.intersect1d(common, rlabels["valid_time"].values)
                    if len(common) == 0:
                        raise ValueError("No common valid_time values across lagged growth, forecast error, and regime labels.")
                    g = lg.sel(valid_time=common)
                    e = err.sel(valid_time=common)
                    rr = rlabels.sel(valid_time=common)
                    lagged_vals.append(float(g.mean().compute()))
                    error_vals.append(float(e.mean().compute()))
                    for regime in np.unique(rr.values):
                        gv = g.where(rr == regime, drop=True).values
                        ev = e.where(rr == regime, drop=True).values
                        rows.append({
                            "window": tag,
                            "model": model_name,
                            "lead": pd.to_timedelta(lead),
                            "regime": int(regime),
                            "mean_lagged_growth": float(np.nanmean(gv)),
                            "mean_error_growth": float(np.nanmean(ev)),
                            "n": int(np.isfinite(gv).sum()),
                        })
                except Exception as e:
                    error_rows.append({
                        "workflow": "growth",
                        "window": tag,
                        "model": model_name,
                        "lead": pd.to_timedelta(lead),
                        "error": str(e),
                    })
                    continue
            if lagged_vals:
                curve = xr.DataArray(np.array(lagged_vals), dims=["lead_index"], coords={"lead_index": np.arange(len(lagged_vals))})
                fsle = fsle_threshold_times(curve, thresholds=thresholds)
                fsle.update({"window": tag, "model": model_name, "curve": "lagged_growth"})
                fsle_rows.append(fsle)
            if error_vals:
                curve = xr.DataArray(np.array(error_vals), dims=["lead_index"], coords={"lead_index": np.arange(len(error_vals))})
                fsle = fsle_threshold_times(curve, thresholds=thresholds)
                fsle.update({"window": tag, "model": model_name, "curve": "forecast_error"})
                fsle_rows.append(fsle)
        for model_name, ds_name in cfg.ensemble_models.items():
            da = _load_dataarray(ds_name, zspec, window)
            for lead in cfg.leads:
                try:
                    eg = ensemble_growth_curve(da, lead=lead)
                    rows.append({
                        "window": tag,
                        "model": model_name,
                        "lead": pd.to_timedelta(lead),
                        "regime": -1,
                        "mean_lagged_growth": np.nan,
                        "mean_error_growth": np.nan,
                        "mean_ensemble_growth": float(eg.mean().compute()),
                        "n": int(eg.size),
                    })
                except Exception as e:
                    error_rows.append({
                        "workflow": "growth",
                        "window": tag,
                        "model": model_name,
                        "lead": pd.to_timedelta(lead),
                        "error": str(e),
                    })
                    continue
    growth_df = pd.DataFrame(rows)
    fsle_df = pd.DataFrame(fsle_rows)
    errors_df = pd.DataFrame(error_rows)
    _save_df(growth_df, outroot / "growth_metrics.csv")
    _save_df(fsle_df, outroot / "growth_threshold_times.csv")
    _save_df(errors_df, outroot / "growth_errors.csv")
    return {
        "growth_metrics": outroot / "growth_metrics.csv",
        "growth_threshold_times": outroot / "growth_threshold_times.csv",
        "growth_errors": outroot / "growth_errors.csv",
    }


def run_blocking_verification(cfg: AtlasConfig, output_root: Path | None = None) -> dict[str, Path]:
    outroot = _ensure_dir((resolve_output_root() if output_root is None else Path(output_root)) / "blocking")
    zspec = cfg.variables.get("z500") or next(iter(cfg.variables.values()))
    event_rows = []
    rmse_rows = []
    truth_summary_rows = []
    threshold_rows = []
    error_rows = []
    threshold_grid = [0.02, 0.05, 0.1, 0.2, 0.5]
    for window in cfg.date_windows:
        tag = _window_tag(window)
        truth = _load_dataarray(cfg.truth_dataset, zspec, window)
        for model_name, ds_name in cfg.deterministic_models.items():
            da = _load_dataarray(ds_name, zspec, window)
            for lead in cfg.leads:
                try:
                    fc, tr = align_forecast_truth_at_lead(da, truth, lead)
                    fc_block = tibaldi_blocking_mask(fc, assume_geopotential=cfg.assume_geopotential)
                    tr_block = tibaldi_blocking_mask(tr, assume_geopotential=cfg.assume_geopotential)
                    fc_frac = sector_blocking_series(fc_block, cfg.blocking_sectors)
                    tr_frac = sector_blocking_series(tr_block, cfg.blocking_sectors)
                    fc_sector = fc_frac > cfg.blocking_threshold
                    tr_sector = tr_frac > cfg.blocking_threshold
                    common_idx = fc_sector.index.intersection(tr_sector.index)
                    for sector in cfg.blocking_sectors:
                        obs_ser = tr_sector.loc[common_idx, sector]
                        fcst_ser = fc_sector.loc[common_idx, sector]
                        metrics = event_table_from_binary_series(obs_ser, fcst_ser)
                        metrics.update({
                            "truth_positive_count": int(obs_ser.sum()),
                            "forecast_positive_count": int(fcst_ser.sum()),
                            "window": tag,
                            "model": model_name,
                            "lead": pd.to_timedelta(lead),
                            "sector": sector,
                        })
                        event_rows.append(metrics)
                        frac_ser = tr_frac.loc[common_idx, sector]
                        truth_summary_rows.append({
                            "window": tag,
                            "model": model_name,
                            "lead": pd.to_timedelta(lead),
                            "sector": sector,
                            "truth_frac_mean": float(frac_ser.mean()),
                            "truth_frac_q90": float(frac_ser.quantile(0.9)),
                            "truth_frac_q95": float(frac_ser.quantile(0.95)),
                            "truth_frac_max": float(frac_ser.max()),
                            "n_valid_times": int(len(frac_ser)),
                        })
                    max_truth_frac = tr_frac.loc[common_idx].max(axis=1)
                    for thr in threshold_grid:
                        threshold_rows.append({
                            "window": tag,
                            "model": model_name,
                            "lead": pd.to_timedelta(lead),
                            "threshold": thr,
                            "n_truth_blocked": int((max_truth_frac > thr).sum()),
                            "fraction_truth_blocked": float((max_truth_frac > thr).mean()),
                        })
                    blocked_series = (max_truth_frac > cfg.blocking_threshold)
                    blocked_days = xr.DataArray(
                        blocked_series.to_numpy(),
                        dims=["valid_time"],
                        coords={"valid_time": tr.sel(valid_time=common_idx)["valid_time"].values},
                    )
                    fc_common = fc.sel(valid_time=common_idx)
                    tr_common = tr.sel(valid_time=common_idx)
                    blocked = fc_common.where(blocked_days, drop=True)
                    blocked_tr = tr_common.where(blocked_days, drop=True)
                    unblocked = fc_common.where(~blocked_days, drop=True)
                    unblocked_tr = tr_common.where(~blocked_days, drop=True)
                    rmse_rows.append({
                        "window": tag,
                        "model": model_name,
                        "lead": pd.to_timedelta(lead),
                        "blocked_rmse": float(weighted_rmse(blocked, blocked_tr).mean().compute()) if blocked.sizes.get("valid_time", 0) else np.nan,
                        "unblocked_rmse": float(weighted_rmse(unblocked, unblocked_tr).mean().compute()) if unblocked.sizes.get("valid_time", 0) else np.nan,
                        "n_blocked": int(blocked_days.sum().item()),
                        "n_unblocked": int((~blocked_days).sum().item()),
                    })
                except Exception as e:
                    error_rows.append({
                        "workflow": "blocking",
                        "window": tag,
                        "model": model_name,
                        "lead": pd.to_timedelta(lead),
                        "error": str(e),
                    })
                    continue
    event_df = pd.DataFrame(event_rows)
    rmse_df = pd.DataFrame(rmse_rows)
    truth_summary_df = pd.DataFrame(truth_summary_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    errors_df = pd.DataFrame(error_rows)
    _save_df(event_df, outroot / "blocking_event_metrics.csv")
    _save_df(rmse_df, outroot / "blocking_rmse.csv")
    _save_df(truth_summary_df, outroot / "blocking_truth_fraction_summary.csv")
    _save_df(threshold_df, outroot / "blocking_threshold_sweep.csv")
    _save_df(errors_df, outroot / "blocking_errors.csv")
    return {
        "blocking_event_metrics": outroot / "blocking_event_metrics.csv",
        "blocking_rmse": outroot / "blocking_rmse.csv",
        "blocking_truth_fraction_summary": outroot / "blocking_truth_fraction_summary.csv",
        "blocking_threshold_sweep": outroot / "blocking_threshold_sweep.csv",
        "blocking_errors": outroot / "blocking_errors.csv",
    }


def run_probabilistic_atlas(cfg: AtlasConfig, output_root: Path | None = None) -> dict[str, Path]:
    outroot = _ensure_dir((resolve_output_root() if output_root is None else Path(output_root)) / "probabilistic")
    rows = []
    rank_rows = []
    reliab_rows = []
    for window in cfg.date_windows:
        tag = _window_tag(window)
        for vkey, spec in cfg.variables.items():
            truth = _load_dataarray(cfg.truth_dataset, spec, window)
            for model_name, ds_name in cfg.ensemble_models.items():
                da = _load_dataarray(ds_name, spec, window)
                if infer_dims(da)["member"] is None:
                    continue
                for lead in cfg.leads:
                    try:
                        fc, tr = align_forecast_truth_at_lead(da, truth, lead)
                        cr = crps_ensemble(fc, tr)
                        ssr = spread_skill_ratio(fc, tr)
                        row = {
                            "window": tag,
                            "variable": vkey,
                            "model": model_name,
                            "lead": pd.to_timedelta(lead),
                            "crps_mean": float(cr.mean().compute()),
                            "spread_skill_mean": float(ssr.mean().compute()),
                            "n_valid_times": int(fc.sizes.get("valid_time", 0)),
                        }
                        rows.append(row)
                        counts = rank_histogram_counts(fc, tr)
                        for rank, count in enumerate(counts):
                            rank_rows.append({
                                "window": tag,
                                "variable": vkey,
                                "model": model_name,
                                "lead": pd.to_timedelta(lead),
                                "rank": int(rank),
                                "count": int(count),
                            })
                        thresholds = spec.thresholds or []
                        if not thresholds:
                            thresholds = [float(tr.quantile(0.9).compute())]
                        for thr in thresholds:
                            bs = brier_score_exceedance(fc, tr, threshold=float(thr))
                            rows[-1][f"brier_thr_{float(thr):g}"] = float(bs.mean().compute())
                            rel = threshold_reliability_table(fc, tr, threshold=float(thr), n_bins=10)
                            if not rel.empty:
                                rel.insert(0, "window", tag)
                                rel.insert(1, "variable", vkey)
                                rel.insert(2, "model", model_name)
                                rel.insert(3, "lead", pd.to_timedelta(lead))
                                reliab_rows.append(rel)
                    except Exception:
                        continue
    summary_df = pd.DataFrame(rows)
    rank_df = pd.DataFrame(rank_rows)
    reliab_df = pd.concat(reliab_rows, ignore_index=True) if reliab_rows else pd.DataFrame()
    _save_df(summary_df, outroot / "probabilistic_summary.csv")
    _save_df(rank_df, outroot / "rank_histograms.csv")
    _save_df(reliab_df, outroot / "reliability_tables.csv")
    return {
        "probabilistic_summary": outroot / "probabilistic_summary.csv",
        "rank_histograms": outroot / "rank_histograms.csv",
        "reliability_tables": outroot / "reliability_tables.csv",
    }
