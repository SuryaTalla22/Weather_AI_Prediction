
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .wb2_paths import resolve_output_root


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _lead_days(series: pd.Series) -> pd.Series:
    td = pd.to_timedelta(series)
    return td.dt.total_seconds() / 86400.0


def _set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    })


def _save_figure(fig, outdir: Path, stem: str, manifest: list[dict], caption: str):
    _ensure_dir(outdir)
    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    manifest.append({
        "figure_stem": stem,
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
        "caption": caption,
    })
    plt.close(fig)


def _maybe_manifest_row(skipped: list[dict], stem: str, reason: str):
    skipped.append({"figure_stem": stem, "reason": reason})


def generate_flagship_figures(output_root: Path | None = None, figure_root: Path | None = None) -> dict[str, Path]:
    _set_pub_style()

    base = resolve_output_root() if output_root is None else Path(output_root)
    figroot = _ensure_dir((base / "figures") if figure_root is None else Path(figure_root))
    manifest_rows: list[dict] = []
    skipped_rows: list[dict] = []

    audit_dir = base / "audit"
    regime_dir = base / "regimes"
    det_dir = base / "deterministic"
    growth_dir = base / "growth"
    blocking_dir = base / "blocking"

    # 1) Alignment audit heatmap
    align = _read_csv(audit_dir / "alignment_audit.csv")
    if not align.empty and {"dataset", "variable_key", "lead_hours", "n_valid_times", "status"}.issubset(align.columns):
        ok = align[align["status"] == "ok"].copy()
        if not ok.empty:
            ok["label"] = ok["dataset"] + " | " + ok["variable_key"]
            pivot = ok.pivot_table(index="label", columns="lead_hours", values="n_valid_times", aggfunc="max")
            fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.35 * len(pivot.index))))
            im = ax.imshow(pivot.values, aspect="auto")
            ax.set_title("Alignment coverage by dataset, variable, and lead")
            ax.set_xlabel("Lead (hours)")
            ax.set_ylabel("Dataset | variable")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([str(c) for c in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Valid times")
            _save_figure(fig, figroot, "fig_audit_alignment_heatmap", manifest_rows,
                         "Data-coverage audit showing valid aligned forecast-truth samples by dataset, variable, and lead.")
        else:
            _maybe_manifest_row(skipped_rows, "fig_audit_alignment_heatmap", "No alignment rows with status=ok.")
    else:
        _maybe_manifest_row(skipped_rows, "fig_audit_alignment_heatmap", "alignment_audit.csv missing or incompatible.")

    # 2) Regime fractions
    reg_labels = _read_csv(regime_dir / "regime_labels.csv")
    if not reg_labels.empty and {"window", "regime"}.issubset(reg_labels.columns):
        frac = reg_labels.groupby(["window", "regime"]).size().reset_index(name="count")
        frac["fraction"] = frac["count"] / frac.groupby("window")["count"].transform("sum")
        windows = frac["window"].unique().tolist()
        regs = sorted(frac["regime"].unique())
        fig, axes = plt.subplots(len(windows), 1, figsize=(7, max(3.2, 2.8 * len(windows))), squeeze=False)
        for ax, window in zip(axes.ravel(), windows):
            sub = frac[frac["window"] == window].sort_values("regime")
            ax.bar(sub["regime"].astype(str), sub["fraction"])
            ax.set_ylim(0, max(0.45, sub["fraction"].max() * 1.2))
            ax.set_ylabel("Fraction")
            ax.set_title(f"Truth-regime occupancy: {window}")
        axes.ravel()[-1].set_xlabel("Regime")
        _save_figure(fig, figroot, "fig_regime_fractions", manifest_rows,
                     "Truth-regime occupancy fractions for each evaluation window.")
    else:
        _maybe_manifest_row(skipped_rows, "fig_regime_fractions", "regime_labels.csv missing or incompatible.")

    # 3) Regime sensitivity
    reg_sens = _read_csv(regime_dir / "regime_sensitivity.csv")
    if not reg_sens.empty and {"n_components", "n_regimes", "fraction"}.issubset(reg_sens.columns):
        summary = reg_sens.groupby(["window", "n_components", "n_regimes"])["fraction"].agg(["min", "max"]).reset_index()
        windows = summary["window"].unique().tolist()
        fig, axes = plt.subplots(len(windows), 1, figsize=(8, max(3.2, 3.0 * len(windows))), squeeze=False)
        for ax, window in zip(axes.ravel(), windows):
            sub = summary[summary["window"] == window]
            for nreg in sorted(sub["n_regimes"].unique()):
                ss = sub[sub["n_regimes"] == nreg].sort_values("n_components")
                ax.plot(ss["n_components"], ss["min"], marker="o", label=f"{nreg} regimes: min cluster frac")
                ax.plot(ss["n_components"], ss["max"], marker="s", linestyle="--", label=f"{nreg} regimes: max cluster frac")
            ax.set_title(f"Regime-sensitivity balance diagnostics: {window}")
            ax.set_xlabel("EOF components")
            ax.set_ylabel("Cluster fraction")
            ax.set_ylim(0, 1)
            ax.legend(ncol=2, frameon=False)
        _save_figure(fig, figroot, "fig_regime_sensitivity_balance", manifest_rows,
                     "Sensitivity of regime balance to the number of EOF components and target regime count.")
    else:
        _maybe_manifest_row(skipped_rows, "fig_regime_sensitivity_balance", "regime_sensitivity.csv missing or incompatible.")

    # 4) Deterministic summary: RMSE
    det_summary = _read_csv(det_dir / "deterministic_summary.csv")
    if not det_summary.empty and {"window", "variable", "model", "lead", "rmse_mean"}.issubset(det_summary.columns):
        det_summary["lead_days"] = _lead_days(det_summary["lead"])
        for variable in sorted(det_summary["variable"].unique()):
            sub = det_summary[det_summary["variable"] == variable]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for model in sorted(sub["model"].unique()):
                ss = sub[sub["model"] == model].sort_values("lead_days")
                ax.plot(ss["lead_days"], ss["rmse_mean"], marker="o", label=model)
            ax.set_title(f"{variable.upper()} deterministic RMSE by lead")
            ax.set_xlabel("Lead (days)")
            ax.set_ylabel("Area-weighted RMSE")
            ax.legend(frameon=False)
            _save_figure(fig, figroot, f"fig_rmse_{variable}", manifest_rows,
                         f"Deterministic lead-time RMSE comparison for {variable}.")
    else:
        _maybe_manifest_row(skipped_rows, "fig_rmse_variable", "deterministic_summary.csv missing or incompatible.")

    # 5) Deterministic summary: ACC
    if not det_summary.empty and {"window", "variable", "model", "lead", "acc_mean"}.issubset(det_summary.columns):
        for variable in sorted(det_summary["variable"].unique()):
            sub = det_summary[det_summary["variable"] == variable].copy()
            sub["lead_days"] = _lead_days(sub["lead"])
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for model in sorted(sub["model"].unique()):
                ss = sub[sub["model"] == model].sort_values("lead_days")
                ax.plot(ss["lead_days"], ss["acc_mean"], marker="o", label=model)
            ax.set_title(f"{variable.upper()} deterministic ACC by lead")
            ax.set_xlabel("Lead (days)")
            ax.set_ylabel("Anomaly correlation")
            ax.set_ylim(min(0.0, sub["acc_mean"].min() - 0.05), 1.02)
            ax.legend(frameon=False)
            _save_figure(fig, figroot, f"fig_acc_{variable}", manifest_rows,
                         f"Deterministic lead-time anomaly-correlation comparison for {variable}.")
    else:
        _maybe_manifest_row(skipped_rows, "fig_acc_variable", "deterministic_summary.csv missing or incompatible for ACC.")

    # 6) Regime-conditioned metrics for z500 RMSE/ACC
    reg_metrics = _read_csv(det_dir / "regime_conditioned_metrics.csv")
    if not reg_metrics.empty and {"variable", "metric", "model", "lead", "regime", "mean", "ci_lo", "ci_hi"}.issubset(reg_metrics.columns):
        reg_metrics["lead_days"] = _lead_days(reg_metrics["lead"])
        for metric in ["RMSE", "ACC"]:
            sub_metric = reg_metrics[(reg_metrics["variable"] == "z500") & (reg_metrics["metric"] == metric)].copy()
            if sub_metric.empty:
                continue
            models = sorted(sub_metric["model"].unique())
            regimes = sorted(sub_metric["regime"].unique())
            nrows = len(models)
            fig, axes = plt.subplots(nrows, 1, figsize=(8.2, max(3.2, 2.8 * nrows)), squeeze=False, sharex=True)
            for ax, model in zip(axes.ravel(), models):
                sub = sub_metric[sub_metric["model"] == model]
                for regime in regimes:
                    ss = sub[sub["regime"] == regime].sort_values("lead_days")
                    if ss.empty:
                        continue
                    ax.plot(ss["lead_days"], ss["mean"], marker="o", label=f"Regime {regime}")
                    ax.fill_between(ss["lead_days"], ss["ci_lo"], ss["ci_hi"], alpha=0.18)
                ax.set_title(f"{model}: Z500 {metric} by regime")
                ax.set_ylabel(metric)
                ax.legend(ncol=min(4, len(regimes)), frameon=False)
            axes.ravel()[-1].set_xlabel("Lead (days)")
            _save_figure(fig, figroot, f"fig_z500_{metric.lower()}_by_regime", manifest_rows,
                         f"Regime-conditioned Z500 {metric} with bootstrap confidence bands.")
    else:
        _maybe_manifest_row(skipped_rows, "fig_z500_metric_by_regime", "regime_conditioned_metrics.csv missing or incompatible.")

    # 7) Spectral diagnostics
    spectral = _read_csv(det_dir / "spectral_diagnostics.csv")
    if not spectral.empty and {"variable", "model", "lead", "wavenumber", "spectral_retention", "spectral_rmse"}.issubset(spectral.columns):
        spectral["lead_days"] = _lead_days(spectral["lead"])
        for lead_day in sorted(spectral["lead_days"].unique()):
            sub = spectral[(spectral["variable"] == "z500") & (spectral["lead_days"] == lead_day)].copy()
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(7.6, 4.7))
            for model in sorted(sub["model"].unique()):
                ss = sub[sub["model"] == model].sort_values("wavenumber")
                ax.plot(ss["wavenumber"], ss["spectral_retention"], label=model)
            ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black")
            ax.set_title(f"Z500 spectral retention at lead day {lead_day:g}")
            ax.set_xlabel("Wavenumber")
            ax.set_ylabel("Forecast / truth spectrum")
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False)
            _save_figure(fig, figroot, f"fig_z500_spectral_retention_day{int(round(lead_day))}", manifest_rows,
                         f"Z500 spectral-retention curves at {lead_day:g}-day lead.")
        fig, ax = plt.subplots(figsize=(7.6, 4.7))
        for model in sorted(spectral["model"].unique()):
            ss = spectral[(spectral["variable"] == "z500") & (spectral["lead_days"] == spectral["lead_days"].max()) & (spectral["model"] == model)].sort_values("wavenumber")
            if not ss.empty:
                ax.plot(ss["wavenumber"], ss["spectral_rmse"], label=model)
        ax.set_title("Z500 spectral RMSE at the longest available lead")
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Spectral RMSE")
        ax.legend(frameon=False)
        _save_figure(fig, figroot, "fig_z500_spectral_rmse_long_lead", manifest_rows,
                     "Z500 spectral-RMSE curves at the longest lead used in the pilot atlas.")
    else:
        _maybe_manifest_row(skipped_rows, "fig_spectral_diagnostics", "spectral_diagnostics.csv missing or incompatible.")

    # 8) Balance diagnostics
    balance = _read_csv(det_dir / "balance_diagnostics.csv")
    if not balance.empty and {"model", "lead", "div_rmse_mean", "vort_rmse_mean"}.issubset(balance.columns):
        balance["lead_days"] = _lead_days(balance["lead"])
        for metric, stem, ylabel in [
            ("div_rmse_mean", "fig_balance_div_rmse", "Divergence RMSE"),
            ("vort_rmse_mean", "fig_balance_vort_rmse", "Vorticity RMSE"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for model in sorted(balance["model"].unique()):
                ss = balance[balance["model"] == model].sort_values("lead_days")
                ax.plot(ss["lead_days"], ss[metric], marker="o", label=model)
            ax.set_title(ylabel + " by lead")
            ax.set_xlabel("Lead (days)")
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False)
            _save_figure(fig, figroot, stem, manifest_rows, f"{ylabel} by lead for each deterministic model.")
    else:
        _maybe_manifest_row(skipped_rows, "fig_balance_metrics", "balance_diagnostics.csv missing or incompatible.")

    # 9) Growth diagnostics
    growth = _read_csv(growth_dir / "growth_metrics.csv")
    if not growth.empty and {"model", "lead", "mean_lagged_growth", "mean_error_growth"}.issubset(growth.columns):
        growth["lead_days"] = _lead_days(growth["lead"])
        gsum = growth.groupby(["model", "lead_days"], as_index=False)[["mean_lagged_growth", "mean_error_growth"]].mean(numeric_only=True)
        for metric, stem, title, ylabel in [
            ("mean_lagged_growth", "fig_growth_lagged", "Lagged forecast-separation growth", "Lagged growth"),
            ("mean_error_growth", "fig_growth_error", "Forecast-error growth", "Error growth"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for model in sorted(gsum["model"].unique()):
                ss = gsum[gsum["model"] == model].sort_values("lead_days")
                ax.plot(ss["lead_days"], ss[metric], marker="o", label=model)
            ax.set_title(title)
            ax.set_xlabel("Lead (days)")
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False)
            _save_figure(fig, figroot, stem, manifest_rows, f"{title} by lead and model.")
        if "regime" in growth.columns and growth["regime"].dropna().nunique() > 1:
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), squeeze=False)
            for j, metric in enumerate(["mean_lagged_growth", "mean_error_growth"]):
                ax = axes[0, j]
                sub = growth.dropna(subset=[metric]).copy()
                if sub.empty:
                    continue
                pivot = sub.groupby(["regime", "lead"], as_index=False)[metric].mean()
                pivot["lead_days"] = _lead_days(pivot["lead"])
                mat = pivot.pivot_table(index="regime", columns="lead_days", values=metric, aggfunc="mean")
                im = ax.imshow(mat.values, aspect="auto")
                ax.set_title(metric.replace("_", " ").title() + " by regime")
                ax.set_xlabel("Lead (days)")
                ax.set_ylabel("Regime")
                ax.set_xticks(np.arange(len(mat.columns)))
                ax.set_xticklabels([f"{c:g}" for c in mat.columns])
                ax.set_yticks(np.arange(len(mat.index)))
                ax.set_yticklabels([str(i) for i in mat.index])
                fig.colorbar(im, ax=ax)
            _save_figure(fig, figroot, "fig_growth_regime_heatmaps", manifest_rows,
                         "Regime-conditioned growth metrics across lead time.")
    else:
        _maybe_manifest_row(skipped_rows, "fig_growth_metrics", "growth_metrics.csv missing or incompatible.")

    # 10) Blocking threshold sweep
    thr = _read_csv(blocking_dir / "blocking_threshold_sweep.csv")
    if not thr.empty and {"lead", "threshold", "fraction_truth_blocked"}.issubset(thr.columns):
        thr["lead_days"] = _lead_days(thr["lead"])
        sweep = thr.groupby(["lead_days", "threshold"], as_index=False)["fraction_truth_blocked"].mean()
        fig, ax = plt.subplots(figsize=(7.3, 4.6))
        for lead_day in sorted(sweep["lead_days"].unique()):
            ss = sweep[sweep["lead_days"] == lead_day].sort_values("threshold")
            ax.plot(ss["threshold"], ss["fraction_truth_blocked"], marker="o", label=f"{lead_day:g} days")
        ax.set_title("Truth blocking-event frequency versus sector threshold")
        ax.set_xlabel("Sector blocking threshold")
        ax.set_ylabel("Fraction of truth times called blocked")
        ax.legend(frameon=False, ncol=2)
        _save_figure(fig, figroot, "fig_blocking_threshold_sweep", manifest_rows,
                     "Sensitivity of truth blocking-event frequency to the sector blocking threshold.")
    else:
        _maybe_manifest_row(skipped_rows, "fig_blocking_threshold_sweep", "blocking_threshold_sweep.csv missing or incompatible.")

    # 11) Blocking RMSE
    b_rmse = _read_csv(blocking_dir / "blocking_rmse.csv")
    if not b_rmse.empty and {"model", "lead", "blocked_rmse", "unblocked_rmse"}.issubset(b_rmse.columns):
        b_rmse["lead_days"] = _lead_days(b_rmse["lead"])
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), squeeze=False, sharex=True)
        for ax, metric, title in zip(
            axes.ravel(),
            ["blocked_rmse", "unblocked_rmse"],
            ["Blocked-case RMSE", "Unblocked-case RMSE"],
        ):
            for model in sorted(b_rmse["model"].unique()):
                ss = b_rmse[b_rmse["model"] == model].sort_values("lead_days")
                ax.plot(ss["lead_days"], ss[metric], marker="o", label=model)
            ax.set_title(title)
            ax.set_xlabel("Lead (days)")
            ax.set_ylabel("RMSE")
            ax.legend(frameon=False)
        _save_figure(fig, figroot, "fig_blocking_rmse", manifest_rows,
                     "Blocked and unblocked Z500 RMSE by lead and deterministic model.")
    else:
        _maybe_manifest_row(skipped_rows, "fig_blocking_rmse", "blocking_rmse.csv missing or incompatible.")

    # 12) Blocking event skill (CSI)
    events = _read_csv(blocking_dir / "blocking_event_metrics.csv")
    if not events.empty and {"model", "lead", "sector", "CSI", "POD", "FAR"}.issubset(events.columns):
        events["lead_days"] = _lead_days(events["lead"])
        for metric in ["CSI", "POD", "FAR"]:
            sectors = sorted(events["sector"].unique())
            fig, axes = plt.subplots(len(sectors), 1, figsize=(7.4, max(3.5, 2.9 * len(sectors))), squeeze=False, sharex=True)
            for ax, sector in zip(axes.ravel(), sectors):
                sub = events[events["sector"] == sector]
                for model in sorted(sub["model"].unique()):
                    ss = sub[sub["model"] == model].sort_values("lead_days")
                    ax.plot(ss["lead_days"], ss[metric], marker="o", label=model)
                ax.set_title(f"{metric} in {sector}")
                ax.set_ylabel(metric)
                ax.set_ylim(0, 1.05)
                ax.legend(frameon=False)
            axes.ravel()[-1].set_xlabel("Lead (days)")
            _save_figure(fig, figroot, f"fig_blocking_{metric.lower()}", manifest_rows,
                         f"{metric} for sector-based blocking verification by lead and model.")
    else:
        _maybe_manifest_row(skipped_rows, "fig_blocking_event_skill", "blocking_event_metrics.csv missing or incompatible.")

    manifest_df = pd.DataFrame(manifest_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    manifest_path = figroot / "figure_manifest.csv"
    skipped_path = figroot / "figure_manifest_skipped.csv"
    manifest_df.to_csv(manifest_path, index=False)
    skipped_df.to_csv(skipped_path, index=False)

    return {
        "figure_root": figroot,
        "figure_manifest": manifest_path,
        "figure_manifest_skipped": skipped_path,
    }
