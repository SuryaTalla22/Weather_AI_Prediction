from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .wb2_paths import resolve_output_root


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _status_rank(status: str) -> int:
    return {"FAIL": 0, "WARN": 1, "PASS": 2}.get(status, -1)


def _record(rows: list[dict], test_name: str, status: str, criterion: str, evidence: str, action: str):
    rows.append({
        "test_name": test_name,
        "status": status,
        "criterion": criterion,
        "evidence": evidence,
        "recommended_action": action,
    })


def evaluate_publication_readiness(output_root: Path | None = None) -> dict[str, Path]:
    base = resolve_output_root() if output_root is None else Path(output_root)
    outdir = base / "validation"
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    align = _read_csv(base / "audit" / "alignment_audit.csv")
    if align.empty:
        _record(rows, "audit_alignment", "FAIL",
                "alignment_audit.csv should exist and contain aligned forecast-truth rows.",
                "alignment_audit.csv is missing or empty.",
                "Rerun Notebook 00 and verify dataset paths and variable availability.")
    else:
        statuses = align["status"].astype(str).value_counts().to_dict() if "status" in align.columns else {}
        non_ok = align[align["status"] != "ok"] if "status" in align.columns else pd.DataFrame()
        if non_ok.empty:
            _record(rows, "audit_alignment", "PASS",
                    "All aligned forecast-truth checks should report status=ok.",
                    f"All {len(align)} audited rows reported status=ok.",
                    "No action needed.")
        else:
            _record(rows, "audit_alignment", "WARN",
                    "All aligned forecast-truth checks should report status=ok.",
                    f"Found {len(non_ok)} non-ok rows. Status counts: {statuses}",
                    "Inspect non-ok dataset/variable/lead combinations before making publication claims.")

    labels = _read_csv(base / "regimes" / "regime_labels.csv")
    if labels.empty or "regime" not in labels.columns:
        _record(rows, "regime_balance", "FAIL",
                "Main regime-label file should be non-empty and contain regime labels.",
                "regime_labels.csv missing or empty.",
                "Rerun Notebook 01.")
    else:
        frac = labels["regime"].value_counts(normalize=True).sort_index()
        minf = float(frac.min())
        maxf = float(frac.max())
        if minf >= 0.10 and maxf <= 0.50:
            status = "PASS"
            action = "No action needed."
        elif minf >= 0.06 and maxf <= 0.60:
            status = "WARN"
            action = "Keep the main regime count, but present 3-regime sensitivity as a robustness check."
        else:
            status = "FAIL"
            action = "Reduce regime count or broaden the time window before publication."
        _record(rows, "regime_balance", status,
                "No regime should be vanishingly small or overwhelmingly dominant in the main configuration.",
                f"Regime fractions = {dict((int(k), round(v, 3)) for k, v in frac.items())}.",
                action)

    reg_sens = _read_csv(base / "regimes" / "regime_sensitivity.csv")
    if reg_sens.empty:
        _record(rows, "regime_sensitivity_stability", "WARN",
                "Sensitivity table should be available to judge regime-definition stability.",
                "regime_sensitivity.csv missing or empty.",
                "Rerun Notebook 03 for a stronger robustness section.")
    else:
        subset = reg_sens[reg_sens["n_regimes"] == 4].copy()
        if subset.empty:
            _record(rows, "regime_sensitivity_stability", "WARN",
                    "Need at least one 4-regime sensitivity slice for the main analysis.",
                    "No rows found with n_regimes=4.",
                    "Rerun Notebook 03 using n_regimes=4 in the grid.")
        else:
            spread_rows = []
            for _, grp in subset.groupby("n_components"):
                fracs = np.sort(grp["fraction"].to_numpy(dtype=float))
                spread_rows.append(fracs)
            mat = np.vstack(spread_rows) if spread_rows else np.empty((0,))
            max_range = float(np.nanmax(mat, axis=0).max() - np.nanmin(mat, axis=0).min()) if mat.size else np.nan
            min_cluster = float(subset["fraction"].min())
            if min_cluster >= 0.08 and max_range <= 0.15:
                status = "PASS"
                action = "No action needed."
            elif min_cluster >= 0.05 and max_range <= 0.22:
                status = "WARN"
                action = "Keep sensitivity in the appendix and avoid overclaiming regime uniqueness."
            else:
                status = "FAIL"
                action = "Revisit EOF count, regime count, or time-window choice."
            _record(rows, "regime_sensitivity_stability", status,
                    "Sorted cluster fractions should not swing wildly across EOF-component choices.",
                    f"Minimum cluster fraction={min_cluster:.3f}, max spread in sorted fractions={max_range:.3f}.",
                    action)

    det = _read_csv(base / "deterministic" / "deterministic_summary.csv")
    if det.empty:
        _record(rows, "deterministic_summary", "FAIL",
                "deterministic_summary.csv should exist and contain model-variable-lead skill rows.",
                "deterministic_summary.csv missing or empty.",
                "Rerun Notebook 02.")
    else:
        required_vars = {"z500", "t850", "u850", "v850"}
        have_vars = set(det["variable"].unique()) if "variable" in det.columns else set()
        missing_req = sorted(required_vars - have_vars)
        lead_counts = det.groupby(["model", "variable"])["lead"].nunique().min()
        if not missing_req and int(lead_counts) >= 4:
            _record(rows, "deterministic_summary", "PASS",
                    "Core deterministic variables should be present for all models at all headline leads.",
                    f"Variables present={sorted(have_vars)}; minimum lead count per model-variable={int(lead_counts)}.",
                    "No action needed.")
        else:
            _record(rows, "deterministic_summary", "WARN",
                    "Core deterministic variables should be present for all models at all headline leads.",
                    f"Missing required vars={missing_req}; minimum lead count per model-variable={lead_counts}.",
                    "Keep the main paper focused on variables with complete multi-model coverage.")

    avail = _read_csv(base / "deterministic" / "variable_availability.csv")
    if avail.empty:
        _record(rows, "variable_availability", "WARN",
                "Variable-availability table should exist to document reduced-coverage variables.",
                "variable_availability.csv missing or empty.",
                "Use the patched Notebook 02 so missing variables are documented cleanly.")
    else:
        missing = avail[avail["status"] != "ok"] if "status" in avail.columns else pd.DataFrame()
        if missing.empty:
            _record(rows, "variable_availability", "PASS",
                    "All requested variables were available for all requested models.",
                    "No missing-variable rows were logged.",
                    "No action needed.")
        else:
            opt_only = set(missing["variable"].astype(str).unique()) <= {"mslp", "tp"}
            status = "WARN" if opt_only else "FAIL"
            _record(rows, "variable_availability", status,
                    "Missing variables should be optional or explicitly carved out of the main comparison.",
                    f"Missing-variable rows observed for variables={sorted(missing['variable'].astype(str).unique())}.",
                    "Keep reduced-coverage variables out of the main all-model headline table.")

    d_errors = _read_csv(base / "deterministic" / "deterministic_errors.csv")
    if d_errors.empty:
        _record(rows, "deterministic_runtime_errors", "PASS",
                "Deterministic workflow should complete without runtime errors other than documented missing optional variables.",
                "No deterministic runtime errors logged.",
                "No action needed.")
    else:
        err_text = " | ".join(d_errors.get("error", pd.Series(dtype=str)).astype(str).head(5).tolist())
        non_optional = ~d_errors.get("error", pd.Series(dtype=str)).astype(str).str.contains("msl|mslp|mean_sea_level_pressure", case=False, na=False)
        if non_optional.any():
            _record(rows, "deterministic_runtime_errors", "FAIL",
                    "Deterministic workflow should complete without runtime errors other than documented missing optional variables.",
                    f"Found {len(d_errors)} deterministic errors. Example: {err_text}",
                    "Fix these runtime errors before using the outputs in the paper.")
        else:
            _record(rows, "deterministic_runtime_errors", "WARN",
                    "Only optional-variable availability issues should remain.",
                    f"Found only optional-variable errors. Example: {err_text}",
                    "Document reduced coverage for the affected variable.")

    balance = _read_csv(base / "deterministic" / "balance_diagnostics.csv")
    if balance.empty:
        _record(rows, "balance_diagnostics", "WARN",
                "Pole-safe divergence/vorticity diagnostics should be populated for publication-ready balance analysis.",
                "balance_diagnostics.csv missing or empty.",
                "Apply the pole-safe patch and rerun Notebook 02.")
    else:
        vals = pd.concat([balance["div_rmse_mean"], balance["vort_rmse_mean"]], axis=0).astype(float)
        finite = np.isfinite(vals).all()
        vmax = float(np.nanmax(vals)) if len(vals) else np.nan
        if finite and vmax < 1e6:
            status = "PASS"
            action = "No action needed."
        elif finite and vmax < 1e9:
            status = "WARN"
            action = "Check the latitude trimming and ensure no near-pole singularities remain."
        else:
            status = "FAIL"
            action = "Treat current balance diagnostics as non-publication-quality until the pole-safe fix is verified."
        _record(rows, "balance_diagnostics", status,
                "Divergence/vorticity RMSE should be finite and not exhibit obvious pole singularities.",
                f"Finite={finite}; max balance metric={vmax:.3e}.",
                action)

    spectral = _read_csv(base / "deterministic" / "spectral_diagnostics.csv")
    if spectral.empty:
        _record(rows, "spectral_diagnostics", "WARN",
                "Spectral diagnostics should exist for the flagship z500 scale-analysis section.",
                "spectral_diagnostics.csv missing or empty.",
                "Rerun Notebook 02.")
    else:
        good = np.isfinite(spectral["spectral_retention"]).mean() if "spectral_retention" in spectral.columns else 0.0
        neg = float((spectral.get("spectral_retention", pd.Series(dtype=float)) < -0.1).mean()) if "spectral_retention" in spectral.columns else 1.0
        status = "PASS" if good >= 0.95 and neg == 0.0 else "WARN"
        _record(rows, "spectral_diagnostics", status,
                "Most spectral-retention values should be finite and physically interpretable.",
                f"Finite retention fraction={good:.3f}; fraction < -0.1 = {neg:.3f}.",
                "Inspect any suspicious long-wave or near-Nyquist artifacts before final plotting.")

    growth = _read_csv(base / "growth" / "growth_metrics.csv")
    gthr = _read_csv(base / "growth" / "growth_threshold_times.csv")
    gerr = _read_csv(base / "growth" / "growth_errors.csv")
    if growth.empty:
        _record(rows, "growth_diagnostics", "FAIL",
                "growth_metrics.csv should be populated for deterministic growth analysis.",
                "growth_metrics.csv missing or empty.",
                "Use the patched Notebook 04 and rerun.")
    else:
        model_leads = growth.groupby("model")["lead"].nunique()
        full = int(model_leads.min()) >= 4 if not model_leads.empty else False
        if full and (gerr.empty or len(gerr) == 0):
            status = "PASS"
            action = "No action needed."
        elif full:
            status = "WARN"
            action = "Inspect growth_errors.csv and confirm the surviving outputs are still complete."
        else:
            status = "FAIL"
            action = "Fix time-axis alignment until every model has the headline leads."
        _record(rows, "growth_diagnostics", status,
                "Each deterministic model should have populated growth diagnostics across the headline leads.",
                f"Lead counts by model={model_leads.to_dict()}; threshold rows={len(gthr)}; growth errors={len(gerr)}.",
                action)

    b_rmse = _read_csv(base / "blocking" / "blocking_rmse.csv")
    b_thr = _read_csv(base / "blocking" / "blocking_threshold_sweep.csv")
    b_evt = _read_csv(base / "blocking" / "blocking_event_metrics.csv")
    if b_evt.empty:
        _record(rows, "blocking_verification", "FAIL",
                "Blocking verification outputs should exist and contain sector event metrics.",
                "blocking_event_metrics.csv missing or empty.",
                "Rerun Notebook 05.")
    else:
        truth_positives = b_evt.get("truth_positive_count", pd.Series(dtype=float)).sum()
        blocked_rows = b_rmse.get("n_blocked", pd.Series(dtype=float)).fillna(0).sum() if not b_rmse.empty else 0
        thr01 = 0.0
        if not b_thr.empty and "threshold" in b_thr.columns and "fraction_truth_blocked" in b_thr.columns:
            subset = b_thr[np.isclose(b_thr["threshold"].astype(float), 0.1)]
            if not subset.empty:
                thr01 = float(subset["fraction_truth_blocked"].mean())
        if truth_positives > 0 and blocked_rows > 0:
            status = "PASS"
            action = "No action needed."
        elif thr01 > 0:
            status = "WARN"
            action = "Lower the blocking threshold to about 0.1 or broaden the window before publication."
        else:
            status = "FAIL"
            action = "Current blocking event definition is too strict or too data-sparse for a flagship paper section."
        _record(rows, "blocking_verification", status,
                "Blocking section should have a nontrivial truth-event sample under the chosen threshold.",
                f"truth_positive_count sum={truth_positives}; n_blocked sum={blocked_rows}; mean truth-blocked fraction at thr=0.1 is {thr01:.3f}.",
                action)

    fig_manifest = _read_csv(base / "figures" / "figure_manifest.csv")
    if fig_manifest.empty:
        _record(rows, "flagship_figures", "WARN",
                "A paper-ready figure manifest should exist with exported PNG/PDF pairs.",
                "figure_manifest.csv missing or empty.",
                "Run the flagship figure notebook after all non-ensemble outputs are finalized.")
    else:
        _record(rows, "flagship_figures", "PASS",
                "Paper-ready figure exports should exist for the completed studies.",
                f"Generated {len(fig_manifest)} figure entries.",
                "Review captions and select the strongest subset for the main paper.")

    readiness = pd.DataFrame(rows)
    overall = readiness["status"].map(_status_rank).min()
    overall_label = {0: "NOT READY", 1: "NEARLY READY", 2: "READY"}.get(int(overall), "UNKNOWN")
    summary = {
        "overall_status": overall_label,
        "n_pass": int((readiness["status"] == "PASS").sum()),
        "n_warn": int((readiness["status"] == "WARN").sum()),
        "n_fail": int((readiness["status"] == "FAIL").sum()),
    }

    csv_path = outdir / "publication_readiness.csv"
    json_path = outdir / "publication_readiness_summary.json"
    md_path = outdir / "publication_readiness_summary.md"
    readiness.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [f"# Publication readiness summary", "",
             f"Overall status: **{overall_label}**", "",
             f"- PASS: {summary['n_pass']}",
             f"- WARN: {summary['n_warn']}",
             f"- FAIL: {summary['n_fail']}", "", "## Test-by-test assessment", ""]
    for _, row in readiness.iterrows():
        lines.extend([
            f"### {row['test_name']} — {row['status']}",
            f"- Criterion: {row['criterion']}",
            f"- Evidence: {row['evidence']}",
            f"- Recommended action: {row['recommended_action']}",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "publication_readiness_csv": csv_path,
        "publication_readiness_summary_json": json_path,
        "publication_readiness_summary_md": md_path,
    }
