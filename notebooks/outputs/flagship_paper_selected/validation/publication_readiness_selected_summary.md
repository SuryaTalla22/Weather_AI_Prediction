# Selective flagship publication-readiness summary

Overall status: **NOT READY**

- PASS: 0
- WARN: 1
- FAIL: 9

## Main interpretation

- PASS means the section is strong enough to support the deterministic flagship paper now.
- WARN means the section is usable, but should be framed cautiously or upgraded before submission.
- FAIL means the section still blocks a true flagship-level claim.

## Test-by-test assessment

### audit_alignment — FAIL
- Criterion: alignment_audit.csv should exist and contain aligned forecast-truth checks.
- Evidence: alignment_audit.csv missing or empty.
- Recommended action: Rerun Notebook 00.

### regime_balance — FAIL
- Criterion: Main regime-label file should be non-empty and contain regime labels.
- Evidence: regime_labels.csv missing or empty.
- Recommended action: Rerun Notebook 01.

### regime_stability — FAIL
- Criterion: A flagship paper needs a regime-sensitivity table.
- Evidence: regime_sensitivity.csv missing or empty.
- Recommended action: Rerun Notebook 03.

### deterministic_core — FAIL
- Criterion: deterministic_summary.csv should exist and contain model-variable-lead rows.
- Evidence: deterministic_summary.csv missing or empty.
- Recommended action: Rerun Notebook 02.

### reduced_coverage_variables — WARN
- Criterion: Reduced-coverage variables should be documented clearly.
- Evidence: variable_availability.csv missing or empty.
- Recommended action: Use the patched Notebook 02 and keep optional variables out of the headline table.

### balance_diagnostics — FAIL
- Criterion: Pole-safe balance diagnostics should exist for flagship-level claims.
- Evidence: balance_diagnostics.csv missing or empty.
- Recommended action: Use the pole-safe patch and rerun Notebook 02.

### spectral_diagnostics — FAIL
- Criterion: Spectral diagnostics should exist for the scale-conditioned flagship section.
- Evidence: spectral_diagnostics.csv missing or empty.
- Recommended action: Rerun Notebook 02.

### growth_diagnostics — FAIL
- Criterion: growth_metrics.csv should be populated for deterministic growth analysis.
- Evidence: growth_metrics.csv missing or empty.
- Recommended action: Use the patched Notebook 04 and rerun.

### blocking_diagnostics — FAIL
- Criterion: Blocking verification outputs should contain a nontrivial truth-event sample.
- Evidence: blocking_event_metrics.csv missing or empty.
- Recommended action: Rerun Notebook 05 with the threshold sweep enabled.

### selected_figure_set — FAIL
- Criterion: The flagship paper figure set should exist in a dedicated export directory.
- Evidence: Missing expected figures=['fig01_z500_core_skill', 'fig02_multivariable_rmse_support', 'fig03_z500_regime_rmse', 'fig04_z500_regime_acc', 'fig05_z500_spectral_summary', 'fig06_balance_diagnostics', 'fig07_growth_diagnostics', 'fig08_blocking_summary'].
- Recommended action: Rerun the figure cells and inspect missing upstream outputs.
