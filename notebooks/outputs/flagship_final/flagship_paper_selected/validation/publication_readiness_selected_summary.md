# Selective flagship publication-readiness summary

Overall status: **NOT READY**

- PASS: 8
- WARN: 1
- FAIL: 1

## Main interpretation

- PASS means the section is strong enough to support the deterministic flagship paper now.
- WARN means the section is usable, but should be framed cautiously or upgraded before submission.
- FAIL means the section still blocks a true flagship-level claim.

## Test-by-test assessment

### audit_alignment — WARN
- Criterion: All forecast-truth alignment checks should report status=ok.
- Evidence: Found 20 non-ok rows.
- Recommended action: Fix the non-ok combinations before using them in the paper.

### regime_balance — PASS
- Criterion: No regime should be too small or overwhelmingly dominant in the main setup.
- Evidence: Regime fractions={0: 0.231, 1: 0.118, 2: 0.354, 3: 0.297}.
- Recommended action: No action needed.

### regime_stability — FAIL
- Criterion: Sorted cluster fractions should not vary wildly across EOF-count choices.
- Evidence: Minimum cluster fraction=0.118; max spread in sorted fractions=0.245.
- Recommended action: For absolute flagship level, retrain regimes on a longer climatological window, keep 4 regimes as main, and show 3-regime robustness in the paper.

### deterministic_core — PASS
- Criterion: Core deterministic variables should be present for all models across the headline leads.
- Evidence: Variables present=['t850', 'u850', 'v850', 'z500']; minimum lead count per model-variable=4.
- Recommended action: No action needed.

### reduced_coverage_variables — PASS
- Criterion: Optional variables should either be fully covered or documented as appendix-only.
- Evidence: No missing-variable rows were logged.
- Recommended action: No action needed.

### balance_diagnostics — PASS
- Criterion: Balance diagnostics should be finite and free of obvious pole singularities.
- Evidence: Finite=True; max balance metric=2.775e-05.
- Recommended action: No action needed.

### spectral_diagnostics — PASS
- Criterion: Spectral-retention curves should be finite and physically interpretable.
- Evidence: Finite fraction=1.000; fraction below -0.1 = 0.000.
- Recommended action: No action needed.

### growth_diagnostics — PASS
- Criterion: Each deterministic model should populate all headline growth leads without workflow errors.
- Evidence: Lead counts by model={'GraphCast': 4, 'HRES': 4, 'NeuralGCM': 4}; growth errors=0.
- Recommended action: No action needed.

### blocking_diagnostics — PASS
- Criterion: Blocking verification should have a nontrivial truth-event sample under the chosen threshold.
- Evidence: truth_positive_count sum=720; n_blocked sum=636; mean truth-blocked fraction at thr=0.1 is 0.305.
- Recommended action: No action needed.

### selected_figure_set — PASS
- Criterion: The flagship paper figure set should exist in a dedicated export directory.
- Evidence: Saved 10 selected figures across main and appendix tiers.
- Recommended action: No action needed.
