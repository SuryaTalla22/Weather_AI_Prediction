# Flagship Predictability — next-pass atlas bundle

This bundle is a substantial upgrade of the original `flagship_predictability` codebase. It is designed to move the project from a strong pilot notebook set toward a reproducible, multi-variable, regime-aware, ensemble-aware benchmark pipeline.

## What this bundle adds

- **Dataset coverage audit** before science runs
- **Stable, reusable package layout** under `src/flagship_predictability`
- **Multi-variable deterministic atlas** with regime conditioning and confidence intervals
- **Probabilistic diagnostics** for true ensemble systems: CRPS, spread-skill ratio, rank histograms, threshold Brier scores, reliability tables
- **Regime sensitivity study** across EOF count and regime count
- **Growth diagnostics** that separate lagged-growth, forecast-error growth, and ensemble spread growth
- **Blocking verification** with sector handling robust to wrapped longitude conventions
- **Run scripts** that write CSV / NetCDF / PNG outputs instead of keeping results trapped in notebooks
- **Smoke tests** on synthetic data
- **Example NERSC slurm script**

## Recommended execution order

1. `python workflows/run_dataset_audit.py`
2. `python workflows/run_truth_regimes.py`
3. `python workflows/run_deterministic_atlas.py`
4. `python workflows/run_regime_sensitivity.py`
5. `python workflows/run_growth_diagnostics.py`
6. `python workflows/run_blocking_verification.py`
7. `python workflows/run_probabilistic_atlas.py`  

## Environment assumptions

- WeatherBench2-style local Zarr stores on disk
- `FLAGSHIP_WB2_ROOT` or `PSCRATCH` set
- Python packages: `numpy`, `pandas`, `xarray`, `matplotlib`, `scikit-learn`

## Output layout

Outputs default to `./outputs` unless `FLAGSHIP_OUTPUT_ROOT` is set.

- `outputs/audit/`
- `outputs/regimes/`
- `outputs/deterministic/`
- `outputs/growth/`
- `outputs/blocking/`
- `outputs/probabilistic/`

