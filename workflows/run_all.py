from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from examples.default_settings import SETTINGS
from flagship_predictability import (
    run_dataset_audit,
    run_truth_regimes,
    run_regime_sensitivity,
    run_deterministic_atlas,
    run_growth_diagnostics,
    run_blocking_verification,
    run_probabilistic_atlas,
)

if __name__ == "__main__":
    steps = [
        run_dataset_audit,
        run_truth_regimes,
        run_regime_sensitivity,
        run_deterministic_atlas,
        run_growth_diagnostics,
        run_blocking_verification,
        run_probabilistic_atlas,
    ]
    for fn in steps:
        print(f"\n=== {fn.__name__} ===")
        out = fn(SETTINGS)
        for k, v in out.items():
            print(f"{k}: {v}")
