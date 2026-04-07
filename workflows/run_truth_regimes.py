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
from flagship_predictability import run_truth_regimes

if __name__ == "__main__":
    out = run_truth_regimes(SETTINGS)
    for k, v in out.items():
        print(f"{k}: {v}")
