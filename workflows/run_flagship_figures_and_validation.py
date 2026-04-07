from __future__ import annotations

from pathlib import Path
from flagship_predictability.figures import generate_flagship_figures
from flagship_predictability.validation import evaluate_publication_readiness
from flagship_predictability.wb2_paths import resolve_output_root


def main():
    base = resolve_output_root()
    figs = generate_flagship_figures(output_root=base)
    val = evaluate_publication_readiness(output_root=base)
    print({**{k: str(v) for k, v in figs.items()}, **{k: str(v) for k, v in val.items()}})


if __name__ == "__main__":
    main()
