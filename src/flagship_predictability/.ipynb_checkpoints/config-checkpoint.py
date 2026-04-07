from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json
import numpy as np


@dataclass
class VariableSpec:
    name: str
    candidates: list[str]
    level: int | float | None = None
    climatology_groupby: str | None = None
    thresholds: list[float] = field(default_factory=list)


@dataclass
class AtlasConfig:
    truth_dataset: str
    deterministic_models: dict[str, str]
    ensemble_models: dict[str, str] = field(default_factory=dict)
    date_windows: list[tuple[str, str]] = field(default_factory=lambda: [("2020-01-01", "2020-03-31")])
    leads_hours: list[int] = field(default_factory=lambda: [24, 72, 120, 168])
    variables: dict[str, VariableSpec] = field(default_factory=dict)
    n_regimes: int = 4
    n_eof_components: int = 8
    bootstrap_n: int = 400
    bootstrap_block: int = 5
    blocking_sectors: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "EuroAtlantic": (-60.0, 60.0),
        "Pacific": (120.0, 240.0),
    })
    blocking_threshold: float = 0.5
    assume_geopotential: bool = True
    lag: str = "12h"

    @property
    def leads(self) -> list[np.timedelta64]:
        return [np.timedelta64(int(h), "h") for h in self.leads_hours]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
