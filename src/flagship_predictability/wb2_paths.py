from __future__ import annotations
import os
from pathlib import Path


def resolve_wb2_root() -> Path:
    override = os.environ.get("FLAGSHIP_WB2_ROOT")
    return Path(override).expanduser().resolve()


def resolve_output_root() -> Path:
    override = os.environ.get("FLAGSHIP_OUTPUT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path.cwd() / "outputs"


def dataset_registry() -> dict[str, str]:
    return {
        "era5_truth_240": "era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
        "hres_t0_240": "hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
        "hres_0012_240": "hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr",
        "graphcast_2020_240": "graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr",
        "neuralgcm_det_2020_240": "neuralgcm_deterministic/2020-240x121_equiangular_with_poles_conservative.zarr",
        "neuralgcm_ens_mean_2020_240": "neuralgcm_ens/2020-240x121_equiangular_with_poles_conservative_mean.zarr",
        "ifs_ens_240": "ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr",
    }


def resolve_dataset_path(name: str, root: Path | None = None) -> Path:
    reg = dataset_registry()
    if name not in reg:
        raise KeyError(f"Unknown dataset name: {name}")
    base = resolve_wb2_root() if root is None else Path(root)
    return (base / reg[name]).resolve()
