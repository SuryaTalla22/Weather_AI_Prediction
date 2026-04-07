from __future__ import annotations
import numpy as np
from flagship_predictability.config import AtlasConfig, VariableSpec

SETTINGS = AtlasConfig(
    truth_dataset="era5_truth_240",
    deterministic_models={
        "HRES": "hres_0012_240",
        "GraphCast": "graphcast_2020_240",
        "NeuralGCM": "neuralgcm_det_2020_240",
    },
    ensemble_models={
        # Example only. Replace or remove depending on your actual available ensemble store.
        #"IFS-ENS": "ifs_ens_240",
    },
    date_windows=[
        ("2020-01-01", "2020-03-31"),
    ],
    leads_hours=[24, 72, 120, 168],
    variables={
        "z500": VariableSpec(name="z500", candidates=["z", "geopotential", "gh"], level=500),
        "t850": VariableSpec(name="t850", candidates=["t", "temperature"], level=850),
        "u850": VariableSpec(name="u850", candidates=["u", "u_component_of_wind"], level=850),
        "v850": VariableSpec(name="v850", candidates=["v", "v_component_of_wind"], level=850),
        "mslp": VariableSpec(name="mslp", candidates=["msl", "mean_sea_level_pressure", "mslp"], level=None),
        # Add precipitation only if your stores expose it consistently.
        # "tp": VariableSpec(name="tp", candidates=["tp", "total_precipitation", "precipitation"], level=None),
    },
    n_regimes=4,
    n_eof_components=8,
    bootstrap_n=400,
    bootstrap_block=5,
    blocking_sectors={
        "EuroAtlantic": (-60.0, 60.0),
        "Pacific": (120.0, 240.0),
    },
    blocking_threshold=0.1,
    assume_geopotential=True,
    lag="12h",
)
