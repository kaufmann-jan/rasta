"""Thin wrapper around canonical RAO xarray datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from .validation import validate_dataset


@dataclass(frozen=True)
class RAOSet:
    """Validated wrapper for canonical RAO dataset."""

    dataset: xr.Dataset

    def __post_init__(self) -> None:
        validated = validate_dataset(self.dataset)
        object.__setattr__(self, "dataset", validated)

    @property
    def rao(self) -> xr.DataArray:
        return self.dataset["rao"]

    @property
    def amp(self) -> xr.DataArray:
        return np.abs(self.rao)

    @property
    def phase_deg(self) -> xr.DataArray:
        return xr.apply_ufunc(np.angle, self.rao, kwargs={"deg": True})

    def sel(self, **indexers: Any) -> "RAOSet":
        return RAOSet(self.dataset.sel(**indexers))

    def sel_dir(self, direction_deg: Any, method: str | None = None) -> "RAOSet":
        return RAOSet(self.dataset.sel(dir=direction_deg, method=method))

    def sel_speed(self, speed: Any, method: str | None = None) -> "RAOSet":
        if "speed" not in self.dataset.coords:
            raise KeyError("'speed' coordinate not present")
        return RAOSet(self.dataset.sel(speed=speed, method=method))

    @classmethod
    def from_amp_phase(
        cls,
        amp: xr.DataArray,
        phase_deg: xr.DataArray,
        attrs: dict[str, Any],
    ) -> "RAOSet":
        """Construct RAOSet from amplitude and phase in degrees."""
        phase_rad = np.deg2rad(phase_deg)
        rao = amp * xr.apply_ufunc(np.exp, 1j * phase_rad)
        dataset = xr.Dataset({"rao": rao}, attrs=attrs)
        return cls(dataset)
