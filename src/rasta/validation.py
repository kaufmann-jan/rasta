"""Schema validation for canonical RAO datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

REQUIRED_DIMS = ("freq", "dir", "resp")
REQUIRED_ATTRS = {
    "freq_unit": "rad/s",
    "dir_unit": "deg",
    "dir_convention": "180=head, 0=following",
    "coord_sys": "body-fixed RH",
    "axis_x": "forward",
    "axis_y": "port",
    "axis_z": "up",
    "rotation_convention": "RH about +x/+y/+z for roll/pitch/yaw",
    "rao_definition": "complex response per unit wave amplitude",
    "angle_unit": "rad",
}


class ValidationError(ValueError):
    """Raised when a dataset does not satisfy the canonical schema."""


@dataclass(frozen=True)
class ValidationResult:
    dataset: xr.Dataset


def _require(condition: bool, msg: str) -> None:
    if not condition:
        raise ValidationError(msg)


def _is_float_coord(coord: xr.DataArray) -> bool:
    return np.issubdtype(coord.dtype, np.floating)


def validate_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Validate and normalize a dataset to canonical internal form.

    Normalization steps:
    - sort by freq
    - wrap dir into [0, 360)
    """

    _require("rao" in dataset.data_vars, "missing data variable 'rao'")

    rao = dataset["rao"]
    _require(np.issubdtype(rao.dtype, np.complexfloating), "'rao' must be complex dtype")

    for dim in REQUIRED_DIMS:
        _require(dim in rao.dims, f"'rao' missing required dim '{dim}'")
        _require(dim in dataset.coords, f"missing required coordinate '{dim}'")
        _require(dataset.coords[dim].ndim == 1, f"coordinate '{dim}' must be 1-D")

    freq = dataset.coords["freq"]
    direction = dataset.coords["dir"]

    _require(_is_float_coord(freq), "'freq' must be float dtype")
    _require(_is_float_coord(direction), "'dir' must be float dtype")

    if freq.size >= 2:
        freq_values = np.asarray(freq.values)
        _require(np.all(np.diff(freq_values) > 0), "'freq' must be strictly increasing")

    for key, expected in REQUIRED_ATTRS.items():
        value = dataset.attrs.get(key)
        _require(value == expected, f"dataset attr '{key}' must be '{expected}', got '{value}'")

    out = dataset.sortby("freq")

    wrapped_dir = np.mod(np.asarray(out.coords["dir"].values, dtype=float), 360.0)
    out = out.assign_coords(dir=("dir", wrapped_dir))

    return out
