"""Operational profile representation and validation."""

from __future__ import annotations

import numpy as np
import xarray as xr


def validate_operational_profile(profile: xr.Dataset) -> xr.Dataset:
    """Validate and normalize operational profile dataset with variable `w`."""
    if "w" not in profile.data_vars:
        raise ValueError("operational profile must contain data variable 'w'")
    if "mean_dir" not in profile.coords:
        raise ValueError("operational profile must contain 'mean_dir' coordinate")

    w = profile["w"].astype(float)
    if np.any(w.values < 0.0):
        raise ValueError("operational weights must be non-negative")

    total = float(w.sum().values)
    if total <= 0.0:
        raise ValueError("operational weights must sum to a positive value")

    out = profile.copy()
    out["w"] = w / total
    return out


def make_operational_profile(
    *,
    mean_dirs: np.ndarray | list[float],
    speeds: np.ndarray | list[float] | None = None,
    depths: np.ndarray | list[float] | None = None,
    weights: xr.DataArray | None = None,
    uniform_headings: bool = True,
) -> xr.Dataset:
    """Construct an operational profile over mean_dir and optional speed/depth."""
    mean_dirs_arr = np.asarray(mean_dirs, dtype=float)
    if mean_dirs_arr.ndim != 1 or mean_dirs_arr.size == 0:
        raise ValueError("mean_dirs must be a non-empty 1-D array")

    coords: dict[str, np.ndarray] = {"mean_dir": mean_dirs_arr}
    dims = ["mean_dir"]

    if speeds is not None:
        s_arr = np.asarray(speeds, dtype=float)
        if s_arr.ndim != 1 or s_arr.size == 0:
            raise ValueError("speeds must be a non-empty 1-D array")
        coords["speed"] = s_arr
        dims.append("speed")

    if depths is not None:
        d_arr = np.asarray(depths, dtype=float)
        if d_arr.ndim != 1 or d_arr.size == 0:
            raise ValueError("depths must be a non-empty 1-D array")
        coords["depth"] = d_arr
        dims.append("depth")

    if weights is None:
        shape = tuple(coords[d].size for d in dims)
        data = np.ones(shape, dtype=float)
        w = xr.DataArray(data, dims=tuple(dims), coords={d: coords[d] for d in dims})
    else:
        w = weights

    ds = xr.Dataset({"w": w}, coords=coords)
    ds.attrs["uniform_headings"] = bool(uniform_headings)
    return validate_operational_profile(ds)
