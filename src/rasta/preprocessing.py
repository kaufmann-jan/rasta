"""Preprocessing and RAO derivation utilities."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .preprocess import point_acceleration, point_motion
from .rao import RAOSet

MOTION_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def _freq_da(raoset: RAOSet) -> xr.DataArray:
    return xr.DataArray(raoset.dataset.coords["freq"], dims=("freq",), coords={"freq": raoset.dataset.coords["freq"]})


def displacement_to_acceleration(raoset: RAOSet, *, suffix: str = "_acc") -> RAOSet:
    """Convert displacement RAOs to acceleration RAOs using -omega^2 X."""
    omega = _freq_da(raoset)
    acc = -(omega**2) * raoset.rao

    resp = raoset.dataset.coords["resp"].astype(str).values
    new_resp = [f"{name}{suffix}" for name in resp]

    ds = raoset.dataset.copy()
    ds["rao"] = acc
    ds = ds.assign_coords(resp=("resp", new_resp))
    return RAOSet(ds)


def acceleration_from_displacement(raoset: RAOSet) -> RAOSet:
    return displacement_to_acceleration(raoset)


def derive_point_displacement(raoset: RAOSet, x: float, y: float, z: float, resp_name: str = "point") -> RAOSet:
    """Backward-compatible wrapper for deriving point motions in relative mode."""
    return point_motion(
        raoset,
        point=(float(x), float(y), float(z)),
        point_mode="relative",
        point_name=resp_name,
    )


def derive_point_acceleration(raoset: RAOSet, x: float, y: float, z: float, resp_name: str = "point") -> RAOSet:
    """Backward-compatible wrapper for deriving point accelerations in relative mode."""
    return point_acceleration(
        raoset,
        point=(float(x), float(y), float(z)),
        point_mode="relative",
        point_name=resp_name,
        derive_accelerations=True,
    )
