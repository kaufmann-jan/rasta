"""Preprocessing and RAO derivation utilities."""

from __future__ import annotations

import numpy as np
import xarray as xr

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
    """Derive linearized point displacement RAO at position r=(x,y,z) from 6DoF motion RAOs.

    u_point = u + theta x r
    """
    rao = raoset.rao
    resp = raoset.dataset.coords["resp"].astype(str)

    needed = set(MOTION_NAMES)
    found = set(resp.values.tolist())
    missing = sorted(needed - found)
    if missing:
        raise ValueError(f"missing required motion responses: {missing}")

    surge = rao.sel(resp="surge")
    sway = rao.sel(resp="sway")
    heave = rao.sel(resp="heave")
    roll = rao.sel(resp="roll")
    pitch = rao.sel(resp="pitch")
    yaw = rao.sel(resp="yaw")

    u_px = surge + pitch * z - yaw * y
    u_py = sway + yaw * x - roll * z
    u_pz = heave + roll * y - pitch * x

    point = xr.concat([u_px, u_py, u_pz], dim="resp")
    point = point.assign_coords(resp=("resp", [f"{resp_name}_x", f"{resp_name}_y", f"{resp_name}_z"]))

    ds = xr.Dataset({"rao": point}, attrs=raoset.dataset.attrs)
    for cname, c in raoset.dataset.coords.items():
        if cname != "resp" and cname in ds.dims:
            ds = ds.assign_coords({cname: c})
    return RAOSet(ds)
