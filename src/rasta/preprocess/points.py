"""Point-motion and point-acceleration derivation from 6DoF motion RAOs."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..rao import RAOSet

_REQUIRED_MOTIONS = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def _validate_motion_inputs(ds: xr.Dataset) -> None:
    resp_vals = {str(v) for v in ds.coords["resp"].values.tolist()}
    missing = sorted(set(_REQUIRED_MOTIONS) - resp_vals)
    if missing:
        raise ValueError(f"missing required motion responses: {missing}")

    if ds.attrs.get("angle_unit") != "rad":
        raise ValueError("roll, pitch, yaw must be in radians (dataset attr angle_unit='rad')")


def _resolve_point_relative_coordinates(
    ds: xr.Dataset, point: tuple[float, float, float], point_mode: str
) -> tuple[float, float, float]:
    mode = point_mode.lower()
    if mode == "relative":
        return tuple(float(v) for v in point)
    if mode != "absolute":
        raise ValueError("point_mode must be either 'relative' or 'absolute'")

    missing = [name for name in ("xref", "yref", "zref") if name not in ds.attrs]
    if missing:
        raise ValueError(f"absolute point_mode requires dataset attrs {missing}")

    x, y, z = (float(v) for v in point)
    return (
        x - float(ds.attrs["xref"]),
        y - float(ds.attrs["yref"]),
        z - float(ds.attrs["zref"]),
    )


def _point_translation_from_rigid_body(
    trans: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    rot: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    dx: float,
    dy: float,
    dz: float,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    x_trans, y_trans, z_trans = trans
    roll, pitch, yaw = rot
    x_point = x_trans + pitch * dz - yaw * dy
    y_point = y_trans + yaw * dx - roll * dz
    z_point = z_trans + roll * dy - pitch * dx
    return x_point, y_point, z_point


def _freq_da(ds: xr.Dataset) -> xr.DataArray:
    return xr.DataArray(ds.coords["freq"], dims=("freq",), coords={"freq": ds.coords["freq"]})


def _derive_motion_accelerations(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    omega = _freq_da(ds)
    scale = -(omega**2)
    rao = ds["rao"]
    surge_acc = scale * rao.sel(resp="surge")
    sway_acc = scale * rao.sel(resp="sway")
    heave_acc = scale * rao.sel(resp="heave")
    roll_acc = scale * rao.sel(resp="roll")
    pitch_acc = scale * rao.sel(resp="pitch")
    yaw_acc = scale * rao.sel(resp="yaw")
    return surge_acc, sway_acc, heave_acc, roll_acc, pitch_acc, yaw_acc


def _build_output_dataset(ds: xr.Dataset, out: xr.DataArray, *, point: tuple[float, float, float], point_mode: str, point_name: str | None, relative_point: tuple[float, float, float]) -> RAOSet:
    ds_out = xr.Dataset({"rao": out}, attrs=dict(ds.attrs))
    for name, coord in ds.coords.items():
        if name == "resp":
            continue
        if coord.ndim == 0:
            ds_out = ds_out.assign_coords({name: coord})
        elif name in out.dims:
            ds_out = ds_out.assign_coords({name: coord})
    ds_out.attrs["point_name"] = None if point_name is None else str(point_name)
    ds_out.attrs["point_mode"] = str(point_mode)
    ds_out.attrs["point_coordinates_input"] = tuple(float(v) for v in point)
    ds_out.attrs["point_coordinates_relative"] = tuple(float(v) for v in relative_point)
    return RAOSet(ds_out)


def point_motion(
    rs: RAOSet,
    *,
    point: tuple[float, float, float],
    point_mode: str = "relative",
    point_name: str | None = None,
) -> RAOSet:
    """Derive translatory point-motion RAOs from 6DoF motion RAOs."""
    ds = rs.dataset
    _validate_motion_inputs(ds)

    dx, dy, dz = _resolve_point_relative_coordinates(ds, point, point_mode)
    base = rs.rao

    x_point, y_point, z_point = _point_translation_from_rigid_body(
        (base.sel(resp="surge"), base.sel(resp="sway"), base.sel(resp="heave")),
        (base.sel(resp="roll"), base.sel(resp="pitch"), base.sel(resp="yaw")),
        dx,
        dy,
        dz,
    )

    prefix = "point" if point_name is None else str(point_name)
    derived = xr.concat([x_point, y_point, z_point], dim="resp").assign_coords(
        resp=("resp", [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"])
    )

    out = xr.concat([base, derived], dim="resp")
    return _build_output_dataset(
        ds,
        out,
        point=point,
        point_mode=point_mode,
        point_name=point_name,
        relative_point=(dx, dy, dz),
    )


def point_acceleration(
    rs: RAOSet,
    *,
    point: tuple[float, float, float],
    point_mode: str = "relative",
    point_name: str | None = None,
    derive_accelerations: bool = True,
) -> RAOSet:
    """Derive translatory point accelerations from 6DoF motion RAOs.

    Uses only the linear rigid-body term `a_point = a_ref + alpha x r` and
    explicitly excludes nonlinear centripetal terms `omega x (omega x r)`.
    """
    ds = rs.dataset
    _validate_motion_inputs(ds)
    dx, dy, dz = _resolve_point_relative_coordinates(ds, point, point_mode)

    base = rs.rao
    if derive_accelerations:
        surge_acc, sway_acc, heave_acc, roll_acc, pitch_acc, yaw_acc = _derive_motion_accelerations(ds)
    else:
        required = ("surge_acc", "sway_acc", "heave_acc", "roll_acc", "pitch_acc", "yaw_acc")
        resp_vals = {str(v) for v in ds.coords["resp"].values.tolist()}
        missing = sorted(set(required) - resp_vals)
        if missing:
            raise ValueError(f"derive_accelerations=False but acceleration RAOs missing: {missing}")
        surge_acc = base.sel(resp="surge_acc")
        sway_acc = base.sel(resp="sway_acc")
        heave_acc = base.sel(resp="heave_acc")
        roll_acc = base.sel(resp="roll_acc")
        pitch_acc = base.sel(resp="pitch_acc")
        yaw_acc = base.sel(resp="yaw_acc")

    x_acc, y_acc, z_acc = _point_translation_from_rigid_body(
        (surge_acc, sway_acc, heave_acc),
        (roll_acc, pitch_acc, yaw_acc),
        dx,
        dy,
        dz,
    )

    prefix = "point" if point_name is None else str(point_name)
    derived = xr.concat([x_acc, y_acc, z_acc], dim="resp").assign_coords(
        resp=("resp", [f"{prefix}_x_acc", f"{prefix}_y_acc", f"{prefix}_z_acc"])
    )

    return _build_output_dataset(
        ds,
        derived,
        point=point,
        point_mode=point_mode,
        point_name=point_name,
        relative_point=(dx, dy, dz),
    )
