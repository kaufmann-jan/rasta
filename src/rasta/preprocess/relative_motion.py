"""Relative vertical motion preprocessing utilities."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..rao import RAOSet
from .points import _build_output_dataset, _resolve_point_relative_coordinates


def wave_number(
    freq,
    *,
    depth: float | None = None,
    g: float = 9.81,
    deep_water: bool = False,
):
    """Return wave number `k` for angular frequency `freq`."""
    omega = xr.DataArray(freq) if not isinstance(freq, xr.DataArray) else freq.astype(float)
    if g <= 0.0:
        raise ValueError("g must be > 0")
    if deep_water or depth is None or np.isinf(depth):
        return (omega**2) / g
    if depth <= 0.0:
        raise ValueError("depth must be > 0")

    omega_vals = np.asarray(omega.values, dtype=float)
    k = np.maximum((omega_vals**2) / g, 1e-12)
    for _ in range(30):
        tanh_kh = np.tanh(k * depth)
        f = g * k * tanh_kh - omega_vals**2
        df = g * tanh_kh + g * k * depth * (1.0 - tanh_kh**2)
        k_new = k - f / np.maximum(df, 1e-12)
        if np.allclose(k_new, k, rtol=1e-12, atol=1e-12):
            k = k_new
            break
        k = np.maximum(k_new, 1e-12)
    return xr.DataArray(k, dims=omega.dims, coords=omega.coords)


def _validate_relative_motion_inputs(ds: xr.Dataset) -> None:
    resp_vals = {str(v) for v in ds.coords["resp"].values.tolist()}
    missing = sorted({"heave", "roll", "pitch"} - resp_vals)
    if missing:
        raise ValueError(f"missing required motion responses: {missing}")
    if ds.attrs.get("angle_unit") != "rad":
        raise ValueError("roll and pitch must be in radians (dataset attr angle_unit='rad')")
    if "dir_convention" not in ds.attrs:
        raise ValueError("direction convention metadata missing")


def _resolve_absolute_point(ds: xr.Dataset, point: tuple[float, float, float], point_mode: str) -> tuple[float, float, float]:
    mode = point_mode.lower()
    x, y, z = (float(v) for v in point)
    if mode == "absolute":
        return x, y, z
    if mode != "relative":
        raise ValueError("point_mode must be either 'relative' or 'absolute'")
    missing = [name for name in ("xref", "yref", "zref") if name not in ds.attrs]
    if missing:
        raise ValueError(f"relative point_mode requires dataset attrs {missing} for wave phase evaluation")
    return x + float(ds.attrs["xref"]), y + float(ds.attrs["yref"]), z + float(ds.attrs["zref"])


def _vertical_point_motion(ds: xr.Dataset, *, dx: float, dy: float) -> xr.DataArray:
    rao = ds["rao"]
    return rao.sel(resp="heave") + rao.sel(resp="roll") * dy - rao.sel(resp="pitch") * dx


def incident_wave_elevation(
    rs: RAOSet,
    *,
    point: tuple[float, float, float],
    point_mode: str = "relative",
    water_depth: float | None = None,
    deep_water: bool = False,
    point_name: str | None = None,
) -> RAOSet:
    """Derive incident-wave elevation RAO at a point."""
    ds = rs.dataset
    _validate_relative_motion_inputs(ds)
    missing = [name for name in ("x_wave_ref", "y_wave_ref", "z_wave_ref") if name not in ds.attrs]
    if missing:
        raise ValueError(f"incident-wave reference point missing: {missing}")

    dx, dy, dz = _resolve_point_relative_coordinates(ds, point, point_mode)
    x_abs, y_abs, _ = _resolve_absolute_point(ds, point, point_mode)
    dx_wave = x_abs - float(ds.attrs["x_wave_ref"])
    dy_wave = y_abs - float(ds.attrs["y_wave_ref"])

    depth = water_depth
    if depth is None and "depth" in ds.coords and ds.coords["depth"].ndim == 0:
        depth = float(ds.coords["depth"].values)
    if not deep_water and depth is None:
        raise ValueError("finite depth requested but depth unavailable")

    k = wave_number(ds.coords["freq"], depth=depth, deep_water=deep_water)
    theta_from = np.deg2rad(xr.DataArray(ds.coords["dir"], dims=("dir",), coords={"dir": ds.coords["dir"]}))
    kx = -k * xr.apply_ufunc(np.cos, theta_from)
    ky = -k * xr.apply_ufunc(np.sin, theta_from)
    phi = kx * dx_wave + ky * dy_wave
    eta = xr.apply_ufunc(np.exp, 1j * phi).transpose("dir", "freq")

    prefix = "point" if point_name is None else str(point_name)
    derived = eta.expand_dims(resp=[f"{prefix}_eta"])
    out = xr.concat([rs.rao, derived], dim="resp")
    result = _build_output_dataset(
        ds,
        out,
        point=point,
        point_mode=point_mode,
        point_name=point_name,
        relative_point=(dx, dy, dz),
    )
    result.dataset.attrs["wave_reference_point"] = (
        float(ds.attrs["x_wave_ref"]),
        float(ds.attrs["y_wave_ref"]),
        float(ds.attrs["z_wave_ref"]),
    )
    result.dataset.attrs["wave_number_model"] = "deep_water" if deep_water or depth is None else "finite_depth_dispersion"
    result.dataset.attrs["wave_direction_convention"] = "from-direction, 0 following, 180 head"
    result.dataset.attrs["wave_phase_formula"] = "phi = kx*(x-x_wave_ref) + ky*(y-y_wave_ref), kx=-k*cos(dir), ky=-k*sin(dir)"
    return result


def relative_vertical_motion(
    rs: RAOSet,
    *,
    point: tuple[float, float, float],
    point_mode: str = "relative",
    water_depth: float | None = None,
    deep_water: bool = False,
    point_name: str | None = None,
) -> RAOSet:
    """Derive point vertical motion, incident-wave elevation, and relative vertical motion."""
    ds = rs.dataset
    _validate_relative_motion_inputs(ds)
    dx, dy, dz = _resolve_point_relative_coordinates(ds, point, point_mode)
    z_point = _vertical_point_motion(ds, dx=dx, dy=dy)
    eta_rs = incident_wave_elevation(
        rs,
        point=point,
        point_mode=point_mode,
        water_depth=water_depth,
        deep_water=deep_water,
        point_name=point_name,
    )
    prefix = "point" if point_name is None else str(point_name)
    eta = eta_rs.rao.sel(resp=f"{prefix}_eta")
    z_rel = z_point - eta

    derived = xr.concat(
        [
            z_point.expand_dims(resp=[f"{prefix}_z"]),
            eta.expand_dims(resp=[f"{prefix}_eta"]),
            z_rel.expand_dims(resp=[f"{prefix}_z_rel"]),
        ],
        dim="resp",
    )
    out = xr.concat([rs.rao, derived], dim="resp")
    result = _build_output_dataset(
        ds,
        out,
        point=point,
        point_mode=point_mode,
        point_name=point_name,
        relative_point=(dx, dy, dz),
    )
    result.dataset.attrs["wave_reference_point"] = eta_rs.dataset.attrs["wave_reference_point"]
    result.dataset.attrs["wave_number_model"] = eta_rs.dataset.attrs["wave_number_model"]
    result.dataset.attrs["wave_direction_convention"] = eta_rs.dataset.attrs["wave_direction_convention"]
    result.dataset.attrs["wave_phase_formula"] = eta_rs.dataset.attrs["wave_phase_formula"]
    return result
