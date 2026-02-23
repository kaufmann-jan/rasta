"""Response spectrum construction from RAOSet and wave spectra."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import xarray as xr

from ..rao import RAOSet
from ..spectra import bretschneider, goda, jonswap, pierson_moskowitz, spreading_cos2, spreading_cos4

_EVEN_RESP = {"surge", "heave", "pitch", "Fx", "Fz", "My"}
_ODD_RESP = {"sway", "roll", "yaw", "Fy", "Mx", "Mz"}


def _as_resp_list(resp: str | Iterable[str]) -> list[str]:
    if isinstance(resp, str):
        return [resp]
    return [str(r) for r in resp]


def _default_omega_grid(
    rs: RAOSet,
    omega_min: float | None,
    omega_max: float | None,
    domega: float | None,
) -> xr.DataArray:
    freq_vals = np.asarray(rs.dataset.coords["freq"].values, dtype=float)
    omin = max(0.05, float(freq_vals.min())) if omega_min is None else float(omega_min)
    omax = float(freq_vals.max()) if omega_max is None else float(omega_max)

    if omax <= omin:
        raise ValueError("omega_max must be greater than omega_min")

    if domega is None:
        domega = min(0.02, (omax - omin) / 2000.0)
        domega = max(float(domega), 1e-3)

    grid = np.arange(omin, omax + 0.5 * domega, domega, dtype=float)
    return xr.DataArray(grid, dims=("freq",), coords={"freq": grid})


def extend_symmetric_raos(rs: RAOSet, *, symmetry: bool = True) -> RAOSet:
    """Extend 0..180 RAOs to 0..360 using even/odd response parity rules."""
    if not symmetry:
        return rs

    ds = rs.dataset
    dir_vals = np.asarray(ds.coords["dir"].values, dtype=float)
    if dir_vals.size < 2:
        return rs

    span = float(np.max(dir_vals) - np.min(dir_vals))
    if span > 180.0 + 1e-9:
        return rs

    mirror_src = [d for d in dir_vals if d > 0.0 + 1e-9 and d < 180.0 - 1e-9]
    if not mirror_src:
        return rs

    mirrored_datasets: list[xr.Dataset] = [ds]
    for d in mirror_src:
        dm = float((360.0 - d) % 360.0)
        if np.any(np.isclose(dir_vals, dm)):
            continue

        slice_ds = ds.sel(dir=d).expand_dims(dir=[dm]).copy()
        resp_names = [str(r) for r in slice_ds.coords["resp"].values]
        sign = np.ones(len(resp_names), dtype=float)
        for i, name in enumerate(resp_names):
            if name in _ODD_RESP:
                sign[i] = -1.0
            elif name not in _EVEN_RESP:
                sign[i] = 1.0

        sign_da = xr.DataArray(sign, dims=("resp",), coords={"resp": slice_ds.coords["resp"]})
        slice_ds["rao"] = slice_ds["rao"] * sign_da
        mirrored_datasets.append(slice_ds)

    out = xr.concat(mirrored_datasets, dim="dir").sortby("dir")
    _, idx = np.unique(np.asarray(out.coords["dir"].values, dtype=float), return_index=True)
    out = out.isel(dir=np.sort(idx))
    return RAOSet(out)


def _interpolate_rao_to_grid(
    rao: xr.DataArray,
    omega_grid: xr.DataArray,
    mode: str,
) -> xr.DataArray:
    def _interp1d_nan(fp: np.ndarray, xp: np.ndarray, xnew: np.ndarray) -> np.ndarray:
        return np.interp(xnew, xp, fp, left=np.nan, right=np.nan)

    xp_da = xr.DataArray(np.asarray(rao.coords["freq"].values, dtype=float), dims=("freq",), coords={"freq": rao.coords["freq"]})
    xnew_da = xr.DataArray(np.asarray(omega_grid.values, dtype=float), dims=("freq_out",), coords={"freq_out": np.asarray(omega_grid.values, dtype=float)})

    real_i = xr.apply_ufunc(
        _interp1d_nan,
        np.real(rao),
        xp_da,
        xnew_da,
        input_core_dims=[["freq"], ["freq"], ["freq_out"]],
        output_core_dims=[["freq_out"]],
        vectorize=True,
        output_dtypes=[float],
    )
    imag_i = xr.apply_ufunc(
        _interp1d_nan,
        np.imag(rao),
        xp_da,
        xnew_da,
        input_core_dims=[["freq"], ["freq"], ["freq_out"]],
        output_core_dims=[["freq_out"]],
        vectorize=True,
        output_dtypes=[float],
    )
    rao_i = (real_i + 1j * imag_i).rename({"freq_out": "freq"}).assign_coords(freq=omega_grid.values)

    freq_src = np.asarray(rao.coords["freq"].values, dtype=float)
    fmin = float(freq_src.min())
    fmax = float(freq_src.max())
    fg = np.asarray(omega_grid.values, dtype=float)

    # Below minimum: constant extension.
    if np.any(fg < fmin):
        first = rao.isel(freq=0)
        low_mask = xr.DataArray(fg < fmin, dims=("freq",), coords={"freq": omega_grid.values})
        rao_i = xr.where(low_mask, first, rao_i)

    if mode != "zero_at_omega_max":
        return rao_i.fillna(0.0 + 0.0j)

    # Above maximum: taper magnitude linearly to zero at integration omega max.
    fgrid_max = float(fg.max())
    if fgrid_max <= fmax or not np.any(fg > fmax):
        return rao_i.fillna(0.0 + 0.0j)

    last = rao.sel(freq=fmax)
    last_mag = np.abs(last)
    last_phase = xr.apply_ufunc(np.angle, last)

    denom = max(fgrid_max - fmax, 1e-12)
    factor = np.clip(1.0 - (fg - fmax) / denom, 0.0, 1.0)
    factor_da = xr.DataArray(factor, dims=("freq",), coords={"freq": omega_grid.values})

    tail_mag = last_mag * factor_da
    tail = tail_mag * xr.apply_ufunc(np.exp, 1j * last_phase)

    high_mask = xr.DataArray(fg > fmax, dims=("freq",), coords={"freq": omega_grid.values})
    rao_i = xr.where(high_mask, tail, rao_i)
    return rao_i.fillna(0.0 + 0.0j)


def _wave_spectrum(
    model: str,
    freq: xr.DataArray,
    hs: float,
    tp: float,
    spectrum_kwargs: dict | None,
) -> xr.DataArray:
    kw = dict(spectrum_kwargs or {})
    key = model.lower()
    if key == "bretschneider":
        return bretschneider(freq, hs=hs, tp=tp)
    if key == "pm":
        return pierson_moskowitz(freq, hs=hs, tp=tp)
    if key == "jonswap":
        return jonswap(freq, hs=hs, tp=tp, gamma=float(kw.get("gamma", 3.3)))
    if key == "goda":
        return goda(
            freq,
            hs=hs,
            tp=tp,
            gamma=float(kw.get("gamma", 3.3)),
            sigma_a=float(kw.get("sigma_a", 0.07)),
            sigma_b=float(kw.get("sigma_b", 0.09)),
        )
    raise ValueError("spectrum_model must be one of: bretschneider, pm, jonswap, goda")


def compute_response_spectrum(
    rs: RAOSet,
    *,
    resp: str | list[str],
    hs: float,
    tp: float,
    mean_dir: float,
    spectrum_model: str = "jonswap",
    spectrum_kwargs: dict | None = None,
    spreading: str | None = None,
    spreading_kwargs: dict | None = None,
    symmetry: bool = True,
    omega_grid: xr.DataArray | None = None,
    omega_min: float | None = None,
    omega_max: float | None = None,
    domega: float | None = None,
    rao_tail_extrapolation: str = "zero_at_omega_max",
) -> xr.Dataset:
    """Compute response spectrum S_r(omega) for selected response channels."""
    if hs <= 0.0 or tp <= 0.0:
        raise ValueError("hs and tp must be > 0")

    rs_eff = extend_symmetric_raos(rs, symmetry=symmetry)
    resp_names = _as_resp_list(resp)

    missing = [r for r in resp_names if r not in set(rs_eff.dataset.coords["resp"].astype(str).values.tolist())]
    if missing:
        raise ValueError(f"responses not found: {missing}")

    rao = rs_eff.rao.sel(resp=resp_names)
    omega = omega_grid if omega_grid is not None else _default_omega_grid(rs_eff, omega_min, omega_max, domega)
    omega = omega.astype(float)

    rao_w = _interpolate_rao_to_grid(rao, omega, rao_tail_extrapolation)
    sea = _wave_spectrum(spectrum_model, omega, hs, tp, spectrum_kwargs)

    mean_dir_mod = float(mean_dir) % 360.0

    if spreading is None:
        dir_vals = np.asarray(rao_w.coords["dir"].values, dtype=float)
        idx = int(np.argmin(np.abs((dir_vals - mean_dir_mod + 180.0) % 360.0 - 180.0)))
        h_mean = rao_w.isel(dir=idx)
        s_r = (np.abs(h_mean) ** 2) * sea
    else:
        spread_key = spreading.lower()
        if spread_key == "cos2":
            D = spreading_cos2(rao_w.coords["dir"], mean_dir=mean_dir_mod, **(spreading_kwargs or {}))
        elif spread_key == "cos4":
            D = spreading_cos4(rao_w.coords["dir"], mean_dir=mean_dir_mod, **(spreading_kwargs or {}))
        else:
            raise ValueError("spreading must be one of: cos2, cos4")

        integrand = (np.abs(rao_w) ** 2) * sea * D
        theta = np.deg2rad(np.asarray(integrand.coords["dir"].values, dtype=float))

        s_r = xr.apply_ufunc(
            np.trapezoid,
            integrand,
            xr.DataArray(theta, dims=("dir",), coords={"dir": integrand.coords["dir"]}),
            input_core_dims=[["dir"], ["dir"]],
            output_core_dims=[[]],
            vectorize=True,
        )
        if "freq" not in s_r.dims:
            s_r = s_r.expand_dims(freq=omega.values)

    # Canonical order: resp, (any extra dims), freq
    non_freq_dims = [d for d in s_r.dims if d not in {"resp", "freq"}]
    desired = []
    if "resp" in s_r.dims:
        desired.append("resp")
    desired.extend(non_freq_dims)
    if "freq" in s_r.dims:
        desired.append("freq")
    s_r = s_r.transpose(*desired)

    ds_out = xr.Dataset({"S_r": s_r})
    ds_out = ds_out.assign_coords(hs=float(hs), tp=float(tp), mean_dir=float(mean_dir_mod))

    for cname in ("speed", "depth"):
        if cname in rs_eff.dataset.coords and rs_eff.dataset.coords[cname].ndim == 0:
            ds_out = ds_out.assign_coords({cname: float(rs_eff.dataset.coords[cname].values)})

    ds_out.attrs["spectrum_model"] = spectrum_model
    ds_out.attrs["spreading"] = "none" if spreading is None else spreading
    ds_out.attrs["symmetry"] = bool(symmetry)
    ds_out.attrs["rao_tail_extrapolation"] = rao_tail_extrapolation
    return ds_out
