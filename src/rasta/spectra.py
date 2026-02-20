"""Wave spectrum and directional spreading utilities.

Frequency is in rad/s. Direction is in degrees by default (rasta convention:
0 following sea, 180 head sea).

Reference formulas are aligned with the popcorn implementation:
https://github.com/kaufmann-jan/popcorn/blob/main/src/popcorn/signal/wave.py
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _as_1d_coord_array(values: xr.DataArray | np.ndarray | list[float], *, dim: str) -> xr.DataArray:
    if isinstance(values, xr.DataArray):
        if values.ndim != 1:
            raise ValueError(f"{dim} input must be 1-D")
        if dim in values.dims:
            out = values.rename(dim)
        else:
            out = values.rename({values.dims[0]: dim})
        out = out.astype(float)
        if dim not in out.coords:
            out = out.assign_coords({dim: out.values})
        return out

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{dim} input must be 1-D")
    return xr.DataArray(arr, dims=(dim,), coords={dim: arr})


def _validate_hs_tp(hs: float, tp: float) -> None:
    if hs <= 0.0:
        raise ValueError("hs must be > 0")
    if tp <= 0.0:
        raise ValueError("tp must be > 0")


def _normalize_to_hs(s: xr.DataArray, hs: float) -> xr.DataArray:
    target_m0 = (hs / 4.0) ** 2
    m0 = s.integrate(coord="freq")
    m0_val = float(m0.values)
    if not np.isfinite(m0_val) or m0_val <= 0.0:
        raise ValueError("spectrum integral is not positive/finite")
    return s * (target_m0 / m0_val)


def _sanitize_spectrum(s: xr.DataArray, omega: xr.DataArray) -> xr.DataArray:
    s = s.where(omega > 0.0, 0.0)
    s = s.where(np.isfinite(s), 0.0)
    return s


def bretschneider(freq: xr.DataArray | np.ndarray | list[float], *, hs: float, tp: float) -> xr.DataArray:
    """Bretschneider spectrum S(omega).

    Parameters
    - `freq`: angular frequency grid [rad/s]
    - `hs`: significant wave height [m]
    - `tp`: peak period [s]

    Returns
    - `xr.DataArray` over `freq` with spectral density [m^2 s].

    Reference
    - https://github.com/kaufmann-jan/popcorn/blob/main/src/popcorn/signal/wave.py
    """
    _validate_hs_tp(hs, tp)
    omega = _as_1d_coord_array(freq, dim="freq")

    wp = 2.0 * np.pi / tp
    core = omega**-5 * xr.apply_ufunc(np.exp, -1.25 * (wp / omega) ** 4)
    s = _sanitize_spectrum(core, omega)
    s = _normalize_to_hs(s, hs)
    s.name = "S_omega"
    s.attrs["model"] = "bretschneider"
    return s


def pierson_moskowitz(
    freq: xr.DataArray | np.ndarray | list[float],
    *,
    hs: float | None = None,
    tp: float | None = None,
    u10: float | None = None,
    g: float = 9.81,
) -> xr.DataArray:
    """Pierson-Moskowitz spectrum S(omega).

    Parameters
    - `freq`: angular frequency grid [rad/s]
    - Either (`hs`, `tp`) or `u10`.
      `hs`, `tp`: sea-state specification [m], [s]
      `u10`: 10 m wind speed [m/s]
    - `g`: gravity [m/s^2]

    Returns
    - `xr.DataArray` over `freq` with spectral density [m^2 s].

    Reference
    - https://github.com/kaufmann-jan/popcorn/blob/main/src/popcorn/signal/wave.py
    """
    omega = _as_1d_coord_array(freq, dim="freq")

    if u10 is None:
        if hs is None or tp is None:
            raise ValueError("provide either (hs, tp) or u10")
        _validate_hs_tp(hs, tp)
        wp = 2.0 * np.pi / tp
        core = omega**-5 * xr.apply_ufunc(np.exp, -1.25 * (wp / omega) ** 4)
        s = _sanitize_spectrum(core, omega)
        s = _normalize_to_hs(s, hs)
    else:
        if u10 <= 0.0:
            raise ValueError("u10 must be > 0")
        alpha = 8.1e-3
        beta = 0.74
        wp = 0.877 * g / u10
        core = alpha * g**2 * omega**-5 * xr.apply_ufunc(np.exp, -beta * (wp / omega) ** 4)
        s = _sanitize_spectrum(core, omega)
        if hs is not None:
            if hs <= 0.0:
                raise ValueError("hs must be > 0")
            s = _normalize_to_hs(s, hs)

    s.name = "S_omega"
    s.attrs["model"] = "pm"
    return s


def jonswap(
    freq: xr.DataArray | np.ndarray | list[float],
    *,
    hs: float,
    tp: float,
    gamma: float = 3.3,
) -> xr.DataArray:
    """JONSWAP spectrum S(omega).

    Parameters
    - `freq`: angular frequency grid [rad/s]
    - `hs`: significant wave height [m]
    - `tp`: peak period [s]
    - `gamma`: peak enhancement factor [-]

    Returns
    - `xr.DataArray` over `freq` with spectral density [m^2 s].

    Reference
    - https://github.com/kaufmann-jan/popcorn/blob/main/src/popcorn/signal/wave.py
    """
    return goda(freq, hs=hs, tp=tp, gamma=gamma, sigma_a=0.07, sigma_b=0.09)


def goda(
    freq: xr.DataArray | np.ndarray | list[float],
    *,
    hs: float,
    tp: float,
    gamma: float = 3.3,
    sigma_a: float = 0.07,
    sigma_b: float = 0.09,
) -> xr.DataArray:
    """Goda-type spectrum (JONSWAP family) S(omega).

    Parameters
    - `freq`: angular frequency grid [rad/s]
    - `hs`: significant wave height [m]
    - `tp`: peak period [s]
    - `gamma`: peak enhancement factor [-]
    - `sigma_a`: width parameter for omega <= omega_p [-]
    - `sigma_b`: width parameter for omega > omega_p [-]

    Returns
    - `xr.DataArray` over `freq` with spectral density [m^2 s].

    Reference
    - https://github.com/kaufmann-jan/popcorn/blob/main/src/popcorn/signal/wave.py
    """
    _validate_hs_tp(hs, tp)
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")
    if sigma_a <= 0.0 or sigma_b <= 0.0:
        raise ValueError("sigma_a and sigma_b must be > 0")

    omega = _as_1d_coord_array(freq, dim="freq")
    wp = 2.0 * np.pi / tp

    sigma = xr.where(omega <= wp, sigma_a, sigma_b)
    exponent = -((omega / wp - 1.0) ** 2) / (2.0 * sigma**2)
    peak = xr.apply_ufunc(np.power, gamma, xr.apply_ufunc(np.exp, exponent))

    core = omega**-5 * xr.apply_ufunc(np.exp, -1.25 * (wp / omega) ** 4) * peak
    s = _sanitize_spectrum(core, omega)
    s = _normalize_to_hs(s, hs)
    s.name = "S_omega"
    s.attrs["model"] = "goda"
    return s


def _spreading_cosn(
    direction: xr.DataArray | np.ndarray | list[float],
    *,
    mean_dir: float,
    power: int,
    dir_unit: str = "deg",
) -> xr.DataArray:
    d = _as_1d_coord_array(direction, dim="dir")
    unit = dir_unit.lower()

    if unit == "deg":
        theta = np.deg2rad(d)
        theta_mean = np.deg2rad(mean_dir)
        mean_deg = float(mean_dir)
    elif unit == "rad":
        theta = d
        theta_mean = float(mean_dir)
        mean_deg = float(np.rad2deg(mean_dir))
    else:
        raise ValueError("dir_unit must be 'deg' or 'rad'")

    delta = (theta - theta_mean + np.pi) % (2.0 * np.pi) - np.pi
    w = xr.apply_ufunc(np.cos, 0.5 * delta) ** power
    w = w.where(np.abs(delta) <= np.pi, 0.0)
    w = w.where(w > 0.0, 0.0)

    denom = np.trapezoid(w.values, theta.values)
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("directional spreading normalization failed")

    D = w / denom
    D.name = "D_dir"
    D.attrs["spreading"] = f"cos{power}"
    D.attrs["mean_dir_deg"] = mean_deg
    return D


def spreading_cos2(
    direction: xr.DataArray | np.ndarray | list[float],
    *,
    mean_dir: float,
    dir_unit: str = "deg",
) -> xr.DataArray:
    """Cosine-power directional spreading with power 2 (cos2).

    Parameters
    - `direction`: direction grid [deg] by default
    - `mean_dir`: mean direction in same unit as `direction`
    - `dir_unit`: `"deg"` or `"rad"`

    Returns
    - `xr.DataArray` over `dir`, normalized so integral over theta is 1.

    Reference
    - https://github.com/kaufmann-jan/popcorn/blob/main/src/popcorn/signal/wave.py
    """
    return _spreading_cosn(direction, mean_dir=mean_dir, power=2, dir_unit=dir_unit)


def spreading_cos4(
    direction: xr.DataArray | np.ndarray | list[float],
    *,
    mean_dir: float,
    dir_unit: str = "deg",
) -> xr.DataArray:
    """Cosine-power directional spreading with power 4 (cos4).

    Parameters
    - `direction`: direction grid [deg] by default
    - `mean_dir`: mean direction in same unit as `direction`
    - `dir_unit`: `"deg"` or `"rad"`

    Returns
    - `xr.DataArray` over `dir`, normalized so integral over theta is 1.

    Reference
    - https://github.com/kaufmann-jan/popcorn/blob/main/src/popcorn/signal/wave.py
    """
    return _spreading_cosn(direction, mean_dir=mean_dir, power=4, dir_unit=dir_unit)


def directional_spectrum(
    freq: xr.DataArray | np.ndarray | list[float],
    direction: xr.DataArray | np.ndarray | list[float],
    *,
    S_omega: xr.DataArray,
    D_dir: xr.DataArray,
) -> xr.DataArray:
    """Build directional spectrum S(dir, freq) = D(dir) * S(freq).

    Parameters
    - `freq`: frequency grid [rad/s]
    - `direction`: direction grid [deg]
    - `S_omega`: 1D frequency spectrum over `freq`
    - `D_dir`: directional spreading over `dir` (normalized)

    Returns
    - `xr.DataArray` with dims `(dir, freq)`.

    Reference
    - https://github.com/kaufmann-jan/popcorn/blob/main/src/popcorn/signal/wave.py
    """
    f = _as_1d_coord_array(freq, dim="freq")
    d = _as_1d_coord_array(direction, dim="dir")

    if "freq" not in S_omega.dims or S_omega.ndim != 1:
        raise ValueError("S_omega must be 1-D with dim 'freq'")
    if "dir" not in D_dir.dims or D_dir.ndim != 1:
        raise ValueError("D_dir must be 1-D with dim 'dir'")

    S = S_omega.reindex(freq=f.values)
    D = D_dir.reindex(dir=d.values)

    out = D * S
    out = out.transpose("dir", "freq")
    out.name = "S_dir_omega"
    return out


def make_directional_spectrum(
    freq: xr.DataArray | np.ndarray | list[float],
    direction: xr.DataArray | np.ndarray | list[float],
    *,
    model: str,
    hs: float,
    tp: float,
    mean_dir: float,
    spreading: str = "cos2",
    **kwargs: float,
) -> xr.DataArray:
    """Convenience builder for directional spectrum S(dir, freq).

    Parameters
    - `freq`: frequency grid [rad/s]
    - `direction`: direction grid [deg]
    - `model`: one of `bretschneider`, `pm`, `jonswap`, `goda`
    - `hs`, `tp`: sea-state parameters [m], [s]
    - `mean_dir`: mean wave direction [deg]
    - `spreading`: one of `cos2`, `cos4`

    Returns
    - `xr.DataArray` with dims `(dir, freq)`.

    Reference
    - https://github.com/kaufmann-jan/popcorn/blob/main/src/popcorn/signal/wave.py
    """
    model_key = model.lower()
    if model_key == "bretschneider":
        S_omega = bretschneider(freq, hs=hs, tp=tp)
    elif model_key == "pm":
        S_omega = pierson_moskowitz(freq, hs=hs, tp=tp)
    elif model_key == "jonswap":
        S_omega = jonswap(freq, hs=hs, tp=tp, gamma=float(kwargs.get("gamma", 3.3)))
    elif model_key == "goda":
        S_omega = goda(
            freq,
            hs=hs,
            tp=tp,
            gamma=float(kwargs.get("gamma", 3.3)),
            sigma_a=float(kwargs.get("sigma_a", 0.07)),
            sigma_b=float(kwargs.get("sigma_b", 0.09)),
        )
    else:
        raise ValueError("model must be one of: bretschneider, pm, jonswap, goda")

    spread_key = spreading.lower()
    if spread_key == "cos2":
        D_dir = spreading_cos2(direction, mean_dir=mean_dir, dir_unit="deg")
    elif spread_key == "cos4":
        D_dir = spreading_cos4(direction, mean_dir=mean_dir, dir_unit="deg")
    else:
        raise ValueError("spreading must be one of: cos2, cos4")

    return directional_spectrum(freq, direction, S_omega=S_omega, D_dir=D_dir)


# Backward-compatible names from previous version.
bretschneider_spectrum = bretschneider


def directional_spreading_cos2n(
    direction_deg: xr.DataArray | np.ndarray | list[float],
    mean_dir_deg: float,
    n: float,
) -> xr.DataArray:
    """Backward-compatible cosine-power spreading helper.

    Uses shape cos(delta/2)^(2n) and normalizes numerically over direction.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    d = _as_1d_coord_array(direction_deg, dim="dir")
    theta = np.deg2rad(d)
    theta_mean = np.deg2rad(mean_dir_deg)
    delta = (theta - theta_mean + np.pi) % (2.0 * np.pi) - np.pi
    w = xr.apply_ufunc(np.cos, 0.5 * delta) ** (2.0 * float(n))
    w = w.where(w > 0.0, 0.0)
    denom = np.trapezoid(w.values, theta.values)
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("directional spreading normalization failed")
    return w / denom
