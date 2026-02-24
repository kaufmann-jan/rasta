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


def _dir_theta(direction: xr.DataArray | np.ndarray | list[float], *, dir_unit: str) -> xr.DataArray:
    d = _as_1d_coord_array(direction, dim="dir")
    unit = dir_unit.lower()
    if unit == "deg":
        return np.deg2rad(d)
    elif unit == "rad":
        return d
    else:
        raise ValueError("dir_unit must be 'deg' or 'rad'")


def _normalize_directional(w: xr.DataArray, theta: xr.DataArray) -> xr.DataArray:
    if int(theta.sizes.get("dir", 0)) == 1:
        return xr.ones_like(w, dtype=float)

    if "freq" in w.dims:
        denom = xr.apply_ufunc(
            np.trapezoid,
            w,
            theta,
            input_core_dims=[["dir"], ["dir"]],
            output_core_dims=[[]],
            vectorize=True,
            output_dtypes=[float],
        )
        denom = xr.where(denom > 0.0, denom, np.nan)
        D = w / denom
    else:
        denom = float(np.trapezoid(np.asarray(w.values, dtype=float), np.asarray(theta.values, dtype=float)))
        if not np.isfinite(denom) or denom <= 0.0:
            raise ValueError("directional spreading normalization failed")
        D = w / denom

    D = D.where(np.isfinite(D), 0.0)
    return D


def spreading_cos2s_full(
    direction: xr.DataArray | np.ndarray | list[float],
    *,
    mean_dir: float,
    s: float = 1.0,
    dir_unit: str = "deg",
) -> xr.DataArray:
    """Cosine-2s directional spreading with full +/-180° support (ITTC).

    Implements D(theta)=C(s) cos^(2s)((theta-theta_mean)/2) for |delta|<=pi,
    normalized numerically on the provided discrete direction grid.
    """
    if s < 0.0:
        raise ValueError("s must be >= 0")

    theta = _dir_theta(direction, dir_unit=dir_unit)
    theta_mean = np.deg2rad(mean_dir) if dir_unit.lower() == "deg" else float(mean_dir)
    mean_deg = float(mean_dir) if dir_unit.lower() == "deg" else float(np.rad2deg(mean_dir))

    delta = (theta - theta_mean + np.pi) % (2.0 * np.pi) - np.pi
    w = xr.apply_ufunc(np.cos, 0.5 * delta) ** (2.0 * float(s))
    w = w.where(np.abs(delta) <= np.pi, 0.0)
    w = w.where(w > 0.0, 0.0)
    D = _normalize_directional(w, theta)
    D.name = "D_dir"
    D.attrs.update({"spreading": "cos2s_full", "mean_dir_deg": mean_deg, "s": float(s)})
    return D


def spreading_cosN_half(
    direction: xr.DataArray | np.ndarray | list[float],
    *,
    mean_dir: float,
    N: float = 2.0,
    dir_unit: str = "deg",
) -> xr.DataArray:
    """Cosine-N directional spreading with +/-90° support (ITTC)."""
    if N < 0.0:
        raise ValueError("N must be >= 0")

    theta = _dir_theta(direction, dir_unit=dir_unit)
    theta_mean = np.deg2rad(mean_dir) if dir_unit.lower() == "deg" else float(mean_dir)
    mean_deg = float(mean_dir) if dir_unit.lower() == "deg" else float(np.rad2deg(mean_dir))

    delta = (theta - theta_mean + np.pi) % (2.0 * np.pi) - np.pi
    w = xr.apply_ufunc(np.cos, delta) ** float(N)
    w = w.where(np.abs(delta) <= (0.5 * np.pi), 0.0)
    w = w.where(w > 0.0, 0.0)
    D = _normalize_directional(w, theta)
    D.name = "D_dir"
    D.attrs.update({"spreading": "cosN_half", "mean_dir_deg": mean_deg, "N": float(N)})
    return D


def spreading_mitsuyasu(
    direction: xr.DataArray | np.ndarray | list[float],
    freq: xr.DataArray | np.ndarray | list[float],
    *,
    mean_dir: float,
    s_p : float,
    omega_p: float | None = None,
    tp: float | None = None,
    dir_unit: str = "deg",
) -> xr.DataArray:
    """Mitsuyasu frequency-dependent spreading (ITTC) returning D(dir, freq).

    The key parameter is
    
    s_p = s(ω_p)
    
    i.e. the spreading exponent at the spectral peak frequency. 
    Spreading widens away from ω_p according to Mitsuyasu law
    
    Typical values depend on sea state maturity (wind sea vs swell) and are reasonably well established in offshore practice.
    
    
    | Sea type         | Typical s_p |
    | ---------------- | ----------- |
    | Broad / confused | 2 – 5       |
    | Moderate wind    | 5 – 15      |
    | Narrow wind sea  | 15 – 30     |
    | Swell            | 25 – 75     |
    
    
    Default: s_p = 10

    
    """
    if s_p < 0.0:
        raise ValueError("s_p must be >= 0")
    if omega_p is None:
        if tp is None or tp <= 0.0:
            raise ValueError("provide omega_p or a positive tp")
        omega_p = 2.0 * np.pi / float(tp)
    if omega_p <= 0.0:
        raise ValueError("omega_p must be > 0")

    theta = _dir_theta(direction, dir_unit=dir_unit)
    f = _as_1d_coord_array(freq, dim="freq").astype(float)
    theta_mean = np.deg2rad(mean_dir) if dir_unit.lower() == "deg" else float(mean_dir)
    mean_deg = float(mean_dir) if dir_unit.lower() == "deg" else float(np.rad2deg(mean_dir))

    delta = (theta - theta_mean + np.pi) % (2.0 * np.pi) - np.pi
    ratio = f / float(omega_p)
    s_omega = xr.where(f <= omega_p, float(s_p) * ratio**5, float(s_p) * ratio ** (-2.5))

    w = xr.apply_ufunc(np.cos, 0.5 * delta) ** (2.0 * s_omega)
    w = w.where(np.abs(delta) <= np.pi, 0.0)
    w = w.where(w > 0.0, 0.0)
    w = w.transpose("dir", "freq")
    D = _normalize_directional(w, theta)
    D.name = "D_dir"
    D.attrs.update({"spreading": "mitsuyasu", "mean_dir_deg": mean_deg, "s_p": float(s_p), "omega_p": float(omega_p)})
    return D


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
    if "dir" not in D_dir.dims:
        raise ValueError("D_dir must include dim 'dir'")
    if D_dir.ndim not in (1, 2):
        raise ValueError("D_dir must be 1-D (dir) or 2-D (dir,freq)")
    if D_dir.ndim == 2 and "freq" not in D_dir.dims:
        raise ValueError("2-D D_dir must have dims including 'freq'")

    S = S_omega.reindex(freq=f.values)
    D = D_dir.reindex(dir=d.values)
    if "freq" in D.dims:
        D = D.reindex(freq=f.values)

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
    spreading: str = "cos2s_full",
    **kwargs: float,
) -> xr.DataArray:
    """Convenience builder for directional spectrum S(dir, freq).

    Parameters
    - `freq`: frequency grid [rad/s]
    - `direction`: direction grid [deg]
    - `model`: one of `bretschneider`, `pm`, `jonswap`, `goda`
    - `hs`, `tp`: sea-state parameters [m], [s]
    - `mean_dir`: mean wave direction [deg]
    - `spreading`: one of `cos2s_full`, `cosN_half`, `mitsuyasu`

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
    if spread_key == "cos2s_full":
        D_dir = spreading_cos2s_full(direction, mean_dir=mean_dir, s=float(kwargs.get("s", 1.0)), dir_unit="deg")
    elif spread_key == "cosn_half":
        D_dir = spreading_cosN_half(direction, mean_dir=mean_dir, N=float(kwargs.get("N", 2.0)), dir_unit="deg")
    elif spread_key == "mitsuyasu":
        if "s_p" not in kwargs:
            raise ValueError("mitsuyasu spreading requires parameter 's_p'")
        D_dir = spreading_mitsuyasu(
            direction,
            freq,
            mean_dir=mean_dir,
            s_p=float(kwargs["s_p"]),
            omega_p=kwargs.get("omega_p"),
            tp=kwargs.get("tp", tp),
            dir_unit="deg",
        )
    else:
        raise ValueError("spreading must be one of: cos2s_full, cosN_half, mitsuyasu")

    return directional_spectrum(freq, direction, S_omega=S_omega, D_dir=D_dir)


bretschneider_spectrum = bretschneider
