"""Short-term response statistics."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..rao import RAOSet


def response_variance(raoset: RAOSet, spectrum: xr.DataArray) -> xr.DataArray:
    """Compute variance = integral |H|^2 S dω over freq."""
    integrand = (raoset.amp**2) * spectrum
    return integrand.integrate(coord="freq")


def response_std(raoset: RAOSet, spectrum: xr.DataArray) -> xr.DataArray:
    return response_variance(raoset, spectrum) ** 0.5


def significant_amplitude(raoset: RAOSet, spectrum: xr.DataArray) -> xr.DataArray:
    """4 * sigma for narrow-band Gaussian process."""
    return 4.0 * response_std(raoset, spectrum)


def shortterm_statistics(S_r: xr.DataArray, *, duration_s: float = 3600.0) -> xr.Dataset:
    """Compute short-term spectral moments and narrowband extreme estimates.

    Parameters
    - `S_r`: response spectrum over `freq` (may include extra dims such as `resp`)
    - `duration_s`: sea-state duration in seconds
    """
    if duration_s <= 0.0:
        raise ValueError("duration_s must be > 0")
    if "freq" not in S_r.dims:
        raise ValueError("S_r must have a 'freq' dimension")

    freq = S_r.coords["freq"]
    m0 = S_r.integrate("freq")
    m1 = (S_r * freq).integrate("freq")
    m2 = (S_r * (freq**2)).integrate("freq")

    eps = 1e-16
    m0_safe = xr.where(m0 > eps, m0, eps)
    m2_safe = xr.where(m2 > eps, m2, eps)

    sigma = np.sqrt(m0_safe)
    rms = sigma
    nu0 = (1.0 / (2.0 * np.pi)) * np.sqrt(m2_safe / m0_safe)
    Tz = 1.0 / xr.where(nu0 > eps, nu0, eps)
    Ncycles = duration_s / Tz

    lnN = xr.apply_ufunc(np.log, xr.where(Ncycles > 1.0 + eps, Ncycles, 1.0 + eps))
    root = np.sqrt(2.0 * lnN)
    X_mpm = sigma * root
    X_mean_max = sigma * (root + 0.5772156649015329 / xr.where(root > eps, root, eps))
    X_sig = 4.0 * sigma

    return xr.Dataset(
        {
            "m0": m0,
            "m1": m1,
            "m2": m2,
            "sigma": sigma,
            "rms": rms,
            "nu0": nu0,
            "Tz": Tz,
            "Ncycles": Ncycles,
            "X_mpm": X_mpm,
            "X_mean_max": X_mean_max,
            "X_sig": X_sig,
        },
        attrs={"duration_s": float(duration_s), "assumption": "Gaussian narrowband"},
    )
