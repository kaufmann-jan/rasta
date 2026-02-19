"""Wave spectrum utilities."""

from __future__ import annotations

import numpy as np
import xarray as xr


def bretschneider_spectrum(freq: xr.DataArray, hs: float, tp: float) -> xr.DataArray:
    """Bretschneider spectrum in angular frequency domain.

    Returns spectral density S(omega) with dimensions of freq.
    """
    omega = xr.DataArray(freq, dims=("freq",), coords={"freq": freq})
    wp = 2.0 * np.pi / tp
    a = 5.0 / 16.0 * hs**2 * wp**4
    b = 5.0 / 4.0 * wp**4
    s = a * omega**-5 * xr.apply_ufunc(np.exp, -b * omega**-4)
    return s.where(omega > 0.0, 0.0)


def directional_spreading_cos2n(direction_deg: xr.DataArray, mean_dir_deg: float, n: float) -> xr.DataArray:
    """Unnormalized cosine-2n directional spreading around mean direction."""
    theta = np.deg2rad(direction_deg - mean_dir_deg)
    spread = np.cos(0.5 * theta) ** (2.0 * n)
    return spread.where(np.abs(theta) <= np.pi, 0.0)
