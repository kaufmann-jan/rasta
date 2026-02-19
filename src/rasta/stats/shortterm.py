"""Short-term response statistics."""

from __future__ import annotations

import xarray as xr

from ..rao import RAOSet


def response_variance(raoset: RAOSet, spectrum: xr.DataArray) -> xr.DataArray:
    """Compute variance = integral |H|^2 S dω over freq.

    `spectrum` must have a `freq` coordinate; any extra dimensions are broadcast.
    """
    integrand = (raoset.amp**2) * spectrum
    return integrand.integrate(coord="freq")


def response_std(raoset: RAOSet, spectrum: xr.DataArray) -> xr.DataArray:
    return response_variance(raoset, spectrum) ** 0.5


def significant_amplitude(raoset: RAOSet, spectrum: xr.DataArray) -> xr.DataArray:
    """4 * sigma for narrow-band Gaussian process."""
    return 4.0 * response_std(raoset, spectrum)
