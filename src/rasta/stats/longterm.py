"""Long-term response statistics from sea-state mixtures."""

from __future__ import annotations

import xarray as xr


def mixture_mean(values: xr.DataArray, weights: xr.DataArray, state_dim: str = "sea_state") -> xr.DataArray:
    """Weighted mixture mean across sea states."""
    w = weights / weights.sum(dim=state_dim)
    return (values * w).sum(dim=state_dim)


def mixture_exceedance_prob_rayleigh(x: xr.DataArray, sigma: xr.DataArray, weights: xr.DataArray, state_dim: str = "sea_state") -> xr.DataArray:
    """Weighted exceedance for Rayleigh response envelope: P(X>x)=exp(-x^2/(2 sigma^2))."""
    w = weights / weights.sum(dim=state_dim)
    p = xr.apply_ufunc(lambda xv, sv: __import__("numpy").exp(-(xv**2) / (2.0 * sv**2)), x, sigma)
    return (p * w).sum(dim=state_dim)
