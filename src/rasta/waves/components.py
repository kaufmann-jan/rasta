"""Discretize existing spectra into regular components for CFD wave generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import warnings

import numpy as np
import xarray as xr

from .. import spectra
from ..preprocess.relative_motion import wave_number

# Independent of the number of components; kept here for internal tuning.
_INTEGRATION_POINTS = 20001
_REFERENCE_MIN_RATIO = 0.02
_REFERENCE_MAX_RATIO = 128.0
_CAPTURE_TARGET = 0.99
_CSV_FIELDS = (
    "amplitude", "height", "freq", "period", "phase", "wave_number", "wavelength"
)
_UNITS = {
    "amplitude": "m", "height": "m", "freq": "rad/s", "period": "s",
    "phase": "rad", "wave_number": "rad/m", "wavelength": "m",
    "energy": "m^2", "omega_lower": "rad/s", "omega_upper": "rad/s",
}


def _cumulative(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(
        0.5 * (values[1:] + values[:-1]) * np.diff(grid)
    )))


def generate_wave_components(
    *,
    hs: float,
    tp: float,
    spectrum_model: str = "jonswap",
    n_components: int = 50,
    discretization: str = "equal_energy",
    omega_min: float | None = None,
    omega_max: float | None = None,
    depth: float = np.inf,
    g: float = 9.81,
    seed: int | None = None,
    spectrum_kwargs: dict | None = None,
    normalize_energy: bool = True,
) -> xr.Dataset:
    """Generate a long-crested linear irregular wave train in SI units.

    Equal-energy bins use energy-weighted mean angular frequencies; equally
    spaced frequency bins use bin centers. Both integrate bin energy on a fine
    grid independent of ``n_components``. Amplitudes satisfy E_i = a_i**2/2;
    individual component heights are 2*a_i. Frequencies are intrinsic [rad/s]
    and wave numbers are [rad/m]. ``depth=np.inf`` selects deep water.

    Initial automatic limits are 0.4 and 3 times the peak frequency. Automatic
    limits expand until at least 99% of reference variance is captured. Explicit
    limits are respected. The reference spectrum is integrated from 0.02 to
    128 times the peak frequency, extended to contain any explicit limits.
    ``captured_energy_fraction`` reports variance before normalization.
    Normalization rescales amplitudes to reproduce ``hs`` exactly; strong
    truncation followed by normalization changes the energy distribution.

    Independent phases are uniform in [0, 2*pi), using a local random generator.
    The convention is eta(x,t) = sum(a*cos(k*x - omega*t + phase)). Random
    phases change individual events but preserve theoretical variance. Finite
    realizations can have different sample statistics and need not repeat.
    ``T_beat`` indicates the longest adjacent-frequency beating period, rather
    than an exact recurrence time. One component has undefined spacing/beat
    diagnostics (NaN).

    Example::

        components = generate_wave_components(hs=2.0, tp=8.0, depth=10, seed=42)
        eta = wave_elevation(components, np.linspace(0, 600, 6001))
    """
    for name, value in (("hs", hs), ("tp", tp), ("g", g)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if (isinstance(n_components, (bool, np.bool_)) or
            not isinstance(n_components, (int, np.integer)) or n_components < 1):
        raise ValueError("n_components must be an integer >= 1")
    if np.isnan(depth) or depth <= 0:
        raise ValueError("depth must be > 0 (np.inf is allowed)")
    if discretization not in {"equal_energy", "equal_frequency"}:
        raise ValueError("invalid discretization; use equal_energy or equal_frequency")
    models = {"bretschneider": spectra.bretschneider, "pm": spectra.pierson_moskowitz,
              "pierson_moskowitz": spectra.pierson_moskowitz,
              "jonswap": spectra.jonswap, "goda": spectra.goda}
    model = spectrum_model.lower()
    if model not in models:
        raise ValueError("invalid spectrum_model; use bretschneider, pm, jonswap or goda")
    kwargs = dict(spectrum_kwargs or {})
    kwargs_json = json.dumps(kwargs, sort_keys=True, allow_nan=False)
    wp = 2 * np.pi / tp
    lower = 0.4 * wp if omega_min is None else float(omega_min)
    upper = 3.0 * wp if omega_max is None else float(omega_max)
    if not np.isfinite([lower, upper]).all() or lower <= 0 or lower >= upper:
        raise ValueError("require finite 0 < omega_min < omega_max")

    ref = np.geomspace(min(_REFERENCE_MIN_RATIO * wp, lower),
                       max(_REFERENCE_MAX_RATIO * wp, upper), _INTEGRATION_POINTS)
    density = np.asarray(models[model](ref, hs=hs, tp=tp, **kwargs), dtype=float)
    cumulative = _cumulative(density, ref)
    reference_energy = cumulative[-1]
    if not np.isfinite(reference_energy) or reference_energy <= 0:
        raise ValueError("reference spectrum must have positive finite energy")
    # Check convergence using the variance beyond half the reference upper bound.
    tail = reference_energy - np.interp(ref[-1] / 2, ref, cumulative)
    if tail / reference_energy > 1e-6:
        warnings.warn("Reference spectral integration may not be converged", UserWarning, stacklevel=2)
    for _ in range(20):
        below = np.interp(lower, ref, cumulative) / reference_energy
        above = 1 - np.interp(upper, ref, cumulative) / reference_energy
        captured = 1 - below - above
        if captured >= _CAPTURE_TARGET:
            break
        previous_limits = lower, upper
        if omega_min is None and below > (1 - _CAPTURE_TARGET) / 2:
            lower = max(ref[0], lower / 1.5)
        if omega_max is None and above > (1 - _CAPTURE_TARGET) / 2:
            upper = min(ref[-1], upper * 1.5)
        if previous_limits == (lower, upper):
            break

    # Evaluate on the union so existing spectral normalization uses the entire
    # reference support, not just the truncated component range.
    grid = np.linspace(lower, upper, _INTEGRATION_POINTS)
    union = np.unique(np.concatenate((ref, grid)))
    full_density = np.asarray(models[model](union, hs=hs, tp=tp, **kwargs), dtype=float)
    density = np.interp(grid, union, full_density)
    cumulative = _cumulative(density, grid)
    selected_energy = cumulative[-1]
    reference_energy = _cumulative(full_density, union)[-1]
    captured = selected_energy / reference_energy
    if captured < _CAPTURE_TARGET:
        warnings.warn(f"Selected frequency range captures only {captured:.2%} of spectral variance; "
                      "normalization rescales the retained spectrum when enabled",
                      UserWarning, stacklevel=2)
    if not np.isfinite(selected_energy) or selected_energy <= 0:
        raise ValueError("selected frequency range contains no spectral energy")
    if discretization == "equal_energy":
        targets = np.linspace(0, selected_energy, n_components + 1)
        edges = np.interp(targets, cumulative, grid)
        edges[0], edges[-1] = lower, upper
        energy = np.full(n_components, selected_energy / n_components)
        moment = _cumulative(grid * density, grid)
        freq = np.diff(np.interp(edges, grid, moment)) / energy
    else:
        edges = np.linspace(lower, upper, n_components + 1)
        freq = 0.5 * (edges[:-1] + edges[1:])
        energy = np.diff(np.interp(edges, grid, cumulative))
    if (not np.isfinite(freq).all() or np.any(np.diff(freq) <= 0) or
            np.any(energy <= 0) or not np.isfinite(energy).all()):
        raise ValueError("frequency range/resolution cannot resolve positive distinct components")
    factor = np.sqrt((hs / 4)**2 / energy.sum()) if normalize_energy else 1.0
    amplitude = np.sqrt(2 * energy) * factor
    energy = 0.5 * amplitude**2
    k = np.asarray(wave_number(freq, depth=depth, g=g), dtype=float)
    spacing = float(np.min(np.diff(freq))) if n_components > 1 else float("nan")
    values = dict(amplitude=amplitude, height=2 * amplitude, freq=freq,
                  period=2 * np.pi / freq,
                  phase=np.random.default_rng(seed).uniform(0, 2 * np.pi, n_components),
                  wave_number=k, wavelength=2 * np.pi / k, energy=energy,
                  omega_lower=edges[:-1], omega_upper=edges[1:])
    ds = xr.Dataset({name: ("component", np.asarray(value, dtype=np.float64),
                            {"units": _UNITS[name]}) for name, value in values.items()},
                    coords={"component": np.arange(n_components)})
    ds.attrs.update(hs=float(hs), tp=float(tp), spectrum_model=model,
                    discretization=discretization, n_components=int(n_components),
                    omega_min=lower, omega_max=upper, depth=float(depth), gravity=float(g),
                    seed="None" if seed is None else int(seed),
                    normalize_energy=int(normalize_energy), normalization_factor=float(factor),
                    captured_energy_fraction=float(captured), m0_discrete=float(energy.sum()),
                    hs_discrete=float(4 * np.sqrt(energy.sum())), spectrum_kwargs=kwargs_json,
                    minimum_frequency_spacing=spacing, T_beat=float(2 * np.pi / spacing))
    return ds


def wave_elevation(
    components: xr.Dataset,
    time: np.ndarray,
    *,
    x: float = 0.0,
    phase_sign: str = "negative_time",
) -> xr.DataArray:
    """Return eta(time) [m] = sum(a*cos(k*x - omega*t + phase)).

    ``positive_time`` uses +omega*t instead. Coordinates and propagation
    direction for CFD export are chosen by the target wave generator.
    """
    if phase_sign not in {"negative_time", "positive_time"}:
        raise ValueError("phase_sign must be negative_time or positive_time")
    times = np.asarray(time, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or not np.isfinite(x):
        raise ValueError("time must be a finite 1-D array and x must be finite")
    t = xr.DataArray(times, dims="time", coords={"time": times})
    sign = -1 if phase_sign == "negative_time" else 1
    eta = (components.amplitude * np.cos(
        components.wave_number * x + sign * components.freq * t + components.phase
    )).sum("component")
    eta.name = "eta"
    eta.attrs.update(units="m", phase_sign=phase_sign)
    eta.coords["time"].attrs["units"] = "s"
    return eta


def write_wave_components_csv(components: xr.Dataset, path: str | Path) -> None:
    """Write SI component values to a plain CSV; dataset metadata is omitted."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("component", *_CSV_FIELDS))
        writer.writerows(zip(components.component.values,
                             *(components[name].values for name in _CSV_FIELDS)))


def read_wave_components_csv(path: str | Path) -> xr.Dataset:
    """Read component CSV values and restore SI units (metadata is unavailable)."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["component", *_CSV_FIELDS]:
            raise ValueError("unexpected wave-component CSV columns")
        rows = list(reader)
    return xr.Dataset(
        {name: ("component", np.array([float(row[name]) for row in rows], dtype=np.float64),
                {"units": _UNITS[name]}) for name in _CSV_FIELDS},
        coords={"component": np.array([int(row["component"]) for row in rows])},
    )
