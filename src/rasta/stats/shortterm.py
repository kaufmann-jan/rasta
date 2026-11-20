"""Short-term response statistics."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..rao import RAOSet
from ..preprocess.relative_motion import wave_number


def response_variance(raoset: RAOSet, spectrum: xr.DataArray) -> xr.DataArray:
    """Compute variance = integral |H|^2 S dω over freq."""
    integrand = (raoset.amp**2) * spectrum
    return integrand.integrate(coord="freq")


def response_std(raoset: RAOSet, spectrum: xr.DataArray) -> xr.DataArray:
    return response_variance(raoset, spectrum) ** 0.5


def significant_amplitude(raoset: RAOSet, spectrum: xr.DataArray) -> xr.DataArray:
    """4 * sigma for narrow-band Gaussian process."""
    return 4.0 * response_std(raoset, spectrum)


def _encounter_frequency(
    freq: xr.DataArray,
    direction: xr.DataArray,
    speed: float,
    depth: float,
) -> xr.DataArray:
    """Return signed omega_e = omega - k*U*cos(beta), retaining input grids."""
    k = wave_number(freq, depth=depth)
    encounter = freq - k * speed * np.cos(np.deg2rad(direction))
    encounter.name = "encounter_frequency"
    encounter.attrs["units"] = "rad/s"
    return encounter


def _encounter_inputs(S_r: xr.DataArray, raoset: RAOSet | None):
    if raoset is None:
        raise ValueError("frequency_domain='encounter' requires a raoset")
    ds = raoset.dataset
    scalars = {}
    for name in ("speed", "depth"):
        if name not in ds.coords:
            raise ValueError(f"encounter statistics require RAO coordinate '{name}'")
        if ds.coords[name].ndim != 0:
            raise ValueError(f"select a single scalar '{name}' before calculating encounter statistics")
        scalars[name] = float(ds.coords[name].values)
    speed, depth = scalars["speed"], scalars["depth"]
    if not np.isfinite(speed) or speed < 0:
        raise ValueError("speed must be finite and nonnegative")
    if np.isnan(depth) or depth <= 0:
        raise ValueError("depth must be positive and finite, or positive infinity")
    for source, obj in (("S_r", S_r), ("raoset", ds)):
        if "freq" not in obj.coords:
            raise ValueError(f"{source} requires an explicit 'freq' coordinate")
        values = np.asarray(obj.coords["freq"].values, dtype=float)
        if (obj.coords["freq"].dims != ("freq",) or values.size < 2 or
                not np.isfinite(values).all() or np.any(values <= 0) or
                np.any(np.diff(values) <= 0)):
            raise ValueError(f"{source} frequencies must be finite, positive and strictly increasing (at least two)")
    # The response spectrum can use an interpolated grid, including extrapolated
    # tails. Evaluate dispersion directly on that grid without aligning RAO data.

    rao_dirs = np.asarray(ds.coords["dir"].values, dtype=float)
    if rao_dirs.size == 0 or not np.isfinite(rao_dirs).all():
        raise ValueError("RAO direction values must be finite and nonempty")
    if "dir" in S_r.dims:
        if "dir" not in S_r.coords or S_r.coords["dir"].dims != ("dir",):
            raise ValueError("S_r requires a 1-D 'dir' coordinate")
        direction = S_r.coords["dir"]
        dirs = np.asarray(direction.values, dtype=float)
    else:
        if rao_dirs.size != 1:
            raise ValueError("ambiguous direction: select a single RAO direction when S_r has no 'dir' dimension")
        direction = xr.DataArray(float(rao_dirs[0]))
        dirs = np.asarray(S_r.coords["dir"].values if "dir" in S_r.coords else rao_dirs)
    if dirs.size == 0 or not np.isfinite(dirs).all():
        raise ValueError("response direction values must be finite and nonempty")
    distance = (dirs.reshape(-1, 1) - rao_dirs.reshape(1, -1) + 180) % 360 - 180
    if not np.all(np.any(np.isclose(distance, 0, rtol=0, atol=1e-8), axis=1)):
        raise ValueError("incompatible direction coordinates between S_r and raoset")
    for name, value in scalars.items():
        if name in S_r.coords:
            coord = S_r.coords[name]
            if coord.ndim != 0 or not np.isclose(float(coord.values), value):
                raise ValueError(f"incompatible '{name}' coordinate between S_r and raoset")
    encounter = _encounter_frequency(S_r.freq, direction, speed, depth)
    if (not np.isfinite(encounter.values).all() or np.any(encounter.values <= 0) or
            np.any(encounter.diff("freq").values <= 0)):
        raise ValueError("encounter frequencies must be strictly positive and strictly increasing; "
                         "this case requires special treatment of signed encounter frequencies "
                         "and potentially multiple spectral branches")
    return encounter, scalars


def shortterm_statistics(
    S_r: xr.DataArray,
    *,
    duration_s: float = 3600.0,
    frequency_domain: str = "intrinsic",
    raoset: RAOSet | None = None,
) -> xr.Dataset:
    """Compute short-term statistics from a response spectrum.

    The function assumes a linear, Gaussian, narrowband process and computes
    spectral moments and extreme-value proxies from `S_r(omega)`.

    Parameters
    - `S_r`: response spectrum over `freq` (may include extra dims such as `resp`)
    - `duration_s`: sea-state duration in seconds
    - `frequency_domain`: "intrinsic" (default) or "encounter"
    - `raoset`: required for encounter calculations; scalar coordinates `speed`
      [m/s] and `depth` [m] must be selected. Positive infinity means deep water.
      Encounter frequencies are evaluated on S_r's intrinsic frequency grid;
      it may differ from the original RAO grid, including extrapolated tails
      produced by compute_response_spectrum with its default settings.
      With no `dir` dimension in S_r, select a single RAO heading using a list
      (e.g. rs.sel(dir=[180.0])) to retain the required RAO dimension.

    Returns
    - `xr.Dataset` with:
      - `m0 = ∫ S_r dω`: zeroth moment (variance)
      - `m1 = ∫ ω S_r dω`: first moment
      - `m2 = ∫ ω² S_r dω`: second moment
      - `sigma = sqrt(m0)`: standard deviation of the response
      - `rms = sigma`: root-mean-square response
      - `nu0 = (1/(2π)) * sqrt(m2/m0)`: mean zero-upcrossing rate [1/s]
      - `Tz = 1/nu0 = 2π * sqrt(m0/m2)`: mean zero-upcrossing period [s]
      - `Ncycles = duration_s / Tz`: expected number of cycles in duration
      - `X_mpm = sigma * sqrt(2 ln(Ncycles))`: most probable maximum in duration, i.e. Rmax
      - `X_mean_max ≈ sigma * (sqrt(2 ln N) + 0.5772/sqrt(2 ln N))`:
        expected maximum in duration (Euler-Mascheroni correction)
      - `X_sig = 4*sigma`: significant-amplitude proxy (commonly used engineering metric)

    Notes
    - Encounter moments integrate omega_e**n * S_r over the original intrinsic
      frequency coordinate, with omega_e = omega - k*speed*cos(dir). Direction
      0 degrees is following seas and 180 degrees is head seas. Wave numbers
      use the existing finite-depth dispersion solver or k=omega**2/g in deep
      water (g=9.81 m/s^2). Each direction is handled separately when present.
    - No spectrum interpolation or Jacobian is required when integrating in
      intrinsic coordinates. Thus m0, sigma, rms and X_sig are unchanged;
      Tz, cycle counts and extreme proxies use the encounter moments instead.
    - Only strictly positive, strictly increasing encounter mappings are
      supported. Following/oblique seas with signed or nonmonotonic mappings
      require special treatment of multiple spectral branches and raise errors.
    - Inputs and their intrinsic-frequency coordinates are not modified.
    - Returned amplitudes (`X_mpm`, `X_mean_max`, `X_sig`, `sigma`, `rms`) are in the
      same response units as the RAO response channel represented by `S_r`.

    Example::

        stats = shortterm_statistics(
            S_r, duration_s=1800.0, frequency_domain="encounter", raoset=rs,
        )
    """
    if duration_s <= 0.0:
        raise ValueError("duration_s must be > 0")
    if "freq" not in S_r.dims:
        raise ValueError("S_r must have a 'freq' dimension")
    if frequency_domain not in {"intrinsic", "encounter"}:
        raise ValueError("frequency_domain must be 'intrinsic' or 'encounter'")

    freq = S_r.coords["freq"]
    attrs = {"duration_s": float(duration_s), "assumption": "Gaussian narrowband",
             "frequency_domain": frequency_domain}
    if frequency_domain == "encounter":
        freq, metadata = _encounter_inputs(S_r, raoset)
        attrs.update(metadata)
        attrs["encounter_frequency_formula"] = "omega_e = omega - k*speed*cos(dir)"
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
        attrs=attrs,
    )
