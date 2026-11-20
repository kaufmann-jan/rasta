"""Linear response reconstruction using a prescribed discrete wave realization."""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from ..preprocess.relative_motion import wave_number
from ..rao import RAOSet

_MAX_EXPONENTIAL_ELEMENTS = 1_000_000
_PHASE_CONVENTIONS = {"negative_time", "positive_time"}


def _scalar_coordinate(ds: xr.Dataset, name: str, *, required: bool = False):
    if name not in ds.coords:
        if required:
            raise ValueError(f"encounter reconstruction requires RAO coordinate '{name}'")
        return None
    if ds.coords[name].ndim != 0:
        raise ValueError(f"select a single scalar '{name}' before reconstruction")
    return float(ds.coords[name].values)


def _validate_depth(value: float) -> float:
    value = float(value)
    if np.isnan(value) or value <= 0:
        raise ValueError("depth must be positive and finite or positive infinity")
    return value


def _check_phase_reference(ds: xr.Dataset, components: xr.Dataset, confirmed: bool) -> bool:
    if confirmed:
        return True
    # Generated phases refer to the wave origin. Explicit x/y metadata can
    # describe an origin established externally without applying a phase shift.
    wave_origin = [float(components.attrs.get(key, 0.0)) for key in ("x_wave_ref", "y_wave_ref")]
    known = all(key in ds.attrs for key in ("x_wave_ref", "y_wave_ref"))
    aligned = known and np.allclose(
        [ds.attrs["x_wave_ref"], ds.attrs["y_wave_ref"]], wave_origin,
        rtol=0, atol=1e-8,
    )
    if not aligned:
        warnings.warn(
            "Wave-component and RAO incident-wave phase references cannot be established "
            "as coincident. Ensure spatial references and CFD time origin agree; set "
            "phase_reference_confirmed=True after establishing alignment. No spatial "
            "phase correction is applied.", UserWarning, stacklevel=3,
        )
    return bool(aligned)


def _select_direction(rao: xr.DataArray, direction: float) -> xr.DataArray:
    headings = np.asarray(rao.dir.values, dtype=float)
    if headings.size == 0 or not np.isfinite(headings).all():
        raise ValueError("RAO directions must be finite and nonempty")
    order = np.argsort(headings)
    headings = headings[order]
    if np.any(np.diff(headings) <= 0):
        raise ValueError("RAO directions must be distinct")
    exact = np.flatnonzero(np.isclose(headings, direction, rtol=0, atol=1e-8))
    if exact.size:
        return rao.isel(dir=int(order[exact[0]]), drop=True)
    if direction < headings[0] or direction > headings[-1]:
        raise ValueError("requested direction is outside the available RAO heading range; "
                         "no directional extrapolation or symmetry is assumed")
    upper = int(np.searchsorted(headings, direction))
    fraction = (direction - headings[upper-1]) / (headings[upper] - headings[upper-1])
    lower_rao = rao.isel(dir=int(order[upper-1]), drop=True)
    upper_rao = rao.isel(dir=int(order[upper]), drop=True)
    return lower_rao + fraction * (upper_rao - lower_rao)


def _response_units(ds: xr.Dataset, responses: list[str]) -> list[str]:
    if "response_unit" in ds.coords and ds.response_unit.dims == ("resp",):
        return [str(value) for value in ds.response_unit.sel(resp=responses).values]
    mapping = ds.attrs.get("response_units", {})
    if isinstance(mapping, dict) and mapping:
        return [str(mapping.get(name, "")) for name in responses]
    # HydroStar's global unit only describes one channel reliably. The reader
    # converts rotational deg/m RAOs to rad/m; account for that conversion here.
    if ds.sizes["resp"] == 1:
        unit = str(ds.attrs.get("hydrostar_unit", "")).strip()
        if unit.endswith("/m"):
            unit = unit[:-2]
            if unit == "deg" and ds.attrs.get("hydrostar_raotype", "").upper() == "MOTION":
                unit = "rad"
            return [unit]
    return [""] * len(responses)


def reconstruct_response(
    rs: RAOSet,
    components: xr.Dataset,
    time: np.ndarray,
    *,
    resp: str | list[str] | None = None,
    direction: float = 180.0,
    frequency_domain: str = "encounter",
    interpolation: str = "linear",
    extrapolation: str = "error",
    phase_convention: str = "negative_time",
    rao_phase_convention: str | None = None,
    phase_reference_confirmed: bool = False,
) -> xr.DataArray:
    """Reconstruct first-order responses for CFD comparison from complex RAOs.

    For ``negative_time`` return Re(sum(a*H*exp(i*(phase-omega_t*time))));
    ``positive_time`` changes the time sign to plus. ``omega_t`` is intrinsic
    frequency or signed encounter frequency omega-k*U*cos(direction), with
    0 degrees following and 180 head seas. RAOs are always interpolated at
    intrinsic component frequencies, independently in real and imaginary
    parts. Direction interpolation is linear inside the tabulated range, with
    no assumed symmetry or extrapolation across the heading-grid boundary.
    Frequency extrapolation either raises (``error``) or uses zero RAOs.

    ``rao_phase_convention`` declares how the input H was defined. If different
    from ``phase_convention``, H is conjugated locally; component phases are
    unchanged. The same declaration can be stored in rs.dataset.attrs under
    ``rao_phase_convention``. If unspecified, warn and use H as supplied.
    HydroStar's convention is not inferred automatically from its source name.
    Choosing the time sign alone does not establish harmonic compatibility.

    Wave-component phases refer to their origin (0,0 unless component attrs
    x_wave_ref/y_wave_ref specify it). That origin must coincide with the RAO
    incident-wave phase reference x_wave_ref/y_wave_ref. Unknown or different
    references warn. ``phase_reference_confirmed=True`` declares that the
    caller has established alignment externally; it applies no correction.
    Time is used exactly as supplied. CFD time offsets and ramp-up cannot be
    recovered from RAOs, and no mean removal, normalization, or filtering is
    applied. Nonlinear mean offsets are not included.

    Encounter mode requires scalar RAO speed [m/s] and depth [m]. Component
    wave numbers are authoritative and checked using existing dispersion at
    the stated depth/gravity (component gravity attr, otherwise 9.81 m/s^2).
    Intrinsic mode can omit speed; depth must be known in either dataset.
    Positive infinity means deep water. Signed/nonmonotonic encounter
    frequencies can be summed directly here without spectrum transformation.

    Return dimensions are time, resp, and any additional sectional dimensions.
    Reliable response units are taken from a response_unit(resp) coordinate,
    a response_units mapping attr, or a single-channel HydroStar unit. Mixed
    units are stored in a response_unit coordinate, not a global units attr.
    Theoretical variance is sum(a**2*abs(H)**2)/2 for distinct nonzero temporal
    frequencies; a finite record may differ through beating and interference.
    Continuous-spectrum variance may also differ through discretization.

    Example::

        response = reconstruct_response(
            rs, components, np.arange(0, 660, 0.1), resp=["heave", "pitch"],
            direction=180, rao_phase_convention="negative_time",
            phase_reference_confirmed=True,
        )

    Both phase arguments in this example require prior verification against
    the actual wave generator and RAO harmonic/reference conventions.
    """
    if not isinstance(rs, RAOSet):
        raise ValueError("rs must be a valid RAOSet")
    if not isinstance(components, xr.Dataset):
        raise ValueError("components must be an xarray Dataset")
    if frequency_domain not in {"intrinsic", "encounter"}:
        raise ValueError("frequency_domain must be intrinsic or encounter")
    if interpolation != "linear":
        raise ValueError("only linear interpolation is supported")
    if extrapolation not in {"error", "zero"}:
        raise ValueError("extrapolation must be error or zero")
    if phase_convention not in _PHASE_CONVENTIONS:
        raise ValueError("phase_convention must be negative_time or positive_time")
    source_convention = rao_phase_convention or rs.dataset.attrs.get("rao_phase_convention")
    if source_convention is not None and source_convention not in _PHASE_CONVENTIONS:
        raise ValueError("rao_phase_convention must be negative_time or positive_time")
    times = np.asarray(time, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("time must be one-dimensional and finite")
    if not np.isfinite(direction):
        raise ValueError("direction must be finite")
    direction = float(direction) % 360
    arrays = {}
    for name in ("amplitude", "freq", "phase", "wave_number"):
        if name not in components or components[name].dims != ("component",):
            raise ValueError(f"components require '{name}' with dimension 'component'")
        values = np.asarray(components[name].values, dtype=float)
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f"component {name} must be finite and nonempty")
        arrays[name] = values
    amplitude, omega, phase, k = (arrays[key] for key in ("amplitude", "freq", "phase", "wave_number"))
    if np.any(amplitude < 0):
        raise ValueError("component amplitudes must be nonnegative")
    if np.any(omega <= 0) or np.any(np.diff(omega) <= 0):
        raise ValueError("component frequencies must be positive and strictly increasing")
    if np.any(k <= 0):
        raise ValueError("component wave numbers must be positive")

    ds = rs.dataset
    speed = _scalar_coordinate(ds, "speed", required=frequency_domain == "encounter")
    if speed is not None and (not np.isfinite(speed) or speed < 0):
        raise ValueError("speed must be finite and nonnegative")
    rao_depth = _scalar_coordinate(ds, "depth", required=frequency_domain == "encounter")
    component_depth = components.attrs.get("depth")
    if "depth" in components.coords:
        if components.coords["depth"].ndim != 0:
            raise ValueError("component depth must be scalar")
        coord_depth = _validate_depth(components.coords["depth"].item())
        if component_depth is not None and not np.isclose(coord_depth, _validate_depth(component_depth)):
            raise ValueError("component depth coordinate and attribute differ")
        component_depth = coord_depth
    if rao_depth is not None:
        rao_depth = _validate_depth(rao_depth)
    if component_depth is not None:
        component_depth = _validate_depth(component_depth)
    if rao_depth is not None and component_depth is not None and not np.isclose(
        rao_depth, component_depth, rtol=1e-6, atol=1e-8
    ):
        raise ValueError("wave-component depth differs from RAO depth")
    depth = component_depth if component_depth is not None else rao_depth
    if depth is None:
        raise ValueError("depth metadata is required in the components or RAO dataset")
    gravity = float(components.attrs.get("gravity", 9.81))
    if not np.isfinite(gravity) or gravity <= 0:
        raise ValueError("component gravity must be positive and finite")
    expected_k = wave_number(omega, depth=depth, g=gravity).values
    if not np.allclose(k, expected_k, rtol=1e-6, atol=0):
        raise ValueError("stored wave numbers are inconsistent with component frequencies and depth")
    frequencies = np.asarray(ds.freq.values, dtype=float)
    if (not frequencies.size or not np.isfinite(frequencies).all() or
            np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0)):
        raise ValueError("RAO frequencies must be finite, positive and strictly increasing")
    outside = (omega < frequencies[0]) | (omega > frequencies[-1])
    if extrapolation == "error" and outside.any():
        raise ValueError("component frequencies lie outside the RAO frequency range; "
                         "use extrapolation='zero' to omit their response contributions")
    responses = list(ds.resp.values) if resp is None else ([resp] if isinstance(resp, str) else list(resp))
    if not responses or any(name not in ds.resp.values for name in responses):
        raise ValueError("requested response channels must exist and be nonempty")
    rao = _select_direction(ds.rao.sel(resp=responses), direction)
    if not np.isfinite(rao.values).all():
        raise ValueError("selected RAOs must contain finite complex values")
    extra_dims = [dim for dim in rao.dims if dim not in {"resp", "freq"}]
    rao = rao.transpose("resp", *extra_dims, "freq")
    def interp(values):
        return np.interp(omega, frequencies, values, left=0.0, right=0.0)

    def interp_part(part):
        return xr.apply_ufunc(interp, part, input_core_dims=[["freq"]],
                             output_core_dims=[["component"]], vectorize=True,
                             output_dtypes=[float])

    h = interp_part(rao.real).values + 1j * interp_part(rao.imag).values
    if source_convention is None:
        warnings.warn("RAO harmonic convention is unspecified; using complex RAOs as supplied. "
                      "Declare rao_phase_convention to establish compatibility with the time convention.",
                      UserWarning, stacklevel=2)
    elif source_convention != phase_convention:
        h = h.conj()
    aligned = _check_phase_reference(ds, components, phase_reference_confirmed)
    temporal_freq = omega if frequency_domain == "intrinsic" else omega - k * speed * np.cos(np.deg2rad(direction))
    coefficients = (h * amplitude * np.exp(1j * phase)).reshape(-1, omega.size)
    result = np.empty((times.size, coefficients.shape[0]), dtype=float)
    block_size = max(1, _MAX_EXPONENTIAL_ELEMENTS // omega.size)
    sign = -1 if phase_convention == "negative_time" else 1
    for start in range(0, times.size, block_size):
        stop = min(start + block_size, times.size)
        exponential = np.exp(sign * 1j * times[start:stop, None] * temporal_freq)
        result[start:stop] = (exponential @ coefficients.T).real
    dims = ("time", "resp", *extra_dims)
    coords = {key: value for key, value in rao.coords.items() if "freq" not in value.dims}
    coords["time"] = times
    result = result.reshape(times.size, *(rao.sizes[dim] for dim in dims[1:]))
    output = xr.DataArray(result, dims=dims, coords=coords, name="response")
    output.coords["time"].attrs["units"] = "s"
    output.attrs.update(direction=direction, frequency_domain=frequency_domain, depth=depth,
                        phase_convention=phase_convention, interpolation=interpolation,
                        extrapolation=extrapolation, n_components=omega.size,
                        rao_phase_convention=source_convention or "unspecified",
                        phase_reference_confirmed=aligned)
    if speed is not None:
        output.attrs["speed"] = speed
    units = _response_units(ds, responses)
    if any(units):
        output = output.assign_coords(response_unit=("resp", units))
        if len(set(units)) == 1:
            output.attrs["units"] = units[0]
    return output
