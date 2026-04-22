from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from rasta.preprocess import incident_wave_elevation, relative_vertical_motion, wave_number
from rasta.rao import RAOSet
from rasta.validation import REQUIRED_ATTRS


def _rs() -> RAOSet:
    freq = np.array([1.0], dtype=float)
    direction = np.array([0.0, 90.0, 180.0, 270.0], dtype=float)
    resp = np.array(["heave", "roll", "pitch"], dtype=object)
    data = np.array(
        [
            [[1.0 + 0.0j], [1.0 + 0.0j], [1.0 + 0.0j], [1.0 + 0.0j]],
            [[0.1 + 0.0j], [0.1 + 0.0j], [0.1 + 0.0j], [0.1 + 0.0j]],
            [[0.2 + 0.0j], [0.2 + 0.0j], [0.2 + 0.0j], [0.2 + 0.0j]],
        ],
        dtype=np.complex128,
    )
    attrs = dict(REQUIRED_ATTRS)
    attrs.update(
        {
            "xref": 50.0,
            "yref": 0.0,
            "zref": 0.0,
            "x_wave_ref": 50.0,
            "y_wave_ref": 0.0,
            "z_wave_ref": 0.0,
        }
    )
    ds = xr.Dataset(
        {"rao": (("resp", "dir", "freq"), data)},
        coords={"resp": resp, "dir": direction, "freq": freq, "depth": 20.0},
        attrs=attrs,
    )
    return RAOSet(ds)


def test_incident_wave_reference_point_is_unity() -> None:
    rs = _rs()
    out = incident_wave_elevation(rs, point=(50.0, 0.0, 0.0), point_mode="absolute", deep_water=True)
    eta = out.rao.sel(resp="point_eta").values
    assert np.allclose(eta, 1.0 + 0.0j)


def test_relative_vertical_motion_zero_offset_equals_heave_minus_eta() -> None:
    rs = _rs()
    out = relative_vertical_motion(rs, point=(0.0, 0.0, 0.0), point_mode="relative", deep_water=True)
    assert np.allclose(out.rao.sel(resp="point_z").values, out.rao.sel(resp="heave").values)
    assert np.allclose(
        out.rao.sel(resp="point_z_rel").values,
        out.rao.sel(resp="point_z").values - out.rao.sel(resp="point_eta").values,
    )


def test_relative_and_absolute_point_definitions_match() -> None:
    rs = _rs()
    rel = relative_vertical_motion(rs, point=(20.0, 3.0, 0.0), point_mode="relative", deep_water=True, point_name="deck")
    abs_ = relative_vertical_motion(rs, point=(70.0, 3.0, 0.0), point_mode="absolute", deep_water=True, point_name="deck")
    assert np.allclose(rel.rao.sel(resp="deck_z_rel").values, abs_.rao.sel(resp="deck_z_rel").values)


def test_wave_number_deep_water_finite() -> None:
    freq = xr.DataArray(np.array([0.5, 1.0, 2.0]), dims=("freq",), coords={"freq": np.array([0.5, 1.0, 2.0])})
    k = wave_number(freq, deep_water=True)
    assert np.all(np.isfinite(k.values))


def test_wave_number_infinite_depth_matches_deep_water() -> None:
    freq = xr.DataArray(np.array([0.5, 1.0, 2.0]), dims=("freq",), coords={"freq": np.array([0.5, 1.0, 2.0])})
    k_inf = wave_number(freq, depth=np.inf)
    k_deep = wave_number(freq, deep_water=True)
    assert np.allclose(k_inf.values, k_deep.values)


def test_missing_wave_reference_point_raises() -> None:
    rs = _rs()
    ds = rs.dataset.copy()
    for name in ("x_wave_ref", "y_wave_ref", "z_wave_ref"):
        ds.attrs.pop(name, None)
    with pytest.raises(ValueError, match="incident-wave reference point missing"):
        incident_wave_elevation(RAOSet(ds), point=(0.0, 0.0, 0.0), deep_water=True)


def test_preserve_dims_and_scalar_coords() -> None:
    rs = _rs()
    out = relative_vertical_motion(rs, point=(0.0, 0.0, 0.0), deep_water=False)
    assert out.rao.sel(resp="point_eta").dims == ("dir", "freq")
    assert "depth" in out.dataset.coords


def test_direction_mapping_regression() -> None:
    rs = _rs()
    k = float(wave_number(rs.dataset.coords["freq"], deep_water=True).values[0])

    head = incident_wave_elevation(rs, point=(51.0, 0.0, 0.0), point_mode="absolute", deep_water=True)
    assert np.allclose(head.rao.sel(resp="point_eta", dir=180.0).values.item(), np.exp(1j * k * 1.0))
    assert np.allclose(head.rao.sel(resp="point_eta", dir=0.0).values.item(), np.exp(-1j * k * 1.0))

    port = incident_wave_elevation(rs, point=(50.0, 1.0, 0.0), point_mode="absolute", deep_water=True)
    assert np.allclose(port.rao.sel(resp="point_eta", dir=90.0).values.item(), np.exp(-1j * k * 1.0))
    assert np.allclose(port.rao.sel(resp="point_eta", dir=270.0).values.item(), np.exp(1j * k * 1.0))


def test_head_sea_phase_progression() -> None:
    rs = _rs()
    k = float(wave_number(rs.dataset.coords["freq"], deep_water=True).values[0])
    out = incident_wave_elevation(rs, point=(52.0, 0.0, 0.0), point_mode="absolute", deep_water=True)
    assert np.allclose(out.rao.sel(resp="point_eta", dir=180.0).values.item(), np.exp(1j * k * 2.0))
