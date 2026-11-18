"""Physical and numerical checks for discrete CFD wave components."""

import csv
import json

import numpy as np
import pytest
import xarray as xr

from rasta import generate_wave_components
from rasta.waves import (
    read_wave_components_csv,
    wave_elevation,
    write_wave_components_csv,
)


@pytest.mark.parametrize("method", ["equal_energy", "equal_frequency"])
@pytest.mark.parametrize("model", ["bretschneider", "pm", "pierson_moskowitz", "jonswap", "goda"])
def test_generation_and_variance(method, model):
    ds = generate_wave_components(hs=2, tp=8, spectrum_model=model,
                                  discretization=method, seed=42)
    assert dict(ds.sizes) == {"component": 50}
    for var in ds.data_vars.values():
        assert var.dtype == np.float64
        assert np.isfinite(var.values).all()
        assert "units" in var.attrs
    assert (ds.freq.values > 0).all()
    assert (np.diff(ds.freq.values) > 0).all()
    assert (ds.amplitude.values > 0).all()
    assert ((ds.phase.values >= 0) & (ds.phase.values < 2 * np.pi)).all()
    np.testing.assert_allclose(ds.height, 2 * ds.amplitude)
    np.testing.assert_allclose(ds.energy, ds.amplitude**2 / 2)
    assert 4 * np.sqrt(float((ds.amplitude**2 / 2).sum())) == pytest.approx(2, rel=1e-14)
    assert ds.attrs["captured_energy_fraction"] >= 0.99
    if method == "equal_energy":
        np.testing.assert_allclose(ds.energy, ds.energy.values[0], rtol=1e-12)
        assert np.ptp(np.diff(ds.freq)) > 0.01
    else:
        np.testing.assert_allclose(np.diff(ds.freq), np.diff(ds.freq)[0], rtol=1e-12)
        assert np.ptp(ds.amplitude.values) > 0.01


def test_equal_energy_uses_energy_weighted_frequencies():
    ds = generate_wave_components(hs=2, tp=8, n_components=5, spectrum_model="pm")
    # Independent high-resolution integration of the unnormalized PM shape.
    wp = 2 * np.pi / 8
    for i in range(5):
        w = np.linspace(ds.omega_lower.values[i], ds.omega_upper.values[i], 50001)
        s = w**-5 * np.exp(-1.25 * (wp / w)**4)
        expected = np.trapezoid(w * s, w) / np.trapezoid(s, w)
        assert ds.freq.values[i] == pytest.approx(expected, rel=2e-5)


def test_explicit_truncation_preserves_bin_integrals_without_normalization():
    wp = 2 * np.pi / 8
    with pytest.warns(UserWarning, match="captures only"):
        ds = generate_wave_components(hs=2, tp=8, spectrum_model="pm",
                                      discretization="equal_frequency", n_components=7,
                                      omega_min=0.8 * wp, omega_max=1.5 * wp,
                                      normalize_energy=False)
    # Analytic integral of the PM shape: variance CDF = exp(-1.25*(wp/w)^4).
    expected = 0.25 * (np.exp(-1.25 * (wp / ds.omega_upper.values)**4) -
                       np.exp(-1.25 * (wp / ds.omega_lower.values)**4))
    np.testing.assert_allclose(ds.energy.values, expected, rtol=2e-6)
    assert ds.attrs["normalization_factor"] == 1
    assert ds.attrs["captured_energy_fraction"] == pytest.approx(expected.sum() / 0.25, rel=2e-6)
    assert ds.attrs["hs_discrete"] < 2
    assert ds.attrs["omega_min"] == 0.8 * wp
    assert ds.attrs["omega_max"] == 1.5 * wp
    with pytest.warns(UserWarning):
        normalized = generate_wave_components(hs=2, tp=8, spectrum_model="pm",
                                              omega_min=0.8 * wp, omega_max=1.5 * wp)
    assert normalized.attrs["hs_discrete"] == pytest.approx(2)
    assert normalized.attrs["normalization_factor"] > 1


def test_random_phases_local_and_reproducible():
    state = np.random.get_state()
    a = generate_wave_components(hs=2, tp=8, seed=42)
    b = generate_wave_components(hs=2, tp=8, seed=42)
    c = generate_wave_components(hs=2, tp=8, seed=43)
    xr.testing.assert_identical(a, b)
    assert not np.array_equal(a.phase, c.phase)
    xr.testing.assert_equal(a.freq, c.freq)
    xr.testing.assert_equal(a.amplitude, c.amplitude)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


@pytest.mark.parametrize("depth", [np.inf, 10.0, 0.01])
def test_dispersion_and_derived_quantities(depth):
    ds = generate_wave_components(hs=2, tp=8, depth=depth, g=9.8)
    k = ds.wave_number.values
    expected = 9.8 * k if np.isinf(depth) else 9.8 * k * np.tanh(k * depth)
    np.testing.assert_allclose(ds.freq.values**2, expected, rtol=1e-10)
    np.testing.assert_allclose(ds.period, 2 * np.pi / ds.freq)
    np.testing.assert_allclose(ds.wavelength, 2 * np.pi / ds.wave_number)
    assert ds.attrs["T_beat"] == pytest.approx(2 * np.pi / np.min(np.diff(ds.freq)))


def test_reconstruction_and_csv_roundtrip(tmp_path):
    ds = generate_wave_components(hs=2, tp=8, seed=42)
    original = ds.copy(deep=True)
    eta = wave_elevation(ds, np.array([0, 1, 2]))
    assert eta.dims == ("time",)
    assert eta.values[0] == pytest.approx(float((ds.amplitude * np.cos(ds.phase)).sum()))
    x = 2.5
    for convention, sign in [("negative_time", -1), ("positive_time", 1)]:
        eta = wave_elevation(ds, np.array([1.5]), x=x, phase_sign=convention)
        expected = (ds.amplitude * np.cos(ds.wave_number*x + sign*ds.freq*1.5 + ds.phase)).sum()
        assert eta.item() == pytest.approx(expected.item())
    xr.testing.assert_identical(ds, original)
    path = tmp_path / "waves.csv"
    write_wave_components_csv(ds, path)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 50
    restored = read_wave_components_csv(path)
    for name in restored.data_vars:
        xr.testing.assert_identical(restored[name], ds[name])


def test_single_component_kwargs_and_metadata():
    ds = generate_wave_components(hs=2, tp=8, n_components=1,
                                  spectrum_kwargs={"gamma": 5}, seed=None)
    assert ds.sizes["component"] == 1
    assert np.isnan(ds.attrs["minimum_frequency_spacing"])
    assert np.isnan(ds.attrs["T_beat"])
    assert json.loads(ds.attrs["spectrum_kwargs"]) == {"gamma": 5}
    assert ds.attrs["hs_discrete"] == pytest.approx(2)


@pytest.mark.parametrize("kwargs", [
    {"hs": 0}, {"hs": -1}, {"hs": np.nan}, {"tp": 0}, {"tp": np.inf},
    {"n_components": 0}, {"n_components": 1.5}, {"n_components": True},
    {"omega_min": 1, "omega_max": 1}, {"omega_min": 0}, {"omega_max": np.inf},
    {"discretization": "other"}, {"spectrum_model": "other"},
    {"depth": 0}, {"depth": -np.inf}, {"depth": np.nan}, {"g": 0},
])
def test_invalid_inputs(kwargs):
    options = {"hs": 2, "tp": 8, **kwargs}
    with pytest.raises(ValueError):
        generate_wave_components(**options)
