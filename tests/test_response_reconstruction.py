"""Analytical checks of complex linear response reconstruction."""

import numpy as np
import pytest
import xarray as xr

from rasta import RAOSet, reconstruct_response, wave_number
from rasta.validation import REQUIRED_ATTRS
from rasta.waves import generate_wave_components, wave_elevation


def _components(depth=np.inf):
    w = np.array([0.5, 1.0, 1.5])
    return xr.Dataset(
        {"amplitude": ("component", [0.2, 0.3, 0.4]),
         "freq": ("component", w), "phase": ("component", [0.3, 1.2, 2.3]),
         "wave_number": ("component", wave_number(w, depth=depth).values)},
        coords={"component": [0, 1, 2]}, attrs={"depth": depth, "gravity": 9.81},
    )


def _rs(values=(1+0j,), *, speed=0.0, depth=np.inf, directions=(0.0, 180.0),
        freq=(0.2, 2.0, 4.0)):
    names = ["heave", "pitch", "heave_acc", "My"][:len(values)]
    data = np.broadcast_to(np.asarray(values, dtype=complex)[:, None, None],
                           (len(values), len(directions), len(freq))).copy()
    return RAOSet(xr.Dataset(
        {"rao": (("resp", "dir", "freq"), data)},
        coords={"resp": names, "dir": np.array(directions, dtype=float),
                "freq": np.array(freq, dtype=float), "speed": speed, "depth": depth},
        attrs={**REQUIRED_ATTRS, "rao_phase_convention": "negative_time",
               "x_wave_ref": 0.0, "y_wave_ref": 0.0},
    ))


def _reference(c, times, h, *, speed=0, direction=180, sign=-1):
    omega_t = c.freq.values - c.wave_number.values*speed*np.cos(np.deg2rad(direction))
    return np.real(np.sum(c.amplitude.values[:, None]*np.asarray(h)[..., None]*np.exp(
        1j*(c.phase.values[:, None] + sign*omega_t[:, None]*times)), axis=0))


@pytest.mark.parametrize("factor", [1, 2])
@pytest.mark.parametrize("convention", ["negative_time", "positive_time"])
def test_real_constant_rao_matches_elevation(factor, convention):
    c = _components()
    t = np.array([0, 0.17, 2.2, 50.1])
    out = reconstruct_response(_rs([factor]), c, t, phase_convention=convention)
    np.testing.assert_allclose(out[:, 0], factor*wave_elevation(c, t, phase_sign=convention), atol=1e-14)
    assert out.dims == ("time", "resp")
    np.testing.assert_array_equal(out.time, t)


def test_complex_phase_and_multiple_channels():
    c = _components()
    t = np.array([0, 1.7, 4.3, 100])
    values = [1+2j, -0.2+0.3j, -3j, 100-50j]
    rs = _rs(values)
    original_c, original_rs = c.copy(deep=True), rs.dataset.copy(deep=True)
    out = reconstruct_response(rs, c, t)
    assert list(out.resp.values) == ["heave", "pitch", "heave_acc", "My"]
    assert "units" not in out.attrs
    for index, value in enumerate(values):
        np.testing.assert_allclose(out[:, index], _reference(c, t, value), atol=1e-13)
    xr.testing.assert_identical(c, original_c)
    xr.testing.assert_identical(rs.dataset, original_rs)
    selected = reconstruct_response(rs, c, t, resp=["My", "heave"])
    xr.testing.assert_equal(selected, out.sel(resp=["My", "heave"]))
    assert reconstruct_response(rs, c, t, resp="pitch").sizes["resp"] == 1


def test_complex_frequency_and_heading_interpolation():
    c = _components()
    rs = _rs(freq=(0.2, 0.8, 2.0), directions=(0, 90, 180))
    ds = rs.dataset.copy(deep=True)
    # Both real and imaginary parts vary independently with frequency/heading.
    ds["rao"] = xr.ones_like(ds.rao) * (2*ds.freq + 3j*ds.freq + ds.dir*(0.01-0.02j))
    t = np.array([0, 1, 3, 7])
    out = reconstruct_response(RAOSet(ds), c, t, direction=135)
    h = 2*c.freq.values + 3j*c.freq.values + 135*(0.01-0.02j)
    np.testing.assert_allclose(out[:, 0], _reference(c, t, h, direction=135), atol=1e-14)


@pytest.mark.parametrize("depth", [np.inf, 7.0])
@pytest.mark.parametrize("direction", [0.0, 45.0, 90.0, 180.0])
def test_encounter_time_dependence(depth, direction):
    c = _components(depth)
    rs = _rs([1+0.7j], speed=5.0, depth=depth)
    t = np.array([0, 1.1, 10.3, 19])
    out = reconstruct_response(rs, c, t, direction=direction)
    np.testing.assert_allclose(out[:, 0], _reference(c, t, 1+0.7j, speed=5, direction=direction), atol=1e-14)
    assert out.attrs["depth"] == depth
    assert out.attrs["speed"] == 5
    assert out.attrs["direction"] == direction
    assert out.attrs["n_components"] == 3
    assert out.attrs["frequency_domain"] == "encounter"
    # Intrinsic mode changes only the temporal frequency.
    intrinsic = reconstruct_response(rs, c, t, direction=direction, frequency_domain="intrinsic")
    np.testing.assert_allclose(intrinsic[:, 0], _reference(c, t, 1+0.7j), atol=1e-14)


def test_zero_speed_and_phase_conversion():
    rs = _rs([1+2j])
    c = _components()
    t = np.array([0, 1.7, 4.8])
    intrinsic = reconstruct_response(rs, c, t, frequency_domain="intrinsic")
    encounter = reconstruct_response(rs, c, t)
    np.testing.assert_array_equal(intrinsic, encounter)
    for convention, sign in [("positive_time", 1), ("negative_time", -1)]:
        out = reconstruct_response(rs, c, t, phase_convention=convention,
                                   rao_phase_convention="positive_time")
        h = 1+2j if convention == "positive_time" else 1-2j
        np.testing.assert_allclose(out[:, 0], _reference(c, t, h, sign=sign))
    # Explicit conjugation of a copy matches the automatic conversion exactly.
    ds = rs.dataset.copy(deep=True)
    ds["rao"] = ds.rao.conj()
    explicit = reconstruct_response(RAOSet(ds), c, t, phase_convention="positive_time",
                                     rao_phase_convention="positive_time")
    automatic = reconstruct_response(rs, c, t, phase_convention="positive_time")
    np.testing.assert_array_equal(explicit, automatic)


def test_frequency_extrapolation():
    c = _components()
    rs = _rs(freq=(0.8, 1.0, 1.2))
    t = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="outside the RAO frequency range"):
        reconstruct_response(rs, c, t)
    out = reconstruct_response(rs, c, t, extrapolation="zero")
    expected = 0.3*np.cos(1.2-t)
    np.testing.assert_allclose(out[:, 0], expected, atol=1e-14)


def test_generated_components_reproducible_and_seeds_change_signal():
    rs = _rs()
    t = np.linspace(0, 600, 1000)
    a = generate_wave_components(hs=1, tp=8, n_components=10, seed=42)
    b = generate_wave_components(hs=1, tp=8, n_components=10, seed=42)
    c = generate_wave_components(hs=1, tp=8, n_components=10, seed=43)
    xr.testing.assert_identical(a.freq, c.freq)
    xr.testing.assert_identical(a.amplitude, c.amplitude)
    assert not np.array_equal(a.phase, c.phase)
    xr.testing.assert_identical(reconstruct_response(rs, a, t), reconstruct_response(rs, b, t))
    assert not np.array_equal(reconstruct_response(rs, a, t), reconstruct_response(rs, c, t))


def test_depth_and_wave_number_consistency():
    c = _components(10)
    with pytest.raises(ValueError, match="depth differs"):
        reconstruct_response(_rs(), c, [0, 1])
    rs = _rs(depth=10)
    c.wave_number.values[0] *= 1.1
    with pytest.raises(ValueError, match="wave numbers are inconsistent"):
        reconstruct_response(rs, c, [0, 1])


def test_reference_and_harmonic_warnings():
    rs = _rs()
    ds = rs.dataset.copy(deep=True)
    ds.attrs.pop("rao_phase_convention")
    with pytest.warns(UserWarning, match="harmonic convention is unspecified"):
        reconstruct_response(RAOSet(ds), _components(), [0, 1])
    ds.attrs.update(rao_phase_convention="negative_time", x_wave_ref=50)
    with pytest.warns(UserWarning, match="phase references"):
        unconfirmed = reconstruct_response(RAOSet(ds), _components(), [0, 1])
    confirmed = reconstruct_response(RAOSet(ds), _components(), [0, 1], phase_reference_confirmed=True)
    np.testing.assert_array_equal(unconfirmed, confirmed)
    assert not unconfirmed.attrs["phase_reference_confirmed"]
    assert confirmed.attrs["phase_reference_confirmed"]
    c = _components()
    c.attrs.update(x_wave_ref=50, y_wave_ref=0)
    established = reconstruct_response(RAOSet(ds), c, [0, 1])
    assert established.attrs["phase_reference_confirmed"]
    ds.attrs.pop("x_wave_ref")
    with pytest.warns(UserWarning, match="phase references"):
        reconstruct_response(RAOSet(ds), c, [0, 1])


def test_sectional_dimensions_and_response_units():
    rs = _rs([1+2j, 2+0j, -1j, 20+0j])
    ds = rs.dataset.expand_dims(x=[10.0, 20.0]).copy(deep=True)
    ds["rao"] = ds.rao * (ds.x/10)
    ds = ds.assign_coords(response_unit=("resp", ["m", "rad", "m/s^2", "N m"]),
                          body_id=("x", [1, 2]))
    out = reconstruct_response(RAOSet(ds), _components(), [0, 1, 2])
    assert out.dims == ("time", "resp", "x")
    np.testing.assert_allclose(out.sel(x=20), out.sel(x=10)*2)
    np.testing.assert_array_equal(out.body_id, [1, 2])
    assert "units" not in out.attrs
    assert out.response_unit.sel(resp="heave_acc").item() == "m/s^2"
    heave = reconstruct_response(RAOSet(ds), _components(), [0], resp="heave")
    assert heave.attrs["units"] == "m"
    assert heave.time.attrs["units"] == "s"


def test_single_hydrostar_rotation_units():
    rs = _rs()
    ds = rs.dataset.assign_coords(resp=["pitch"])
    ds.attrs.update(hydrostar_unit="deg/m", hydrostar_raotype="MOTION")
    out = reconstruct_response(RAOSet(ds), _components(), [0, 1])
    assert out.attrs["units"] == "rad"


def test_long_record_variance_and_chunking(monkeypatch):
    import rasta.waves.reconstruction as module
    monkeypatch.setattr(module, "_MAX_EXPONENTIAL_ELEMENTS", 50000)
    c = generate_wave_components(hs=1, tp=8, n_components=5, seed=42)
    rs = _rs([1+2j])
    t = np.arange(0, 100000, 0.5)
    out = reconstruct_response(rs, c, t)
    theoretical = float((c.amplitude**2 * 5 / 2).sum())
    assert float(out.var()) == pytest.approx(theoretical, rel=0.005)


@pytest.mark.parametrize("kwargs,match", [
    ({"interpolation": "cubic"}, "linear"),
    ({"extrapolation": "linear"}, "extrapolation"),
    ({"frequency_domain": "other"}, "frequency_domain"),
    ({"phase_convention": "other"}, "phase_convention"),
    ({"rao_phase_convention": "other"}, "rao_phase_convention"),
    ({"direction": np.nan}, "direction"),
    ({"direction": 270}, "heading range"),
    ({"resp": "missing"}, "response channels"),
    ({"resp": []}, "response channels"),
])
def test_invalid_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        reconstruct_response(_rs(), _components(), [0, 1], **kwargs)


@pytest.mark.parametrize("name,value,match", [
    ("amplitude", -1, "nonnegative"), ("amplitude", np.nan, "finite"),
    ("freq", 0, "positive"), ("freq", 1.0, "increasing"),
    ("phase", np.inf, "finite"), ("wave_number", 0, "positive"),
])
def test_invalid_components(name, value, match):
    c = _components()
    c[name].values[0] = value
    with pytest.raises(ValueError, match=match):
        reconstruct_response(_rs(), c, [0, 1])


@pytest.mark.parametrize("name", ["amplitude", "freq", "phase", "wave_number"])
def test_missing_component_variables(name):
    with pytest.raises(ValueError, match=name):
        reconstruct_response(_rs(), _components().drop_vars(name), [0, 1])


@pytest.mark.parametrize("name", ["speed", "depth"])
def test_missing_or_unselected_metadata(name):
    ds = _rs().dataset.drop_vars(name)
    with pytest.raises(ValueError, match=name):
        reconstruct_response(RAOSet(ds), _components(), [0, 1])
    with pytest.raises(ValueError, match="single scalar"):
        reconstruct_response(RAOSet(ds.expand_dims({name: [1, 2]})), _components(), [0, 1])


@pytest.mark.parametrize("time", [[[0, 1]], [0, np.nan], [np.inf]])
def test_invalid_time(time):
    with pytest.raises(ValueError, match="time"):
        reconstruct_response(_rs(), _components(), time)


def test_intrinsic_can_omit_speed_and_zero_amplitude():
    ds = _rs().dataset.drop_vars("speed")
    c = _components()
    c.amplitude.values[:] = 0
    out = reconstruct_response(RAOSet(ds), c, [0, 1], frequency_domain="intrinsic")
    np.testing.assert_array_equal(out.values, 0)
    assert "speed" not in out.attrs
