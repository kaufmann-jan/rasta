import numpy as np
import pytest
import xarray as xr

from rasta.rao import RAOSet
from rasta.stats.response_spectrum import compute_response_spectrum
from rasta.stats.shortterm import _encounter_frequency, response_variance, shortterm_statistics
from rasta.validation import REQUIRED_ATTRS


def test_response_variance_shape():
    freq = np.array([1.0, 2.0, 3.0], dtype=float)
    direction = np.array([0.0, 180.0], dtype=float)
    resp = np.array(["heave"], dtype=object)

    rao = xr.DataArray(
        np.ones((3, 2, 1), dtype=np.complex128),
        dims=("freq", "dir", "resp"),
        coords={"freq": freq, "dir": direction, "resp": resp},
    )
    ds = xr.Dataset({"rao": rao}, attrs=dict(REQUIRED_ATTRS))

    rs = RAOSet(ds)
    spec = xr.DataArray(np.ones((3, 2)), dims=("freq", "dir"), coords={"freq": freq, "dir": direction})
    var = response_variance(rs, spec)
    assert "freq" not in var.dims
    assert set(var.dims) == {"dir", "resp"}


def test_shortterm_statistics_duration_dependency():
    freq = np.linspace(0.1, 2.0, 1000)
    # simple synthetic spectrum
    s = xr.DataArray(np.exp(-((freq - 1.0) ** 2) / (2 * 0.15**2)), dims=("freq",), coords={"freq": freq})

    st1 = shortterm_statistics(s, duration_s=1800.0)
    st2 = shortterm_statistics(s, duration_s=7200.0)

    assert float(st1["m0"].values) > 0.0
    assert float(st1["m2"].values) > 0.0
    assert float(st1["Tz"].values) > 0.0
    assert float(st2["Ncycles"].values) > float(st1["Ncycles"].values)
    assert float(st2["X_mpm"].values) > float(st1["X_mpm"].values)


def _encounter_case(*, speed=3.0, depth=np.inf, directions=(180.0,), directional=False):
    freq = np.linspace(0.1, 2.0, 10001)
    rs = RAOSet(xr.Dataset(
        {"rao": (("resp", "dir", "freq"),
                 np.ones((2, len(directions), freq.size), dtype=complex))},
        coords={"resp": ["heave", "pitch"], "dir": np.asarray(directions, dtype=float),
                "freq": freq, "speed": speed, "depth": depth},
        attrs=dict(REQUIRED_ATTRS),
    ))
    s = xr.DataArray(np.stack([np.exp(-((freq-1)/0.2)**2),
                               0.3*np.exp(-((freq-0.8)/0.15)**2)]),
                     dims=("resp", "freq"), coords={"resp": ["heave", "pitch"], "freq": freq})
    if directional:
        s = s.expand_dims(dir=np.asarray(directions, dtype=float)).transpose("resp", "dir", "freq")
    return s, rs


def test_intrinsic_default_is_unchanged():
    s, _ = _encounter_case()
    implicit = shortterm_statistics(s)
    explicit = shortterm_statistics(s, frequency_domain="intrinsic")
    xr.testing.assert_identical(implicit, explicit)
    for n in range(3):
        np.testing.assert_allclose(implicit[f"m{n}"], np.trapezoid(s.values*s.freq.values**n, s.freq.values))


def test_encounter_zero_speed():
    s, rs = _encounter_case(speed=0)
    intrinsic = shortterm_statistics(s)
    encounter = shortterm_statistics(s, frequency_domain="encounter", raoset=rs)
    for name in intrinsic.data_vars:
        xr.testing.assert_equal(encounter[name], intrinsic[name])


@pytest.mark.parametrize("depth", [np.inf, 10.0])
def test_head_sea_moments_and_invariants(depth):
    s, rs = _encounter_case(depth=depth)
    original_s, original_rs = s.copy(deep=True), rs.dataset.copy(deep=True)
    intrinsic = shortterm_statistics(s, duration_s=1800)
    encounter = shortterm_statistics(s, duration_s=1800, frequency_domain="encounter", raoset=rs)
    w = s.freq.values
    if np.isinf(depth):
        k = w**2/9.81
    else:
        # Independent bracketing solution of g*k*tanh(k*h) = omega**2.
        low = np.zeros_like(w)
        high = w**2/9.81 + w/np.sqrt(9.81*depth) + 1/depth
        for _ in range(70):
            mid = (low+high)/2
            below = 9.81*mid*np.tanh(mid*depth) < w**2
            low = np.where(below, mid, low)
            high = np.where(below, high, mid)
        k = (low+high)/2
    we = w + 3*k
    for n in range(3):
        expected = np.trapezoid(s.values*we**n, w)
        np.testing.assert_allclose(encounter[f"m{n}"], expected, rtol=1e-10)
    for name in ("m0", "sigma", "rms", "X_sig"):
        xr.testing.assert_equal(encounter[name], intrinsic[name])
    assert (encounter.Tz < intrinsic.Tz).all()
    assert (encounter.Ncycles > intrinsic.Ncycles).all()
    assert (encounter.X_mpm > intrinsic.X_mpm).all()
    assert (encounter.X_mean_max > intrinsic.X_mean_max).all()
    assert encounter.m1.dims == ("resp",)
    assert "freq" not in encounter.dims
    assert encounter.attrs["frequency_domain"] == "encounter"
    assert encounter.attrs["speed"] == 3
    assert encounter.attrs["depth"] == depth
    assert encounter.attrs["duration_s"] == 1800
    assert "encounter_frequency_formula" in encounter.attrs
    xr.testing.assert_identical(s, original_s)
    xr.testing.assert_identical(rs.dataset, original_rs)


def test_directions_broadcast_and_helper():
    s, rs = _encounter_case(speed=0.5, directions=(0.0, 60.0, 90.0, 180.0), directional=True)
    we = _encounter_frequency(s.freq, s.dir, 0.5, np.inf)
    assert set(we.dims) == {"freq", "dir"}
    xr.testing.assert_equal(we.freq, s.freq)
    xr.testing.assert_equal(we.dir, s.dir)
    out = shortterm_statistics(s, frequency_domain="encounter", raoset=rs)
    assert out.m1.dims == ("resp", "dir")
    for direction in s.dir.values:
        expected_w = s.freq.values - 0.5*s.freq.values**2/9.81*np.cos(np.deg2rad(direction))
        for n in range(3):
            expected = np.trapezoid(s.sel(dir=direction).values*expected_w**n, s.freq.values)
            np.testing.assert_allclose(out[f"m{n}"].sel(dir=direction), expected)


def test_scalar_speed_and_depth_after_selection():
    s, rs = _encounter_case()
    ds = rs.dataset.drop_vars(["speed", "depth"]).expand_dims(speed=[1.0, 3.0], depth=[10.0, np.inf])
    selected = RAOSet(ds.sel(speed=3.0, depth=np.inf))
    actual = shortterm_statistics(s, frequency_domain="encounter", raoset=selected)
    expected = shortterm_statistics(s, frequency_domain="encounter", raoset=rs)
    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize("name", ["speed", "depth"])
def test_missing_and_nonscalar_metadata(name):
    s, rs = _encounter_case()
    with pytest.raises(ValueError, match=name):
        shortterm_statistics(s, frequency_domain="encounter", raoset=RAOSet(rs.dataset.drop_vars(name)))
    nonscalar = RAOSet(rs.dataset.drop_vars(name).expand_dims({name: [1.0]}))
    with pytest.raises(ValueError, match=f"single scalar '{name}'"):
        shortterm_statistics(s, frequency_domain="encounter", raoset=nonscalar)


@pytest.mark.parametrize("name,value", [
    ("speed", -1), ("speed", np.nan), ("speed", np.inf),
    ("depth", 0), ("depth", -1), ("depth", -np.inf), ("depth", np.nan),
])
def test_invalid_speed_depth(name, value):
    s, rs = _encounter_case()
    bad = RAOSet(rs.dataset.assign_coords({name: value}))
    with pytest.raises(ValueError, match=name):
        shortterm_statistics(s, frequency_domain="encounter", raoset=bad)


def test_missing_raoset_and_invalid_domain():
    s, _ = _encounter_case()
    with pytest.raises(ValueError, match="requires a raoset"):
        shortterm_statistics(s, frequency_domain="encounter")
    with pytest.raises(ValueError, match="frequency_domain"):
        shortterm_statistics(s, frequency_domain="other")


def test_ambiguous_and_inconsistent_directions():
    s, rs = _encounter_case(directions=(90.0, 180.0))
    with pytest.raises(ValueError, match="ambiguous direction"):
        shortterm_statistics(s, frequency_domain="encounter", raoset=rs)
    s = s.expand_dims(dir=[270.0])
    with pytest.raises(ValueError, match="incompatible direction"):
        shortterm_statistics(s, frequency_domain="encounter", raoset=rs)
    with pytest.raises(ValueError, match="direction values must be finite"):
        shortterm_statistics(s.assign_coords(dir=[np.nan]), frequency_domain="encounter", raoset=rs)


@pytest.mark.parametrize("coord", ["speed", "depth", "dir"])
def test_inconsistent_scalar_spectrum_coordinates(coord):
    s, rs = _encounter_case()
    bad = s.assign_coords({coord: {"speed": 4.0, "depth": 10.0, "dir": 0.0}[coord]})
    with pytest.raises(ValueError, match="incompatible"):
        shortterm_statistics(bad, frequency_domain="encounter", raoset=rs)


def test_invalid_frequency_coordinates():
    s, rs = _encounter_case()
    for value in (0, -1, np.nan, np.inf, s.freq.values[1]):
        grid = s.freq.values.copy()
        grid[0] = value
        with pytest.raises(ValueError, match="frequencies must be"):
            shortterm_statistics(s.assign_coords(freq=grid), frequency_domain="encounter", raoset=rs)


def test_encounter_after_response_spectrum_with_default_frequency_grid():
    _, rs = _encounter_case()
    # A coarse input grid differs from the default response grid, which also
    # extends beyond the highest tabulated RAO frequency.
    rs = RAOSet(rs.dataset.isel(freq=slice(None, None, 500)))
    spectrum = compute_response_spectrum(
        rs, resp="heave", hs=2, tp=8, mean_dir=180,
    )["S_r"]
    assert not np.array_equal(spectrum.freq.values, rs.dataset.freq.values)
    assert spectrum.freq.max() > rs.dataset.freq.max()
    out = shortterm_statistics(spectrum, frequency_domain="encounter", raoset=rs)
    w = spectrum.freq.values
    we = w + w**2 * 3 / 9.81
    for n in range(3):
        expected = np.trapezoid(spectrum.values * we**n, w)
        np.testing.assert_allclose(out[f"m{n}"], expected)
    assert all(np.isfinite(var.values).all() for var in out.data_vars.values())


@pytest.mark.parametrize("speed", [3.0, 6.0])
def test_unsupported_following_sea_mapping(speed):
    # U=3: positive but nonmonotonic; U=6: negative at high frequencies.
    s, rs = _encounter_case(speed=speed, directions=(0.0,))
    with pytest.raises(ValueError, match="signed encounter frequencies.*multiple spectral branches"):
        shortterm_statistics(s, frequency_domain="encounter", raoset=rs)


def test_explicit_encounter_spectrum_conservation():
    s, rs = _encounter_case()
    out = shortterm_statistics(s, frequency_domain="encounter", raoset=rs)
    w = s.freq.values
    we = w + w**2 * 3/9.81
    jacobian = 1 + 2*w*3/9.81
    transformed = s.values / jacobian
    for n in range(3):
        expected = np.trapezoid(transformed*we**n, we)
        np.testing.assert_allclose(out[f"m{n}"], expected, rtol=1e-7)
