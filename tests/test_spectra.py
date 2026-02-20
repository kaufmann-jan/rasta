from __future__ import annotations

import numpy as np
import xarray as xr

from rasta.spectra import (
    bretschneider,
    directional_spectrum,
    goda,
    jonswap,
    make_directional_spectrum,
    pierson_moskowitz,
    spreading_cos2,
    spreading_cos4,
)


def _m0(spec: xr.DataArray) -> float:
    return float(np.trapezoid(spec.values, spec.coords["freq"].values))


def _circ_dist_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_spectrum_output_shapes_and_coords() -> None:
    freq = np.linspace(0.05, 5.0, 5000)
    hs = 3.0
    tp = 8.0

    for fn in (
        lambda: bretschneider(freq, hs=hs, tp=tp),
        lambda: pierson_moskowitz(freq, hs=hs, tp=tp),
        lambda: jonswap(freq, hs=hs, tp=tp),
        lambda: goda(freq, hs=hs, tp=tp),
    ):
        s = fn()
        assert isinstance(s, xr.DataArray)
        assert s.dims == ("freq",)
        assert np.issubdtype(s.coords["freq"].dtype, np.floating)

    direction = np.arange(0.0, 360.0 + 15.0, 15.0)
    d2 = spreading_cos2(direction, mean_dir=180.0)
    d4 = spreading_cos4(direction, mean_dir=180.0)
    assert d2.dims == ("dir",)
    assert d4.dims == ("dir",)
    assert np.issubdtype(d2.coords["dir"].dtype, np.floating)


def test_variance_consistency_m0() -> None:
    freq = np.linspace(0.05, 5.0, 5000)
    hs = 3.0
    tp = 8.0
    target = (hs / 4.0) ** 2

    specs = [
        bretschneider(freq, hs=hs, tp=tp),
        pierson_moskowitz(freq, hs=hs, tp=tp),
        jonswap(freq, hs=hs, tp=tp),
        goda(freq, hs=hs, tp=tp),
    ]

    for s in specs:
        m0 = _m0(s)
        assert np.isclose(m0, target, rtol=0.03, atol=1e-6)


def test_spreading_normalization() -> None:
    direction = np.arange(0.0, 360.0 + 15.0, 15.0)
    th = np.deg2rad(direction)

    for fn in (
        lambda: spreading_cos2(direction, mean_dir=180.0),
        lambda: spreading_cos4(direction, mean_dir=180.0),
    ):
        d = fn()
        integral = float(np.trapezoid(d.values, th))
        assert np.isclose(integral, 1.0, rtol=1e-3, atol=1e-3)


def test_directional_spectrum_energy_recovery() -> None:
    freq = np.linspace(0.05, 5.0, 5000)
    direction = np.arange(0.0, 360.0 + 15.0, 15.0)

    s_omega = jonswap(freq, hs=3.0, tp=8.0)
    d_dir = spreading_cos2(direction, mean_dir=200.0)

    s2d = directional_spectrum(freq, direction, S_omega=s_omega, D_dir=d_dir)
    assert s2d.dims == ("dir", "freq")

    recovered = np.trapezoid(s2d.values, np.deg2rad(direction), axis=0)
    assert np.allclose(recovered, s_omega.values, rtol=3e-2, atol=1e-8)


def test_circular_mean_direction_wrap() -> None:
    direction = np.arange(0.0, 360.0 + 10.0, 10.0)
    mean_dir = 350.0
    d = spreading_cos4(direction, mean_dir=mean_dir)

    peak_dir = float(d.coords["dir"].values[int(np.argmax(d.values))])
    assert _circ_dist_deg(peak_dir, mean_dir) <= 10.0

    d0 = float(d.sel(dir=0.0))
    d350 = float(d.sel(dir=350.0))
    assert d0 > 0.0
    assert d350 > 0.0


def test_make_directional_spectrum_wrapper() -> None:
    freq = np.linspace(0.05, 5.0, 3000)
    direction = np.arange(0.0, 360.0 + 15.0, 15.0)

    s2d = make_directional_spectrum(
        freq,
        direction,
        model="goda",
        hs=3.0,
        tp=8.0,
        mean_dir=170.0,
        spreading="cos4",
    )

    assert s2d.dims == ("dir", "freq")
    assert "dir" in s2d.coords
    assert "freq" in s2d.coords
