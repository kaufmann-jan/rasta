from __future__ import annotations

import numpy as np
import xarray as xr

from rasta.io import read_hydrostar_rao
from rasta.rao import RAOSet
from rasta.stats.response_spectrum import compute_response_spectrum, extend_symmetric_raos
from rasta.validation import REQUIRED_ATTRS


def test_response_spectrum_dims_and_finite() -> None:
    rs = read_hydrostar_rao("tests/hydrostar/heave.rao")
    out = compute_response_spectrum(
        rs,
        resp="heave",
        hs=3.0,
        tp=8.0,
        mean_dir=180.0,
        spectrum_model="jonswap",
    )

    assert "S_r" in out
    assert out["S_r"].dims[-1] == "freq"
    assert np.all(np.isfinite(out["S_r"].values))


def test_directional_spreading_option_works() -> None:
    rs = read_hydrostar_rao("tests/hydrostar/heave.rao")

    uni = compute_response_spectrum(rs, resp="heave", hs=3.0, tp=8.0, mean_dir=180.0, spreading=None)
    spread = compute_response_spectrum(rs, resp="heave", hs=3.0, tp=8.0, mean_dir=180.0, spreading="cos2")

    assert uni["S_r"].dims == spread["S_r"].dims
    assert np.all(np.isfinite(spread["S_r"].values))


def test_symmetry_extension_even_and_odd() -> None:
    freq = np.array([1.0, 2.0], dtype=float)
    direction = np.array([0.0, 90.0, 180.0], dtype=float)
    resp = np.array(["heave", "roll"], dtype=object)

    data = np.ones((2, 3, 2), dtype=np.complex128)
    ds = xr.Dataset(
        {"rao": (("resp", "dir", "freq"), data)},
        coords={"resp": resp, "dir": direction, "freq": freq},
        attrs=dict(REQUIRED_ATTRS),
    )

    out = extend_symmetric_raos(RAOSet(ds), symmetry=True)
    dirs = out.dataset.coords["dir"].values
    assert 270.0 in dirs

    heave_90 = out.rao.sel(resp="heave", dir=90.0)
    heave_270 = out.rao.sel(resp="heave", dir=270.0)
    roll_90 = out.rao.sel(resp="roll", dir=90.0)
    roll_270 = out.rao.sel(resp="roll", dir=270.0)

    assert np.allclose(heave_270.values, heave_90.values)
    assert np.allclose(roll_270.values, -roll_90.values)


def test_rao_tail_extrapolation_to_zero() -> None:
    freq = np.array([0.5, 1.0], dtype=float)
    direction = np.array([180.0], dtype=float)
    resp = np.array(["heave"], dtype=object)

    data = np.array([[[1.0 + 0.0j, 0.5 + 0.0j]]], dtype=np.complex128)
    ds = xr.Dataset(
        {"rao": (("resp", "dir", "freq"), data)},
        coords={"resp": resp, "dir": direction, "freq": freq},
        attrs=dict(REQUIRED_ATTRS),
    )

    rs = RAOSet(ds)
    omega = xr.DataArray(np.linspace(0.5, 2.0, 200), dims=("freq",), coords={"freq": np.linspace(0.5, 2.0, 200)})
    out = compute_response_spectrum(rs, resp="heave", hs=3.0, tp=8.0, mean_dir=180.0, omega_grid=omega)

    assert np.isfinite(out["S_r"].values).all()
    tail = float(out["S_r"].isel(resp=0, freq=-1).values)
    ref = float(out["S_r"].isel(resp=0, freq=10).values)
    assert tail < ref
