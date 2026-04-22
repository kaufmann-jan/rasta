from __future__ import annotations

import numpy as np
import xarray as xr

from rasta.io import read_hydrostar_rao
from rasta.operational import make_operational_profile
from rasta.scatter import validate_scatter
from rasta.stats.longterm import longterm_response_cycle_counts, longterm_statistics


def test_longterm_statistics_smoke() -> None:
    rs = read_hydrostar_rao("tests/hydrostar/heave.rao")

    scatter = xr.Dataset(
        {
            "p": (
                ("hs", "tp"),
                np.array(
                    [
                        [0.25, 0.25],
                        [0.20, 0.30],
                    ],
                    dtype=float,
                ),
            )
        },
        coords={"hs": np.array([2.0, 3.0]), "tp": np.array([7.0, 9.0])},
    )
    scatter = validate_scatter(scatter)

    profile = make_operational_profile(mean_dirs=np.array([150.0, 180.0, 210.0]))

    out = longterm_statistics(
        rs,
        scatter,
        resp="heave",
        years=1.0,
        operational_profile=profile,
        exceedance_probs=[1e-2, 1e-3],
        return_period_years=[1.0, 5.0],
    )

    assert "P_exceed" in out
    assert "x" in out.coords
    assert np.all(np.isfinite(out["P_exceed"].values))
    assert "design_hs" in out
    assert np.isfinite(float(out["design_hs"].sel(resp="heave").values))


def test_longterm_statistics_optional_return_periods() -> None:
    rs = read_hydrostar_rao("tests/hydrostar/heave.rao")

    scatter = xr.Dataset(
        {"p": (("hs", "tp"), np.array([[1.0]], dtype=float))},
        coords={"hs": np.array([2.5]), "tp": np.array([8.0])},
    )
    scatter = validate_scatter(scatter)
    profile = make_operational_profile(mean_dirs=np.array([180.0]))

    out = longterm_statistics(
        rs,
        scatter,
        resp="heave",
        years=25.0,
        operational_profile=profile,
        exceedance_probs=[1e-2],
        return_period_years=[100.0],
        weibull_fit=False,
    )

    x_exc = float(out["x_exceed"].sel(resp="heave", exceedance_prob=1e-2).values)
    x_ret = float(out["x_return"].sel(resp="heave", return_period_year=100.0).values)
    assert np.isfinite(x_exc)
    assert np.isfinite(x_ret)
    assert "N_exceed" not in out
    assert "return_period_year" in out.coords


def test_longterm_response_cycle_counts_monotonic() -> None:
    rs = read_hydrostar_rao("tests/hydrostar/heave.rao")
    scatter = xr.Dataset(
        {"p": (("hs", "tp"), np.array([[1.0]], dtype=float))},
        coords={"hs": np.array([2.5]), "tp": np.array([8.0])},
    )
    scatter = validate_scatter(scatter)
    profile = make_operational_profile(mean_dirs=np.array([180.0]))

    out = longterm_response_cycle_counts(
        rs,
        scatter,
        resp="heave",
        years=1.0,
        operational_profile=profile,
    )

    assert "N_cycles_exceed" in out
    n = out["N_cycles_exceed"].sel(resp="heave").values
    assert np.all(np.isfinite(n))
    assert float(n[0]) > 0.0
    # exceedance cycle count must be non-increasing with threshold x
    assert np.all(np.diff(n) <= 1e-10)


def test_longterm_statistics_accepts_spreading_kwargs() -> None:
    rs = read_hydrostar_rao("tests/hydrostar/heave.rao")
    scatter = xr.Dataset(
        {"p": (("hs", "tp"), np.array([[1.0]], dtype=float))},
        coords={"hs": np.array([2.5]), "tp": np.array([8.0])},
    )
    scatter = validate_scatter(scatter)
    profile = make_operational_profile(mean_dirs=np.array([180.0]))

    out_n2 = longterm_statistics(
        rs,
        scatter,
        resp="heave",
        years=1.0,
        operational_profile=profile,
        spreading="cosN_half",
        spreading_kwargs={"N": 2.0},
        exceedance_probs=[1e-2],
        weibull_fit=False,
    )
    out_n3 = longterm_statistics(
        rs,
        scatter,
        resp="heave",
        years=1.0,
        operational_profile=profile,
        spreading="cosN_half",
        spreading_kwargs={"N": 3.0},
        exceedance_probs=[1e-2],
        weibull_fit=False,
    )

    x2 = float(out_n2["x_exceed"].sel(resp="heave", exceedance_prob=1e-2).values)
    x3 = float(out_n3["x_exceed"].sel(resp="heave", exceedance_prob=1e-2).values)
    assert np.isfinite(x2)
    assert np.isfinite(x3)


def test_longterm_response_cycle_counts_accepts_spreading_kwargs() -> None:
    rs = read_hydrostar_rao("tests/hydrostar/heave.rao")
    scatter = xr.Dataset(
        {"p": (("hs", "tp"), np.array([[1.0]], dtype=float))},
        coords={"hs": np.array([2.5]), "tp": np.array([8.0])},
    )
    scatter = validate_scatter(scatter)
    profile = make_operational_profile(mean_dirs=np.array([180.0]))

    out = longterm_response_cycle_counts(
        rs,
        scatter,
        resp="heave",
        years=1.0,
        operational_profile=profile,
        spreading="cosN_half",
        spreading_kwargs={"N": 3.0},
    )

    assert "N_cycles_exceed" in out
    assert np.all(np.isfinite(out["N_cycles_exceed"].sel(resp="heave").values))
