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
    )

    assert "P_exceed" in out
    assert "x" in out.coords
    assert np.all(np.isfinite(out["P_exceed"].values))
    assert "design_hs" in out
    assert np.isfinite(float(out["design_hs"].sel(resp="heave").values))


def test_longterm_statistics_optional_n_exceed() -> None:
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
        include_n_exceed=True,
        weibull_fit=False,
    )

    assert "N_exceed" in out
    p = out["P_exceed"].sel(resp="heave").values
    n = out["N_exceed"].sel(resp="heave").values
    assert np.allclose(n, -np.log(np.clip(1.0 - p, 1e-300, 1.0)))

    x_exc = float(out["x_exceed"].sel(resp="heave", exceedance_prob=1e-2).values)
    assert np.isfinite(x_exc)
    assert "x_return" not in out
    assert "return_period_year" not in out.coords


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
