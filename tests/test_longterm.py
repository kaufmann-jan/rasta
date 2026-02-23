from __future__ import annotations

import numpy as np
import xarray as xr

from rasta.io import read_hydrostar_rao
from rasta.operational import make_operational_profile
from rasta.scatter import validate_scatter
from rasta.stats.longterm import longterm_statistics


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
