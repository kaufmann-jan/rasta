from __future__ import annotations

import numpy as np
import xarray as xr

from rasta.scatter import (
    load_iacs_rec34_rev2_scatter,
    read_scatter_csv,
    validate_scatter,
    write_scatter_csv,
)


def test_read_bundled_scatter() -> None:
    sc = load_iacs_rec34_rev2_scatter()
    assert "hs" in sc.coords
    assert "tp" in sc.coords
    assert "p" in sc
    assert np.isclose(float(sc["p"].sum().values), 1.0)
    assert np.all(sc["p"].values >= 0.0)


def test_read_write_roundtrip(tmp_path) -> None:
    ds = xr.Dataset(
        {"p": (("hs", "tp"), np.array([[0.2, 0.3], [0.1, 0.4]], dtype=float))},
        coords={"hs": np.array([1.0, 2.0]), "tp": np.array([6.0, 8.0])},
    )
    ds = validate_scatter(ds)

    path = tmp_path / "scatter.csv"
    write_scatter_csv(ds, path)
    loaded = read_scatter_csv(path)

    assert np.isclose(float(loaded["p"].sum().values), 1.0)
    assert set(loaded.dims) == {"hs", "tp"}
