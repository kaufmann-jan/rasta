from __future__ import annotations

import numpy as np
import xarray as xr

from rasta.scatter import (
    load_iacs_rec34_rev2_scatter,
    read_scatter_csv,
    validate_scatter,
    write_scatter_tab,
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


def test_read_scatter_csv_convert_tz_to_tp(tmp_path) -> None:
    path = tmp_path / "scatter_tz.csv"
    path.write_text("hs,tz,p\n1.0,6.0,0.4\n1.0,8.0,0.6\n", encoding="utf-8")

    sc = read_scatter_csv(path, convert_period_to="tp", conversion_model="jonswap")
    assert "tp" in sc.coords
    assert "tz" not in sc.coords
    assert np.allclose(sc.coords["tp"].values, np.array([6.0, 8.0]) * 1.280)
    assert np.isclose(float(sc["p"].sum().values), 1.0)


def test_write_scatter_csv_convert_tp_to_tz(tmp_path) -> None:
    ds = xr.Dataset(
        {"p": (("hs", "tp"), np.array([[0.2, 0.8]], dtype=float))},
        coords={"hs": np.array([1.0]), "tp": np.array([7.04, 8.448])},
    )
    path = tmp_path / "scatter_out.csv"
    write_scatter_csv(ds, path, period="tz", conversion_model="bretschneider")

    loaded = read_scatter_csv(path)
    assert "tz" in loaded.coords
    assert np.allclose(loaded.coords["tz"].values, np.array([7.04, 8.448]) / 1.408)


def test_write_scatter_tab_format_and_normalization(tmp_path) -> None:
    ds = xr.Dataset(
        {"p": (("hs", "tz"), np.array([[2.0, 3.0], [1.0, 4.0]], dtype=float))},
        coords={"hs": np.array([1.5, 3.5]), "tz": np.array([6.5, 8.5])},
    )

    path = tmp_path / "scatter.tab"
    write_scatter_tab(ds, path, area_index=16, season="winter", direction="all")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines[0] == "area : 16 Nr. 16"
    assert lines[1] == "season : winter"
    assert lines[2] == "direction : all"

    tz_vals = np.array([float(v) for v in lines[3].split()], dtype=float)
    assert np.allclose(tz_vals, np.array([6.5, 8.5]))
    # Hs rows must be written from large to small.
    assert float(lines[4].split()[0]) == 3.5
    assert float(lines[5].split()[0]) == 1.5

    p_sum = 0.0
    for row in lines[4:6]:
        vals = np.array([float(v) for v in row.split()], dtype=float)
        p_sum += float(np.sum(vals[1:]))
    assert np.isclose(p_sum, 1.0)


def test_write_scatter_tab_converts_tp_to_tz_by_default(tmp_path) -> None:
    ds = xr.Dataset(
        {"p": (("hs", "tp"), np.array([[0.5, 0.5]], dtype=float))},
        coords={"hs": np.array([1.5]), "tp": np.array([8.96, 10.368])},
    )
    path = tmp_path / "scatter.tab"
    write_scatter_tab(ds, path, area_index=1, conversion_model="bretschneider")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines[0] == "area : 1 Nr. 1"
    tz_vals = np.array([float(v) for v in lines[3].split()], dtype=float)
    assert np.allclose(tz_vals, np.array([8.96, 10.368]) / 1.408)


def test_write_scatter_tab_drop_all_zero_rows(tmp_path) -> None:
    ds = xr.Dataset(
        {"p": (("hs", "tz"), np.array([[0.0, 0.0], [0.2, 0.8]], dtype=float))},
        coords={"hs": np.array([5.0, 3.0]), "tz": np.array([6.5, 8.5])},
    )
    path = tmp_path / "scatter.tab"
    write_scatter_tab(ds, path, area_index=2, drop_all_zero_rows=True)

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # 3 header lines + period row + exactly one Hs row should remain.
    assert len(lines) == 5
    vals = np.array([float(v) for v in lines[4].split()], dtype=float)
    assert np.isclose(vals[0], 3.0)
