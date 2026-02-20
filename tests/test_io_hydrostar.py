from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rasta.io import (
    read_hydrostar_distributed_loads,
    read_hydrostar_rao,
    read_hydrostar_raos,
)
from rasta.validation import REQUIRED_ATTRS


FIXTURES = Path("tests/hydrostar")


def _first_amp_value(path: Path) -> float:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = [float(tok) for tok in s.split()]
            return vals[1]
    raise AssertionError("no numeric data row found")


def test_read_single_heave_file() -> None:
    rs = read_hydrostar_rao(FIXTURES / "heave.rao")

    assert list(rs.dataset.coords["resp"].values) == ["heave"]
    assert np.issubdtype(rs.rao.dtype, np.complexfloating)
    assert np.all(np.diff(rs.dataset.coords["freq"].values) > 0.0)
    assert np.array_equal(rs.dataset.coords["dir"].values, np.arange(0.0, 181.0, 15.0))

    for key, val in REQUIRED_ATTRS.items():
        assert rs.dataset.attrs[key] == val

    assert rs.dataset.attrs["source"] == "HydroStar"
    assert rs.dataset.attrs["hydrostar_raotype"] == "MOTION"
    assert rs.dataset.attrs["hydrostar_component"] == "3"


def test_rotation_component_is_scaled_to_rad() -> None:
    pitch_path = FIXTURES / "pitch.rao"
    rs = read_hydrostar_rao(pitch_path)

    raw_amp = _first_amp_value(pitch_path)
    expected = raw_amp * (np.pi / 180.0)
    got = float(np.abs(rs.rao.sel(resp="pitch").isel(dir=0, freq=0).values))

    assert np.isclose(got, expected, rtol=1e-8, atol=1e-12)


def test_internal_load_mapping_to_my() -> None:
    rs = read_hydrostar_rao(FIXTURES / "Mys1.rao")
    assert list(rs.dataset.coords["resp"].values) == ["My"]


def test_distributed_loads_concat_along_x() -> None:
    paths = [FIXTURES / f"Mys{i}.rao" for i in range(1, 10)]
    rs = read_hydrostar_distributed_loads(paths)

    assert rs.rao.dims == ("resp", "x", "dir", "freq")
    assert list(rs.dataset.coords["resp"].values) == ["My"]

    x_vals = rs.dataset.coords["x"].values
    assert np.all(np.diff(x_vals) > 0.0)
    assert np.allclose(x_vals, np.arange(13.5, 121.5 + 1e-12, 13.5))


def test_multi_file_concat_channels() -> None:
    rs = read_hydrostar_raos(
        [
            FIXTURES / "heave.rao",
            FIXTURES / "pitch.rao",
            FIXTURES / "Mys1.rao",
        ]
    )

    resp = list(rs.dataset.coords["resp"].values)
    assert len(resp) == 3
    assert set(resp) == {"heave", "pitch", "My"}


def test_read_hydrostar_raos_rejects_distributed_inputs() -> None:
    paths = [FIXTURES / f"Mys{i}.rao" for i in range(1, 10)]
    with pytest.raises(ValueError, match="distributed"):
        read_hydrostar_raos(paths)
