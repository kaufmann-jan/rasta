from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from rasta.preprocess import point_acceleration
from rasta.rao import RAOSet
from rasta.validation import REQUIRED_ATTRS


def _motion_rs() -> RAOSet:
    freq = np.array([2.0], dtype=float)
    direction = np.array([180.0], dtype=float)
    resp = np.array(["surge", "sway", "heave", "roll", "pitch", "yaw"], dtype=object)
    data = np.array(
        [
            [[1.0 + 0.0j]],
            [[2.0 + 0.0j]],
            [[3.0 + 0.0j]],
            [[0.1 + 0.0j]],
            [[0.2 + 0.0j]],
            [[0.3 + 0.0j]],
        ],
        dtype=np.complex128,
    )
    ds = xr.Dataset(
        {"rao": (("resp", "dir", "freq"), data)},
        coords={"resp": resp, "dir": direction, "freq": freq},
        attrs=dict(REQUIRED_ATTRS),
    )
    return RAOSet(ds)


def _motion_and_acc_rs() -> RAOSet:
    rs = _motion_rs()
    omega2 = float(rs.dataset.coords["freq"].values[0] ** 2)
    acc = -(omega2) * rs.rao.values
    resp_acc = np.array(["surge_acc", "sway_acc", "heave_acc", "roll_acc", "pitch_acc", "yaw_acc"], dtype=object)
    data = np.concatenate([rs.rao.values, acc], axis=0)
    resp = np.concatenate([rs.dataset.coords["resp"].values, resp_acc], axis=0)
    ds = xr.Dataset(
        {"rao": (("resp", "dir", "freq"), data)},
        coords={"resp": resp, "dir": rs.dataset.coords["dir"].values, "freq": rs.dataset.coords["freq"].values},
        attrs=dict(rs.dataset.attrs),
    )
    return RAOSet(ds)


def test_point_acceleration_zero_offset() -> None:
    rs = _motion_rs()
    out = point_acceleration(rs, point=(0.0, 0.0, 0.0))
    omega2 = 4.0
    assert np.isclose(out.rao.sel(resp="point_x_acc").values.item().real, -omega2 * 1.0)
    assert np.isclose(out.rao.sel(resp="point_y_acc").values.item().real, -omega2 * 2.0)
    assert np.isclose(out.rao.sel(resp="point_z_acc").values.item().real, -omega2 * 3.0)


def test_point_acceleration_pitch_contribution() -> None:
    rs = _motion_rs()
    out = point_acceleration(rs, point=(0.0, 0.0, 10.0))
    expected = -(4.0 * 1.0) + (-(4.0 * 0.2) * 10.0)
    assert np.isclose(out.rao.sel(resp="point_x_acc").values.item().real, expected)


def test_point_acceleration_yaw_contribution() -> None:
    rs = _motion_rs()
    out = point_acceleration(rs, point=(5.0, 0.0, 0.0))
    expected = -(4.0 * 2.0) + (-(4.0 * 0.3) * 5.0)
    assert np.isclose(out.rao.sel(resp="point_y_acc").values.item().real, expected)


def test_point_acceleration_roll_contribution() -> None:
    rs = _motion_rs()
    out = point_acceleration(rs, point=(0.0, 4.0, 0.0))
    expected = -(4.0 * 3.0) + (-(4.0 * 0.1) * 4.0)
    assert np.isclose(out.rao.sel(resp="point_z_acc").values.item().real, expected)


def test_point_acceleration_derived_vs_explicit() -> None:
    rs = _motion_and_acc_rs()
    derived = point_acceleration(rs, point=(5.0, 0.0, 0.0), point_name="bow", derive_accelerations=True)
    explicit = point_acceleration(rs, point=(5.0, 0.0, 0.0), point_name="bow", derive_accelerations=False)
    assert np.allclose(derived.rao.sel(resp="bow_y_acc").values, explicit.rao.sel(resp="bow_y_acc").values)


def test_point_acceleration_relative_vs_absolute_equivalence() -> None:
    rs = _motion_rs()
    ds_abs = rs.dataset.copy()
    ds_abs.attrs.update({"xref": 50.0, "yref": 0.0, "zref": 0.0})
    rs_abs = RAOSet(ds_abs)
    rel = point_acceleration(rs, point=(20.0, 0.0, 0.0), point_mode="relative", point_name="bow")
    abs_ = point_acceleration(rs_abs, point=(70.0, 0.0, 0.0), point_mode="absolute", point_name="bow")
    assert np.allclose(rel.rao.sel(resp="bow_y_acc").values, abs_.rao.sel(resp="bow_y_acc").values)


def test_point_acceleration_absolute_missing_reference_point() -> None:
    rs = _motion_rs()
    with pytest.raises(ValueError, match="xref"):
        point_acceleration(rs, point=(70.0, 0.0, 0.0), point_mode="absolute")


def test_point_acceleration_requires_explicit_acc_when_disabled() -> None:
    rs = _motion_rs()
    with pytest.raises(ValueError, match="acceleration RAOs missing"):
        point_acceleration(rs, point=(0.0, 0.0, 0.0), derive_accelerations=False)
