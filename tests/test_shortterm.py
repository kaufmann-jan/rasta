import numpy as np
import xarray as xr

from rasta.rao import RAOSet
from rasta.stats.shortterm import response_variance, shortterm_statistics
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
