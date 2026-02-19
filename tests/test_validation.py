import numpy as np
import xarray as xr

from rasta.rao import RAOSet
from rasta.validation import REQUIRED_ATTRS, ValidationError


def make_dataset():
    freq = np.array([0.5, 1.0, 1.5], dtype=float)
    direction = np.array([180.0, 210.0], dtype=float)
    resp = np.array(["heave", "roll"], dtype=object)
    data = np.ones((freq.size, direction.size, resp.size), dtype=np.complex128)
    ds = xr.Dataset(
        {"rao": (("freq", "dir", "resp"), data)},
        coords={"freq": freq, "dir": direction, "resp": resp},
        attrs=dict(REQUIRED_ATTRS),
    )
    return ds


def test_valid_dataset_wraps():
    ds = make_dataset()
    out = RAOSet(ds)
    assert out.rao.dtype == np.complex128


def test_invalid_missing_attr():
    ds = make_dataset()
    del ds.attrs["freq_unit"]
    try:
        RAOSet(ds)
    except ValidationError:
        pass
    else:
        raise AssertionError("expected ValidationError")
