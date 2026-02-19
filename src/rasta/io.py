"""I/O helpers producing canonical datasets."""

from __future__ import annotations

import pandas as pd
import xarray as xr

from .rao import RAOSet


def from_dataframe(df: pd.DataFrame, *, attrs: dict) -> RAOSet:
    """Build RAOSet from tidy DataFrame with columns freq, dir, resp, rao."""
    required = {"freq", "dir", "resp", "rao"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required DataFrame columns: {sorted(missing)}")

    ds = df.set_index(["freq", "dir", "resp"]).to_xarray()
    ds.attrs.update(attrs)
    return RAOSet(ds)


def to_dataframe(raoset: RAOSet) -> pd.DataFrame:
    return raoset.dataset[["rao"]].to_dataframe().reset_index()
