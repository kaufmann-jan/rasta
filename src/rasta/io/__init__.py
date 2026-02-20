"""I/O helpers and HydroStar readers."""

from __future__ import annotations

import pandas as pd

from ..rao import RAOSet
from .hydrostar import (
    read_hydrostar_distributed_loads,
    read_hydrostar_rao,
    read_hydrostar_raos,
)


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


__all__ = [
    "from_dataframe",
    "to_dataframe",
    "read_hydrostar_rao",
    "read_hydrostar_raos",
    "read_hydrostar_distributed_loads",
]
