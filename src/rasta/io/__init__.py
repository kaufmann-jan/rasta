"""I/O helpers and HydroStar readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


def write_raoset_xtf(
    raoset: RAOSet,
    path: str | Path,
    *,
    speed: float | None = None,
    g: float = 9.81,
    float_format: str = ".12g",
) -> None:
    """Write RAOSet to line-based transfer-function text format.

    Per output line:
    speed[m/s], wavefreq[rad/s], wavedir[deg], wavelength[m], followed by
    arbitrary many (real, imag) transfer-function pairs.
    """
    ds = raoset.dataset
    rao = ds["rao"]

    if "freq" not in rao.dims or "dir" not in rao.dims:
        raise ValueError("RAOSet must contain 'freq' and 'dir' dimensions")

    if g <= 0.0:
        raise ValueError("g must be > 0")

    if speed is None:
        if "speed" in ds.coords and ds.coords["speed"].ndim == 0:
            speed_val = float(ds.coords["speed"].values)
        else:
            speed_val = 0.0
    else:
        speed_val = float(speed)

    transfer_dims = [d for d in rao.dims if d not in {"freq", "dir"}]
    if not transfer_dims:
        raise ValueError("RAOSet must include at least one transfer-function dimension besides freq/dir")

    ordered = rao.transpose(*transfer_dims, "dir", "freq")
    freqs = np.asarray(ordered.coords["freq"].values, dtype=float)
    dirs = np.asarray(ordered.coords["dir"].values, dtype=float)

    out_path = Path(path)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# speed[m/s] wavefreq[rad/s] wavedir[deg] wavelength[m] (Re Im)*\n")
        for omega in freqs:
            wavelength = (2.0 * np.pi * g / (omega * omega)) if omega > 0.0 else np.inf
            for direction in dirs:
                tf = ordered.sel(freq=float(omega), dir=float(direction)).values
                tf_flat = np.asarray(tf, dtype=np.complex128).reshape(-1)

                vals: list[float] = [speed_val, float(omega), float(direction), float(wavelength)]
                for z in tf_flat:
                    vals.extend((float(np.real(z)), float(np.imag(z))))

                line = " ".join(format(v, float_format) for v in vals)
                f.write(line + "\n")


__all__ = [
    "from_dataframe",
    "to_dataframe",
    "write_raoset_xtf",
    "read_hydrostar_rao",
    "read_hydrostar_raos",
    "read_hydrostar_distributed_loads",
]
