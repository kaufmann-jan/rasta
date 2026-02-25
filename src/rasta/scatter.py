"""Scatter table representation and CSV I/O."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import xarray as xr


def validate_scatter(scatter: xr.Dataset) -> xr.Dataset:
    """Validate and normalize scatter table dataset with variable `p`."""
    if "p" not in scatter.data_vars:
        raise ValueError("scatter must contain data variable 'p'")

    dims = tuple(scatter["p"].dims)
    if "hs" not in dims:
        raise ValueError("scatter must have 'hs' coordinate")
    if not ("tp" in dims or "tz" in dims):
        raise ValueError("scatter must have either 'tp' or 'tz' coordinate")

    p = scatter["p"].astype(float)
    if np.any(p.values < 0.0):
        raise ValueError("scatter probabilities must be non-negative")

    total = float(p.sum().values)
    if total <= 0.0:
        raise ValueError("scatter probabilities must sum to a positive value")

    out = scatter.copy()
    out["p"] = p / total
    return out


def read_scatter_csv(path: str | Path) -> xr.Dataset:
    """Read tidy CSV scatter table with columns hs,tp,p or hs,tz,p."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    names = {name.lower(): name for name in data.dtype.names or ()}

    if "hs" not in names or "p" not in names:
        raise ValueError("scatter CSV must include columns hs and p")

    period_col = "tp" if "tp" in names else ("tz" if "tz" in names else None)
    if period_col is None:
        raise ValueError("scatter CSV must include either tp or tz column")

    hs = np.asarray(data[names["hs"]], dtype=float)
    period = np.asarray(data[names[period_col]], dtype=float)
    prob = np.asarray(data[names["p"]], dtype=float)

    hs_u = np.unique(hs)
    p_u = np.unique(period)
    arr = np.zeros((hs_u.size, p_u.size), dtype=float)

    h_idx = {v: i for i, v in enumerate(hs_u.tolist())}
    t_idx = {v: i for i, v in enumerate(p_u.tolist())}

    for h, t, pr in zip(hs, period, prob):
        arr[h_idx[float(h)], t_idx[float(t)]] += float(pr)

    ds = xr.Dataset(
        {"p": (("hs", period_col), arr)},
        coords={"hs": hs_u.astype(float), period_col: p_u.astype(float)},
        attrs={"source": str(path), "hs_unit": "m", f"{period_col}_unit": "s"},
    )
    return validate_scatter(ds)


def write_scatter_csv(scatter: xr.Dataset, path: str | Path) -> None:
    """Write scatter table to tidy CSV with columns hs,tp,p or hs,tz,p."""
    sc = validate_scatter(scatter)
    period_col = "tp" if "tp" in sc.coords else "tz"

    hs_vals = np.asarray(sc.coords["hs"].values, dtype=float)
    p_vals = np.asarray(sc.coords[period_col].values, dtype=float)
    p = np.asarray(sc["p"].values, dtype=float)

    out = np.empty((hs_vals.size * p_vals.size, 3), dtype=float)
    k = 0
    for i, h in enumerate(hs_vals):
        for j, t in enumerate(p_vals):
            out[k, :] = (h, t, p[i, j])
            k += 1

    header = f"hs,{period_col},p"
    np.savetxt(path, out, delimiter=",", header=header, comments="", fmt="%.10g")


def load_iacs_rec34_rev2_scatter() -> xr.Dataset:
    """Load bundled IACS Rec. 34 Rev. 2 style example scatter table."""
    resource = files("rasta.resources").joinpath("iacs_rec34_rev2_scatter.csv")
    data = np.genfromtxt(resource, delimiter=",", names=True, dtype=float, encoding="utf-8")
    names = {name.lower(): name for name in data.dtype.names or ()}

    # Preferred tidy format.
    if "hs" in names and "p" in names and ("tp" in names or "tz" in names):
        return read_scatter_csv(resource)

    # Bundled IACS example format: hs,tm01,count
    if "hs" in names and "tm01" in names and "count" in names:
        hs = np.asarray(data[names["hs"]], dtype=float)
        tp = np.asarray(data[names["tm01"]], dtype=float)
        count = np.asarray(data[names["count"]], dtype=float)

        hs_u = np.unique(hs)
        tp_u = np.unique(tp)
        arr = np.zeros((hs_u.size, tp_u.size), dtype=float)
        h_idx = {v: i for i, v in enumerate(hs_u.tolist())}
        t_idx = {v: i for i, v in enumerate(tp_u.tolist())}

        for h, t, c in zip(hs, tp, count):
            arr[h_idx[float(h)], t_idx[float(t)]] += float(c)

        ds = xr.Dataset(
            {"p": (("hs", "tp"), arr)},
            coords={"hs": hs_u.astype(float), "tp": tp_u.astype(float)},
            attrs={"source": str(resource), "hs_unit": "m", "tp_unit": "s"},
        )
        return validate_scatter(ds)

    raise ValueError("unsupported bundled scatter resource format")
