"""Scatter table representation and CSV I/O."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import xarray as xr


def _tp_tz_factor(*, model: str, gamma: float) -> float:
    key = model.lower()
    if key == "pm":
        return 1.401
    if key == "jonswap":
        return 1.280
    if key == "bretschneider":
        return 1.408
    if key == "goda":
        g = float(gamma)
        return 1.49 - 0.102 * g + 0.01420 * (g**2) - 0.00079 * (g**3)
    raise ValueError("conversion model must be one of: goda, pm, jonswap, bretschneider")


def _convert_period_coord(
    scatter: xr.Dataset,
    *,
    target: str,
    model: str,
    gamma: float,
) -> xr.Dataset:
    tgt = target.lower()
    if tgt not in {"tp", "tz"}:
        raise ValueError("target period must be 'tp' or 'tz'")

    has_tp = "tp" in scatter.coords
    has_tz = "tz" in scatter.coords
    if has_tp and has_tz:
        raise ValueError("scatter must not contain both 'tp' and 'tz' coordinates")
    if not has_tp and not has_tz:
        raise ValueError("scatter must contain either 'tp' or 'tz' coordinate")

    if (tgt == "tp" and has_tp) or (tgt == "tz" and has_tz):
        return scatter

    factor = _tp_tz_factor(model=model, gamma=gamma)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("invalid Tp/Tz conversion factor")

    out = scatter.copy()
    if has_tz and tgt == "tp":
        tz_vals = np.asarray(out.coords["tz"].values, dtype=float)
        tp_vals = tz_vals * factor
        out = out.rename({"tz": "tp"}).assign_coords(tp=("tp", tp_vals))
        out.attrs["tp_tz_conversion"] = f"tp=tz*{factor:.10g}"
        out.attrs["tp_tz_model"] = model.lower()
        out.attrs["tp_tz_gamma"] = float(gamma)
        out.attrs["tp_unit"] = "s"
        return out

    tp_vals = np.asarray(out.coords["tp"].values, dtype=float)
    tz_vals = tp_vals / factor
    out = out.rename({"tp": "tz"}).assign_coords(tz=("tz", tz_vals))
    out.attrs["tp_tz_conversion"] = f"tz=tp/{factor:.10g}"
    out.attrs["tp_tz_model"] = model.lower()
    out.attrs["tp_tz_gamma"] = float(gamma)
    out.attrs["tz_unit"] = "s"
    return out


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


def read_scatter_csv(
    path: str | Path,
    *,
    convert_period_to: str | None = None,
    conversion_model: str = "goda",
    gamma: float = 3.3,
) -> xr.Dataset:
    """Read tidy CSV scatter table with columns hs,tp,p or hs,tz,p.

    If `convert_period_to` is provided (`'tp'` or `'tz'`), period coordinates are
    converted using the selected relation:
    - `goda`: Tp = Tz * (1.49 - 0.102*gamma + 0.01420*gamma^2 - 0.00079*gamma^3)
    - `pm`: Tp = 1.401 * Tz
    - `jonswap`: Tp = 1.280 * Tz
    - `bretschneider`: Tp = 1.408 * Tz
    """
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
    out = validate_scatter(ds)
    if convert_period_to is not None:
        out = _convert_period_coord(
            out,
            target=convert_period_to,
            model=conversion_model,
            gamma=gamma,
        )
        out = validate_scatter(out)
    return out


def write_scatter_csv(
    scatter: xr.Dataset,
    path: str | Path,
    *,
    period: str | None = None,
    conversion_model: str = "goda",
    gamma: float = 3.3,
) -> None:
    """Write scatter table to tidy CSV with columns hs,tp,p or hs,tz,p.

    Use `period='tp'` or `period='tz'` to enforce output period type and convert
    coordinates when needed.
    """
    sc = validate_scatter(scatter)
    if period is not None:
        sc = _convert_period_coord(sc, target=period, model=conversion_model, gamma=gamma)
        sc = validate_scatter(sc)
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


def write_scatter_tab(
    scatter: xr.Dataset,
    path: str | Path,
    *,
    area_index: int = 1,
    season: str = "all",
    direction: str = "all",
    append: bool = False,
    float_format: str = ".10g",
    period: str | None = "tz",
    conversion_model: str = "goda",
    gamma: float = 3.3,
    drop_all_zero_rows: bool = False,
) -> None:
    """Write one scatter-table block in `scatter.tab` style format.

    The block starts with `area` in column 1, followed by descriptive header
    lines and a table with periods in the first numeric row and subsequent
    rows as `Hs p(Hs,Tz_1) ... p(Hs,Tz_n)`.
    """
    if area_index <= 0:
        raise ValueError("area_index must be >= 1")

    sc = validate_scatter(scatter)
    if period is not None:
        sc = _convert_period_coord(sc, target=period, model=conversion_model, gamma=gamma)
        sc = validate_scatter(sc)
    period_col = "tz" if "tz" in sc.coords else ("tp" if "tp" in sc.coords else None)
    if period_col is None:
        raise ValueError("scatter must provide either 'tz' or 'tp' coordinate")

    hs_vals = np.asarray(sc.coords["hs"].values, dtype=float)
    tz_vals = np.asarray(sc.coords[period_col].values, dtype=float)
    p = np.asarray(sc["p"].values, dtype=float)

    if p.shape != (hs_vals.size, tz_vals.size):
        raise ValueError("scatter table shape does not match hs/period coordinate sizes")

    mode = "a" if append else "w"
    out_path = Path(path)
    with out_path.open(mode, encoding="utf-8") as f:
        f.write(f"area : {int(area_index)} Nr. {int(area_index)}\n")
        f.write(f"season : {season}\n")
        f.write(f"direction : {direction}\n")

        f.write(" ".join(format(float(t), float_format) for t in tz_vals) + "\n")
        order_hs = np.argsort(hs_vals)[::-1]
        for i in order_hs:
            if drop_all_zero_rows and np.all(np.isclose(p[i, :], 0.0)):
                continue
            hs = hs_vals[i]
            row = [format(float(hs), float_format)]
            row.extend(format(float(v), float_format) for v in p[i, :])
            f.write(" ".join(row) + "\n")


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
