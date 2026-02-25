"""Long-term response statistics from scatter and operational profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..operational import make_operational_profile, validate_operational_profile
from ..rao import RAOSet
from ..scatter import validate_scatter
from .response_spectrum import compute_response_spectrum
from .shortterm import shortterm_statistics


@dataclass
class _BinResult:
    resp: str
    weight: float
    hs: float
    tp: float
    mean_dir: float
    speed: float | None
    depth: float | None
    sigma: float
    Tz: float
    Ncycles: float
    X_mpm: float


def _as_resp_list(resp: str | list[str]) -> list[str]:
    return [resp] if isinstance(resp, str) else [str(r) for r in resp]


def _select_condition(rs: RAOSet, speed: float | None, depth: float | None) -> RAOSet:
    ds = rs.dataset
    if speed is not None and "speed" in ds.coords and ds.coords["speed"].ndim == 1:
        ds = ds.sel(speed=float(speed), method="nearest")
    if depth is not None and "depth" in ds.coords and ds.coords["depth"].ndim == 1:
        ds = ds.sel(depth=float(depth), method="nearest")
    return RAOSet(ds)


def _shortterm_bin(
    rs: RAOSet,
    *,
    resp: str,
    hs: float,
    tp: float,
    mean_dir: float,
    spectrum_model: str,
    spreading: str | None,
    symmetry: bool,
) -> xr.Dataset:
    sr = compute_response_spectrum(
        rs,
        resp=resp,
        hs=hs,
        tp=tp,
        mean_dir=mean_dir,
        spectrum_model=spectrum_model,
        spreading=spreading,
        symmetry=symmetry,
    )["S_r"]
    return shortterm_statistics(sr, duration_s=3600.0)


def _collect_bins(
    rs: RAOSet,
    sc: xr.Dataset,
    *,
    resp_names: list[str],
    spectrum_model: str,
    spreading: str | None,
    symmetry: bool,
    operational_profile: xr.Dataset | None,
) -> list[_BinResult]:
    if operational_profile is None:
        mean_dirs = np.asarray(rs.dataset.coords["dir"].values, dtype=float)
        speeds = None
        depths = None
        if "speed" in rs.dataset.coords and rs.dataset.coords["speed"].ndim == 0:
            speeds = [float(rs.dataset.coords["speed"].values)]
        if "depth" in rs.dataset.coords and rs.dataset.coords["depth"].ndim == 0:
            depths = [float(rs.dataset.coords["depth"].values)]
        op = make_operational_profile(mean_dirs=mean_dirs, speeds=speeds, depths=depths)
    else:
        op = validate_operational_profile(operational_profile)

    bins: list[_BinResult] = []

    hs_vals = np.asarray(sc.coords["hs"].values, dtype=float)
    tp_vals = np.asarray(sc.coords["tp"].values, dtype=float)
    p_sc = np.asarray(sc["p"].values, dtype=float)

    w_op = op["w"]
    op_dims = list(w_op.dims)
    idx_ranges = [range(w_op.sizes[d]) for d in op_dims]

    for i_h, hs in enumerate(hs_vals):
        for i_t, tp in enumerate(tp_vals):
            p_state = float(p_sc[i_h, i_t])
            if p_state <= 0.0:
                continue

            for idx_tuple in np.ndindex(*[len(r) for r in idx_ranges]):
                selectors = {d: int(idx_ranges[k][idx_tuple[k]]) for k, d in enumerate(op_dims)}
                w = float(w_op.isel(**selectors).values)
                if w <= 0.0:
                    continue

                mean_dir = float(op.coords["mean_dir"].isel(mean_dir=selectors["mean_dir"]).values)
                speed = None
                depth = None
                if "speed" in selectors:
                    speed = float(op.coords["speed"].isel(speed=selectors["speed"]).values)
                if "depth" in selectors:
                    depth = float(op.coords["depth"].isel(depth=selectors["depth"]).values)

                rs_cond = _select_condition(rs, speed, depth)
                w_bin = p_state * w

                for r in resp_names:
                    st = _shortterm_bin(
                        rs_cond,
                        resp=r,
                        hs=float(hs),
                        tp=float(tp),
                        mean_dir=mean_dir,
                        spectrum_model=spectrum_model,
                        spreading=spreading,
                        symmetry=symmetry,
                    )
                    bins.append(
                        _BinResult(
                            resp=r,
                            weight=w_bin,
                            hs=float(hs),
                            tp=float(tp),
                            mean_dir=mean_dir,
                            speed=speed,
                            depth=depth,
                            sigma=float(st["sigma"].sel(resp=r).values),
                            Tz=float(st["Tz"].sel(resp=r).values),
                            Ncycles=float(st["Ncycles"].sel(resp=r).values),
                            X_mpm=float(st["X_mpm"].sel(resp=r).values),
                        )
                    )

    return bins


def _q_bin(x: np.ndarray, sigma: float, Ncycles: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    N = max(float(Ncycles), 1e-12)
    return 1.0 - np.exp(-N * np.exp(-(x**2) / (2.0 * sigma**2)))


def _interp_level(x: np.ndarray, y: np.ndarray, target: float) -> float:
    y_clip = np.clip(y, 0.0, 1.0)
    order = np.argsort(y_clip)
    ys = y_clip[order]
    xs = x[order]
    ys_unique, idx = np.unique(ys, return_index=True)
    xs_unique = xs[idx]
    if ys_unique.size == 1:
        return float(xs_unique[0])
    t = float(np.clip(target, ys_unique.min(), ys_unique.max()))
    return float(np.interp(t, ys_unique, xs_unique))


def longterm_statistics(
    rs: RAOSet,
    scatter: xr.Dataset,
    *,
    resp: str | list[str],
    years: float = 25.0,
    spectrum_model: str = "jonswap",
    spreading: str | None = None,
    symmetry: bool = True,
    operational_profile: xr.Dataset | None = None,
    exceedance_probs: list[float] | None = None,
    weibull_fit: bool = True,
    include_n_exceed: bool = False,
) -> xr.Dataset:
    """Compute long-term exceedance using bin aggregation.

    Parameters
    - `years`: exposure horizon for `P_exceed(resp, x)`.
    - `include_n_exceed`: if True, include `N_exceed(resp, x)`, the expected
      exceedance count over `years`, computed as `-ln(1 - P_exceed)`.
    """
    if years <= 0.0:
        raise ValueError("years must be > 0")

    sc = validate_scatter(scatter)
    if "tp" not in sc.coords:
        raise ValueError("v1 long-term statistics requires scatter with 'tp' coordinate")

    resp_names = _as_resp_list(resp)
    bins = _collect_bins(
        rs,
        sc,
        resp_names=resp_names,
        spectrum_model=spectrum_model,
        spreading=spreading,
        symmetry=symmetry,
        operational_profile=operational_profile,
    )

    if not bins:
        raise ValueError("no bins available for long-term computation")

    exceedance_probs = exceedance_probs or []
    hours_total = years * 365.25 * 24.0

    x_grid = np.linspace(0.0, max(b.sigma * np.sqrt(2.0 * np.log(max(b.Ncycles, 2.0))) for b in bins) * 6.0, 1200)

    p_exc_mat = np.zeros((len(resp_names), x_grid.size), dtype=float)
    n_exc_mat = np.zeros((len(resp_names), x_grid.size), dtype=float)

    design_hs = np.full(len(resp_names), np.nan)
    design_tp = np.full(len(resp_names), np.nan)
    design_mean_dir = np.full(len(resp_names), np.nan)
    design_speed = np.full(len(resp_names), np.nan)
    design_depth = np.full(len(resp_names), np.nan)
    design_weight = np.full(len(resp_names), np.nan)
    design_sigma = np.full(len(resp_names), np.nan)
    design_Tz = np.full(len(resp_names), np.nan)

    x_exceed = np.full((len(resp_names), len(exceedance_probs)), np.nan)
    weib_k = np.full(len(resp_names), np.nan)
    weib_l = np.full(len(resp_names), np.nan)

    for ir, r in enumerate(resp_names):
        rbins = [b for b in bins if b.resp == r]
        q_total = np.zeros_like(x_grid)
        for b in rbins:
            q_total += b.weight * _q_bin(x_grid, b.sigma, b.Ncycles)

        p_nonexc = np.exp(-hours_total * q_total)
        p_exc = 1.0 - p_nonexc
        p_exc_mat[ir, :] = p_exc
        n_exc_mat[ir, :] = -np.log(np.clip(1.0 - p_exc, 1e-300, 1.0))

        for ip, p in enumerate(exceedance_probs):
            x_exceed[ir, ip] = _interp_level(x_grid, p_exc, float(p))

        x_star = None
        if len(exceedance_probs) > 0 and np.isfinite(x_exceed[ir, 0]):
            x_star = float(x_exceed[ir, 0])
        else:
            x_star = float(np.nanpercentile(x_grid, 90.0))

        contrib = np.array([b.weight * _q_bin(np.array([x_star]), b.sigma, b.Ncycles)[0] for b in rbins])
        j = int(np.nanargmax(contrib))
        bstar = rbins[j]

        design_hs[ir] = bstar.hs
        design_tp[ir] = bstar.tp
        design_mean_dir[ir] = bstar.mean_dir
        design_speed[ir] = np.nan if bstar.speed is None else bstar.speed
        design_depth[ir] = np.nan if bstar.depth is None else bstar.depth
        design_weight[ir] = bstar.weight
        design_sigma[ir] = bstar.sigma
        design_Tz[ir] = bstar.Tz

        if weibull_fit:
            mask = (x_grid > 0.0) & (p_exc > 0.0) & (p_exc < 1.0)
            if np.count_nonzero(mask) > 10:
                lx = np.log(x_grid[mask])
                ly = np.log(-np.log(p_exc[mask]))
                A = np.vstack([lx, np.ones_like(lx)]).T
                k, b0 = np.linalg.lstsq(A, ly, rcond=None)[0]
                if np.isfinite(k) and k > 0:
                    weib_k[ir] = float(k)
                    weib_l[ir] = float(np.exp(-b0 / k))

    data_vars = {
        "P_exceed": (("resp", "x"), p_exc_mat),
        "x_exceed": (("resp", "exceedance_prob"), x_exceed),
        "design_hs": (("resp",), design_hs),
        "design_tp": (("resp",), design_tp),
        "design_mean_dir": (("resp",), design_mean_dir),
        "design_speed": (("resp",), design_speed),
        "design_depth": (("resp",), design_depth),
        "design_weight": (("resp",), design_weight),
        "design_sigma": (("resp",), design_sigma),
        "design_Tz": (("resp",), design_Tz),
        "weibull_k": (("resp",), weib_k),
        "weibull_lambda": (("resp",), weib_l),
    }
    if include_n_exceed:
        data_vars["N_exceed"] = (("resp", "x"), n_exc_mat)

    ds = xr.Dataset(
        data_vars,
        coords={
            "resp": np.array(resp_names, dtype=object),
            "x": x_grid,
            "exceedance_prob": np.asarray(exceedance_probs, dtype=float),
        },
        attrs={"years": float(years), "assumption": "Gaussian narrowband + Poisson aggregation"},
    )
    return ds


def longterm_response_cycle_counts(
    rs: RAOSet,
    scatter: xr.Dataset,
    *,
    resp: str | list[str],
    years: float = 25.0,
    spectrum_model: str = "jonswap",
    spreading: str | None = None,
    symmetry: bool = True,
    operational_profile: xr.Dataset | None = None,
    x_grid: xr.DataArray | np.ndarray | list[float] | None = None,
) -> xr.Dataset:
    """Expected long-term response-cycle exceedance counts versus response level.

    Returns `N_cycles_exceed(resp, x)`: expected number of response cycles with
    amplitude greater than `x` over `years`, using a narrowband Rayleigh tail
    approximation per sea-state bin.
    """
    if years <= 0.0:
        raise ValueError("years must be > 0")

    sc = validate_scatter(scatter)
    if "tp" not in sc.coords:
        raise ValueError("v1 long-term statistics requires scatter with 'tp' coordinate")

    resp_names = _as_resp_list(resp)
    bins = _collect_bins(
        rs,
        sc,
        resp_names=resp_names,
        spectrum_model=spectrum_model,
        spreading=spreading,
        symmetry=symmetry,
        operational_profile=operational_profile,
    )
    if not bins:
        raise ValueError("no bins available for long-term computation")

    if x_grid is None:
        xmax = max(b.sigma * np.sqrt(2.0 * np.log(max(b.Ncycles, 2.0))) for b in bins) * 6.0
        x_vals = np.linspace(0.0, xmax, 1200, dtype=float)
    else:
        x_vals = np.asarray(xr.DataArray(x_grid).values if isinstance(x_grid, xr.DataArray) else x_grid, dtype=float)
        if x_vals.ndim != 1:
            raise ValueError("x_grid must be 1-D")
        if x_vals.size == 0:
            raise ValueError("x_grid must not be empty")

    hours_total = years * 365.25 * 24.0
    n_mat = np.zeros((len(resp_names), x_vals.size), dtype=float)

    for ir, r in enumerate(resp_names):
        rbins = [b for b in bins if b.resp == r]
        n_tot = np.zeros_like(x_vals)
        for b in rbins:
            sigma = max(float(b.sigma), 1e-12)
            cycles_bin = hours_total * float(b.weight) * float(b.Ncycles)
            n_tot += cycles_bin * np.exp(-(x_vals**2) / (2.0 * sigma**2))
        n_mat[ir, :] = n_tot

    return xr.Dataset(
        {"N_cycles_exceed": (("resp", "x"), n_mat)},
        coords={"resp": np.array(resp_names, dtype=object), "x": x_vals},
        attrs={"years": float(years), "assumption": "Gaussian narrowband Rayleigh-cycle tail"},
    )
