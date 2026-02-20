"""HydroStar .rao readers producing canonical rasta RAOSet objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import xarray as xr

from ..rao import RAOSet
from ..validation import REQUIRED_ATTRS

_FLOAT_RE = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?"

_MOTION_MAP = {
    1: "surge",
    2: "sway",
    3: "heave",
    4: "roll",
    5: "pitch",
    6: "yaw",
}

_LOAD_MAP = {
    1: "Fx",
    2: "Fy",
    3: "Fz",
    4: "Mx",
    5: "My",
    6: "Mz",
}


@dataclass(frozen=True)
class _ParsedHydroStar:
    path: Path
    raotype: str
    component: int
    unit: str
    headings_deg: np.ndarray
    freq_rad_s: np.ndarray
    amp: np.ndarray
    phase_deg: np.ndarray
    speed_m_s: float | None
    depth_m: float | None
    x_ref: float | None
    y_ref: float | None
    z_ref: float | None


def _extract_float(pattern: str, line: str) -> float | None:
    match = re.search(pattern, line)
    if not match:
        return None
    return float(match.group(1))


def _infer_resp(raotype: str, component: int) -> str:
    key = raotype.upper()
    if key == "MOTION":
        mapping = _MOTION_MAP
    elif key in {"INTERNALLOAD", "EXTERNALLOAD"}:
        mapping = _LOAD_MAP
    else:
        raise ValueError(f"unsupported RAOTYPE '{raotype}'")

    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(f"unsupported COMPONENT '{component}' for RAOTYPE '{raotype}'") from exc


def _validate_resp_name(resp: str) -> None:
    if re.search(r"\d+$", resp):
        raise ValueError("resp must not contain numeric suffixes")


def _parse_hydrostar_file(path: str | Path) -> _ParsedHydroStar:
    file_path = Path(path)

    raotype: str | None = None
    component: int | None = None
    unit: str | None = None
    nbheading: int | None = None
    headings_deg: np.ndarray | None = None
    speed_m_s: float | None = None
    depth_m: float | None = None
    x_ref: float | None = None
    y_ref: float | None = None
    z_ref: float | None = None

    freq_rows: list[float] = []
    amp_rows: list[np.ndarray] = []
    phase_rows: list[np.ndarray] = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                if "RAOTYPE" in line:
                    match = re.search(r"#\s*RAOTYPE\s*:\s*(\S+)", line)
                    if match:
                        raotype = match.group(1).strip()
                elif "COMPONENT" in line:
                    match = re.search(r"#\s*COMPONENT\s*:\s*(\d+)", line)
                    if match:
                        component = int(match.group(1))
                elif "UNIT" in line:
                    match = re.search(r"#\s*UNIT\s*:\s*(.+)$", line)
                    if match:
                        unit = match.group(1).strip()
                elif "NBHEADING" in line:
                    match = re.search(r"#\s*NBHEADING\s*:?\s*(\d+)", line)
                    if match:
                        nbheading = int(match.group(1))
                elif line.startswith("#HEADING"):
                    vals = re.findall(_FLOAT_RE, line)
                    if vals:
                        headings_deg = np.array([float(v) for v in vals], dtype=float)
                elif "Forward speed" in line:
                    speed_m_s = _extract_float(rf"Forward speed\s*:\s*({_FLOAT_RE})", line)
                elif "Waterdepth" in line:
                    depth_m = _extract_float(rf"Waterdepth\s*:\s*({_FLOAT_RE})", line)
                elif "Reference point of body 1" in line:
                    match = re.search(
                        rf"Reference point of body 1:\s*\(\s*({_FLOAT_RE})\s+({_FLOAT_RE})\s+({_FLOAT_RE})\s*\)",
                        line,
                    )
                    if match:
                        x_ref = float(match.group(1))
                        y_ref = float(match.group(2))
                        z_ref = float(match.group(3))
                continue

            if nbheading is None:
                raise ValueError(f"{file_path}: data row before NBHEADING at line {line_no}")

            try:
                values = [float(tok) for tok in line.split()]
            except ValueError as exc:
                raise ValueError(f"{file_path}: failed parsing numeric row at line {line_no}") from exc

            expected = 1 + 2 * nbheading
            if len(values) != expected:
                raise ValueError(
                    f"{file_path}: heading/data count mismatch at line {line_no}; "
                    f"expected {expected} floats, got {len(values)}"
                )

            freq_rows.append(values[0])
            amp_rows.append(np.asarray(values[1 : 1 + nbheading], dtype=float))
            phase_rows.append(np.asarray(values[1 + nbheading :], dtype=float))

    if raotype is None:
        raise ValueError(f"{file_path}: missing RAOTYPE")
    if component is None:
        raise ValueError(f"{file_path}: missing COMPONENT")
    if unit is None:
        raise ValueError(f"{file_path}: missing UNIT")
    if nbheading is None:
        raise ValueError(f"{file_path}: missing NBHEADING")
    if headings_deg is None:
        raise ValueError(f"{file_path}: missing HEADING values")
    if headings_deg.size != nbheading:
        raise ValueError(
            f"{file_path}: HEADING count mismatch, expected {nbheading}, got {headings_deg.size}"
        )
    if not freq_rows:
        raise ValueError(f"{file_path}: no data rows found")

    return _ParsedHydroStar(
        path=file_path,
        raotype=raotype,
        component=component,
        unit=unit,
        headings_deg=headings_deg,
        freq_rad_s=np.asarray(freq_rows, dtype=float),
        amp=np.asarray(amp_rows, dtype=float),
        phase_deg=np.asarray(phase_rows, dtype=float),
        speed_m_s=speed_m_s,
        depth_m=depth_m,
        x_ref=x_ref,
        y_ref=y_ref,
        z_ref=z_ref,
    )


def _complex_rao(parsed: _ParsedHydroStar) -> np.ndarray:
    phase_rad = np.deg2rad(parsed.phase_deg)
    rao = parsed.amp * np.exp(1j * phase_rad)

    if parsed.raotype.upper() == "MOTION" and parsed.component in {4, 5, 6} and parsed.unit.strip().lower() == "deg/m":
        rao = rao * (np.pi / 180.0)

    return np.asarray(rao, dtype=np.complex128)


def _build_dataset(parsed: _ParsedHydroStar, resp_name: str) -> xr.Dataset:
    _validate_resp_name(resp_name)

    rao_fd = _complex_rao(parsed)
    data = rao_fd.T[np.newaxis, :, :]

    ds = xr.Dataset(
        data_vars={"rao": (("resp", "dir", "freq"), data.astype(np.complex128, copy=False))},
        coords={
            "resp": np.array([resp_name], dtype=object),
            "dir": parsed.headings_deg.astype(float, copy=False),
            "freq": parsed.freq_rad_s.astype(float, copy=False),
        },
        attrs={
            **REQUIRED_ATTRS,
            "source": "HydroStar",
            "hydrostar_raotype": parsed.raotype,
            "hydrostar_component": str(parsed.component),
            "hydrostar_unit": parsed.unit,
        },
    )

    if parsed.speed_m_s is not None:
        ds = ds.assign_coords(speed=float(parsed.speed_m_s))
    if parsed.depth_m is not None:
        ds = ds.assign_coords(depth=float(parsed.depth_m))

    return ds


def _assert_same_grid(reference: xr.Dataset, other: xr.Dataset, *, src: Path) -> None:
    if not np.array_equal(reference.coords["freq"].values, other.coords["freq"].values):
        raise ValueError(f"{src}: freq grid differs from previous files")
    if not np.array_equal(reference.coords["dir"].values, other.coords["dir"].values):
        raise ValueError(f"{src}: dir grid differs from previous files")


def _assert_same_optional_scalars(reference: xr.Dataset, other: xr.Dataset, *, src: Path) -> None:
    for cname in ("speed", "depth"):
        ref_has = cname in reference.coords
        oth_has = cname in other.coords
        if ref_has != oth_has:
            raise ValueError(f"{src}: inconsistent '{cname}' coordinate across files")
        if ref_has and oth_has:
            ref_val = float(reference.coords[cname].values)
            oth_val = float(other.coords[cname].values)
            if not np.isclose(ref_val, oth_val):
                raise ValueError(f"{src}: inconsistent '{cname}' coordinate across files")


def _resp_from_map(path: Path, resp_map: dict[str, str] | None) -> str | None:
    if resp_map is None:
        return None

    candidates = (str(path), path.name)
    for key in candidates:
        if key in resp_map:
            return resp_map[key]
    return None


def read_hydrostar_rao(path: str | Path, *, resp: str | None = None) -> RAOSet:
    """Read one HydroStar .rao file into canonical RAOSet with one response channel."""
    parsed = _parse_hydrostar_file(path)
    resp_name = resp if resp is not None else _infer_resp(parsed.raotype, parsed.component)
    ds = _build_dataset(parsed, resp_name)
    return RAOSet(ds)


def read_hydrostar_raos(paths: Iterable[str | Path], *, resp_map: dict[str, str] | None = None) -> RAOSet:
    """Read multiple HydroStar .rao files and concatenate along resp."""
    path_list = [Path(p) for p in paths]
    if not path_list:
        raise ValueError("paths must not be empty")

    ras: list[RAOSet] = []
    parsed_items = [_parse_hydrostar_file(p) for p in path_list]

    for parsed in parsed_items:
        mapped = _resp_from_map(parsed.path, resp_map)
        resp_name = mapped if mapped is not None else _infer_resp(parsed.raotype, parsed.component)
        ras.append(RAOSet(_build_dataset(parsed, resp_name)))

    resp_names = [str(ra.dataset.coords["resp"].values[0]) for ra in ras]
    if len(set(resp_names)) != len(resp_names):
        raise ValueError(
            "duplicate resp names detected; use read_hydrostar_distributed_loads for distributed channels"
        )

    base = ras[0].dataset
    for ra, parsed in zip(ras[1:], parsed_items[1:]):
        _assert_same_grid(base, ra.dataset, src=parsed.path)
        _assert_same_optional_scalars(base, ra.dataset, src=parsed.path)

    out = xr.concat([ra.dataset for ra in ras], dim="resp")
    return RAOSet(out)


def read_hydrostar_distributed_loads(paths: Iterable[str | Path], *, resp: str | None = None) -> RAOSet:
    """Read distributed HydroStar internal load files and concatenate along x."""
    path_list = [Path(p) for p in paths]
    if not path_list:
        raise ValueError("paths must not be empty")

    parsed_items = [_parse_hydrostar_file(p) for p in path_list]

    if any(item.x_ref is None for item in parsed_items):
        raise ValueError("distributed-load file missing reference point header")

    datasets: list[xr.Dataset] = []
    inferred: str | None = None

    for item in parsed_items:
        name = resp if resp is not None else _infer_resp(item.raotype, item.component)
        _validate_resp_name(name)
        if inferred is None:
            inferred = name
        elif name != inferred:
            raise ValueError("distributed load files must map to a single resp")

        rao_fd = _complex_rao(item)
        data = rao_fd.T[np.newaxis, np.newaxis, :, :]

        ds = xr.Dataset(
            data_vars={"rao": (("resp", "x", "dir", "freq"), data.astype(np.complex128, copy=False))},
            coords={
                "resp": np.array([name], dtype=object),
                "x": np.array([float(item.x_ref)], dtype=float),
                "dir": item.headings_deg.astype(float, copy=False),
                "freq": item.freq_rad_s.astype(float, copy=False),
            },
            attrs={
                **REQUIRED_ATTRS,
                "source": "HydroStar",
                "hydrostar_raotype": item.raotype,
                "hydrostar_component": str(item.component),
                "hydrostar_unit": item.unit,
            },
        )

        if item.speed_m_s is not None:
            ds = ds.assign_coords(speed=float(item.speed_m_s))
        if item.depth_m is not None:
            ds = ds.assign_coords(depth=float(item.depth_m))

        # Keep full 3D reference-point data aligned with x.
        ds = ds.assign_coords(y=("x", np.array([float(item.y_ref)], dtype=float)))
        ds = ds.assign_coords(z=("x", np.array([float(item.z_ref)], dtype=float)))

        datasets.append(ds)

    base = datasets[0]
    for ds, item in zip(datasets[1:], parsed_items[1:]):
        _assert_same_grid(base, ds, src=item.path)
        _assert_same_optional_scalars(base, ds, src=item.path)

    out = xr.concat(datasets, dim="x")

    x_vals = out.coords["x"].values
    if np.unique(x_vals).size != x_vals.size:
        raise ValueError("duplicate x reference points detected")

    out = out.sortby("x")
    return RAOSet(out)
