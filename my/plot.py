#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""
Reproduce the active Gao+2023 figure subset from local Gao+2023 outputs.

This module is intentionally pragmatic: it builds the maintained 10-figure
reproduction subset from local Gao+2023 catalog outputs and the local MPB
table generated in the Gao+2023 workflow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman",
                     "mathtext.default": "regular",
                     "xtick.direction": "in",
                     "ytick.direction": "in",
                     "text.usetex": True,
                     "text.latex.preamble": r"\usepackage{amsmath} \usepackage{bm}"})
import numpy as np
import pandas as pd
from pathlib import Path
import re
import shutil
from typing import Dict, List, Sequence, Tuple

STD_DPI = 512

"""Local MW/M31 NSC+SMBH constants for Gao+2023 plotting."""

M_SMBH_MW = 4.297e6
M_SMBH_MW_err = 0.012e6

M_NSC_MW = 3.15e7
M_NSC_MW_err = 2.15e7
R_NSC_MW = 5.7
R_NSC_MW_err = 3.5

M_SMBH_M31 = 1.7e8
M_SMBH_M31_err = 0.6e8

M_NSC_M31 = 5.0e7
M_NSC_M31_err = 0.0
R_NSC_M31 = 8.0
R_NSC_M31_err = 4.0

ALLCAT_COLUMNS = [
    "hid_z0",
    "logMh_z0",
    "logMstar_z0",
    "logMh_form",
    "logMstar_form",
    "logM_form",
    "zform",
    "feh",
    "isMPB",
    "subfind_form",
    "snap_form",
]
ALLCAT_OPTIONAL_RADIUS_COLUMN = "r_galaxy_kpc"
RUN_METADATA_NAME = "run_metadata.json"


@dataclass
class ModelResult:
    """Container for one Ns model track."""

    ns_value: float
    r_init: np.ndarray
    r_final: np.ndarray
    m_final: np.ndarray
    deposit_profile: "DepositProfile | None" = None
    halo_summary: pd.DataFrame | None = None


@dataclass
class DepositProfile:
    """Final deposited shell masses for one Ns model across all halos."""

    halo_ids: np.ndarray
    r_inner_kpc: List[np.ndarray]
    r_outer_kpc: List[np.ndarray]
    shell_mass_msun: List[np.ndarray]
    cumulative_mass_msun: List[np.ndarray] | None = None


@dataclass(frozen=True)
class GalaxyObs:
    """Reference NSC/SMBH measurements used for overlays."""

    name: str
    m_smbh: float
    m_smbh_err: float
    m_nsc: float
    m_nsc_err: float
    r_nsc_pc: float
    r_nsc_err_pc: float
    color: str


def _safe_log10(arr: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    """Return log10 with a small floor to avoid -inf values."""

    return np.log10(np.clip(arr, floor, None))


def _log10_positive_or_nan(arr: np.ndarray) -> np.ndarray:
    """Return log10 for positive values and NaN elsewhere."""

    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr) & (arr > 0)
    out[mask] = np.log10(arr[mask])
    return out


def _apply_plot_settings_from_data() -> None:
    """Apply the local Gao+2023 plotting style.

    The suggestion file requests TeX rendering. To keep this script runnable
    on systems without a local LaTeX installation, we only enable TeX when a
    ``latex`` binary is available.
    """

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.default": "regular",
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    if shutil.which("latex") is not None:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath} \usepackage{bm}",
            }
        )
    else:
        plt.rcParams.update({"text.usetex": False})


def _get_mw_m31_observations() -> Tuple[GalaxyObs, GalaxyObs]:
    """Read MW/M31 NSC and SMBH reference values from ``data.py``."""

    mw = GalaxyObs(
        name="MW",
        m_smbh=float(M_SMBH_MW),
        m_smbh_err=float(M_SMBH_MW_err),
        m_nsc=float(M_NSC_MW),
        m_nsc_err=float(M_NSC_MW_err),
        r_nsc_pc=float(R_NSC_MW),
        r_nsc_err_pc=float(R_NSC_MW_err),
        color="tab:red",
    )
    m31 = GalaxyObs(
        name="M31",
        m_smbh=float(M_SMBH_M31),
        m_smbh_err=float(M_SMBH_M31_err),
        m_nsc=float(M_NSC_M31),
        m_nsc_err=float(M_NSC_M31_err),
        r_nsc_pc=float(R_NSC_M31),
        r_nsc_err_pc=float(R_NSC_M31_err),
        color="tab:purple",
    )
    return mw, m31


def _add_nsc_smbh_points(ax: plt.Axes, obs: GalaxyObs, show_labels: bool = True) -> None:
    """Overlay separate SMBH and NSC reference masses on one axis."""

    r_kpc = max(obs.r_nsc_pc / 1000.0, 1.0e-4)
    xr = obs.r_nsc_err_pc / 1000.0
    xerr = xr if xr > 0 else None

    ax.errorbar(
        [r_kpc],
        [obs.m_smbh],
        xerr=xerr,
        yerr=obs.m_smbh_err if obs.m_smbh_err > 0 else None,
        fmt="^",
        ms=6,
        mfc="white",
        mec=obs.color,
        color=obs.color,
        capsize=3,
        zorder=7,
        label=f"{obs.name} SMBH" if show_labels else None,
    )
    ax.errorbar(
        [r_kpc],
        [obs.m_nsc],
        xerr=xerr,
        yerr=obs.m_nsc_err if obs.m_nsc_err > 0 else None,
        fmt="o",
        ms=6,
        mfc="white",
        mec=obs.color,
        color=obs.color,
        capsize=3,
        zorder=7,
        label=f"{obs.name} NSC" if show_labels else None,
    )


def _add_nsc_smbh_points_pc(ax: plt.Axes, obs: GalaxyObs, show_labels: bool = True) -> None:
    """Overlay separate SMBH and NSC reference masses using radius in pc."""

    r_pc = max(obs.r_nsc_pc, 1.0e-2)
    xerr = obs.r_nsc_err_pc if obs.r_nsc_err_pc > 0 else None

    ax.errorbar(
        [r_pc],
        [obs.m_smbh],
        xerr=xerr,
        yerr=obs.m_smbh_err if obs.m_smbh_err > 0 else None,
        fmt="o",
        ms=6,
        mfc="white",
        mec=obs.color,
        color=obs.color,
        capsize=3,
        zorder=7,
        label=f"{obs.name} SMBH" if show_labels else None,
    )
    ax.errorbar(
        [r_pc],
        [obs.m_nsc],
        xerr=xerr,
        yerr=obs.m_nsc_err if obs.m_nsc_err > 0 else None,
        fmt="s",
        ms=6,
        mfc="white",
        mec=obs.color,
        color=obs.color,
        capsize=3,
        zorder=7,
        label=f"{obs.name} NSC" if show_labels else None,
    )


def _load_observational_overlays() -> Dict[str, object]:
    """Return observational anchors used in Gao+2023 figure overlays.

    The points/curves below are compact digitized approximations from the
    original Gao+2023 figures and references listed in their captions.
    They are intentionally lightweight and self-contained so reproduction
    works without external catalog files.
    """

    # Fig. 2: observed m_GC / m_halo ratios from literature compilations.
    fig2_ratio_refs = {
        "S09": 7.0e-5,
        "G10": 5.5e-5,
        "H14": 2.8e-5,
        "H17": 4.0e-5,
    }

    # Fig. 3 / 7: digitized directly from Gao+2023 Fig. 3.
    fig3_g14 = {
        "r_kpc": np.array(
            [0.0186, 0.0264, 0.0473, 0.0842, 0.4465, 0.6181, 1.0301, 2.4676, 3.8232, 8.4433, 24.4515, 35.2915],
            dtype=float,
        ),
        "sigma": np.array(
            [2732.0, 2011.0, 1178.0, 664.0, 92.74, 58.73, 25.58, 4.90, 1.732, 0.1807, 0.002509, 0.000394],
            dtype=float,
        ),
    }
    fig3_b21 = {
        "r_kpc": np.array([0.6535, 2.095, 3.832, 5.685, 10.05, 34.70, 86.45], dtype=float),
        "sigma": np.array([3.252, 0.7932, 0.4440, 0.1958, 0.04423, 0.001357, 0.0003268], dtype=float),
        "xerr": np.array([0.30, 0.85, 1.20, 1.60, 4.0, 16.0, 15.0], dtype=float),
        "yerr": np.array([0.90, 0.22, 0.12, 0.05, 0.015, 0.00045, 0.00012], dtype=float),
    }
    fig3_rbc = {
        "r_kpc": np.array([0.6487, 3.425, 9.264, 28.07, 87.29], dtype=float),
        "sigma": np.array([4.629, 0.8908, 0.2175, 0.00440, 0.0001504], dtype=float),
        "xerr": np.array([0.40, 1.50, 3.0, 16.0, 20.0], dtype=float),
        "yerr": np.array([1.5, 0.25, 0.07, 0.0018, 0.00008], dtype=float),
    }

    # Fig. 8: in-situ GC mass-function reference (Baumgardt+2021 trend).
    fig8_mass_obs = {
        "mass_msun": np.array([7.0e3, 1.5e4, 5.0e4, 1.0e5, 2.5e5, 8.0e5, 2.0e6, 4.0e6], dtype=float),
        "count": np.array([1.0, 2.0, 13.0, 25.0, 20.0, 6.5, 1.2, 0.2], dtype=float),
    }

    return {
        "fig2_ratio_refs": fig2_ratio_refs,
        "fig3_g14": fig3_g14,
        "fig3_b21": fig3_b21,
        "fig3_rbc": fig3_rbc,
        "fig8_mass_obs": fig8_mass_obs,
    }


def load_allcat(allcat_path: Path) -> pd.DataFrame:
    """Load and standardize one allcat table."""

    raw = pd.read_csv(
        allcat_path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )
    if raw.shape[1] < len(ALLCAT_COLUMNS):
        raise ValueError(
            f"Allcat file has {raw.shape[1]} columns; expected at least {len(ALLCAT_COLUMNS)}."
        )

    # Keep the canonical columns used by the plotting workflow. Newer outputs
    # may append extra formation-time diagnostics after these.
    # Keep the canonical 11 columns plus optional radius column when present.
    n_keep = min(raw.shape[1], len(ALLCAT_COLUMNS) + 1)
    raw = raw.iloc[:, :n_keep].copy()
    cols = list(ALLCAT_COLUMNS)
    if n_keep > len(ALLCAT_COLUMNS):
        cols.append(ALLCAT_OPTIONAL_RADIUS_COLUMN)
    raw.columns = cols

    for col in ALLCAT_COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if ALLCAT_OPTIONAL_RADIUS_COLUMN in raw.columns:
        raw[ALLCAT_OPTIONAL_RADIUS_COLUMN] = pd.to_numeric(
            raw[ALLCAT_OPTIONAL_RADIUS_COLUMN], errors="coerce"
        )

    gc = raw.dropna(subset=ALLCAT_COLUMNS).copy()
    if ALLCAT_OPTIONAL_RADIUS_COLUMN not in gc.columns:
        gc[ALLCAT_OPTIONAL_RADIUS_COLUMN] = np.nan

    gc["hid_z0"] = gc["hid_z0"].astype(int)
    gc["subfind_form"] = gc["subfind_form"].astype(int)
    gc["snap_form"] = gc["snap_form"].astype(int)
    gc["isMPB"] = gc["isMPB"].astype(int)
    gc["M_form"] = np.power(10.0, gc["logM_form"].to_numpy())
    gc["M_halo_z0"] = np.power(10.0, gc["logMh_z0"].to_numpy())
    gc["M_halo_form"] = np.power(10.0, gc["logMh_form"].to_numpy())
    return gc


def load_mpb(mpb_path: Path) -> pd.DataFrame:
    """Load MPB table (full or topology schema) and add helper columns."""

    mpb = pd.read_csv(mpb_path)
    for col in ["subhalo_id_z0", "SnapNum"]:
        if col not in mpb.columns:
            raise ValueError(f"MPB table is missing required column '{col}': {mpb_path}")
    mpb["subhalo_id_z0"] = pd.to_numeric(mpb["subhalo_id_z0"], errors="coerce").astype(int)
    mpb["SnapNum"] = pd.to_numeric(mpb["SnapNum"], errors="coerce").astype(int)

    # Support both full MPB timeseries and compact topology tables.
    if {"SubhaloSpin_x", "SubhaloSpin_y", "SubhaloSpin_z"}.issubset(mpb.columns):
        mpb["spin_mag"] = np.sqrt(
            np.square(pd.to_numeric(mpb["SubhaloSpin_x"], errors="coerce"))
            + np.square(pd.to_numeric(mpb["SubhaloSpin_y"], errors="coerce"))
            + np.square(pd.to_numeric(mpb["SubhaloSpin_z"], errors="coerce"))
        )
        mpb["spin_mag"] = np.where(np.isfinite(mpb["spin_mag"]), mpb["spin_mag"], 500.0)
    else:
        # Topology table omits spin vectors; use a neutral default.
        mpb["spin_mag"] = 500.0

    if "SubhaloMass" in mpb.columns:
        mpb["SubhaloMass"] = pd.to_numeric(mpb["SubhaloMass"], errors="coerce")
    if "logMh_msun_h" in mpb.columns:
        mpb["logMh_msun_h"] = pd.to_numeric(mpb["logMh_msun_h"], errors="coerce")
    if "Redshift" in mpb.columns:
        mpb["Redshift"] = pd.to_numeric(mpb["Redshift"], errors="coerce")
    return mpb


def load_inputs(allcat_path: Path, mpb_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and standardize GC catalog and MPB tables."""

    gc = load_allcat(allcat_path)
    mpb = load_mpb(mpb_path)
    return gc, mpb


def build_snap_to_z_map(gc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build a snap->redshift interpolation from formed GC rows."""

    snap_z = gc.groupby("snap_form")["zform"].median().sort_index()
    snap_arr = snap_z.index.to_numpy(dtype=float)
    z_arr = snap_z.to_numpy(dtype=float)
    return snap_arr, z_arr


def estimate_zhm(gc: pd.DataFrame, mpb: pd.DataFrame, *, final_redshift: float = 0.0) -> pd.DataFrame:
    """Estimate z_hm for each target halo from MPB and snap->z interpolation."""

    snap_arr, z_arr = build_snap_to_z_map(gc)
    rows: List[dict] = []

    for hid, grp in mpb.groupby("subhalo_id_z0", sort=True):
        g = grp.sort_values("SnapNum", ascending=False)
        if "Redshift" in g.columns:
            z_hist = pd.to_numeric(g["Redshift"], errors="coerce").to_numpy(dtype=float)
        else:
            z_hist = np.interp(
                g["SnapNum"].to_numpy(dtype=float),
                snap_arr,
                z_arr,
                left=z_arr[0],
                right=z_arr[-1],
            )

        if "SubhaloMass" in g.columns:
            mass_msun_h = pd.to_numeric(g["SubhaloMass"], errors="coerce").to_numpy(dtype=float) * 1e10
        elif "logMh_msun_h" in g.columns:
            logm = pd.to_numeric(g["logMh_msun_h"], errors="coerce").to_numpy(dtype=float)
            mass_msun_h = np.power(10.0, logm)
        else:
            raise ValueError("MPB table must provide either 'SubhaloMass' or 'logMh_msun_h'.")

        valid = np.isfinite(mass_msun_h) & (mass_msun_h > 0) & np.isfinite(z_hist)
        if np.any(valid):
            valid_idx = np.where(valid)[0]
            usable_final = valid_idx[z_hist[valid_idx] >= (final_redshift - 1.0e-10)]
            if len(usable_final) > 0:
                idx_final = int(usable_final[np.argmin(z_hist[usable_final])])
            else:
                idx_final = int(valid_idx[-1])
            m0 = float(mass_msun_h[idx_final])
            half = 0.5 * m0
            # Restrict the assembly history to the portion that has already
            # happened by `final_redshift`.
            hist_idx = valid_idx[valid_idx >= idx_final]
            crossed = hist_idx[mass_msun_h[hist_idx] <= half]
            hm_idx = int(crossed[0]) if len(crossed) > 0 else int(hist_idx[-1])
            snap_hm = int(g["SnapNum"].iloc[hm_idx])
            m_hm = float(mass_msun_h[hm_idx])
            m_halo_z0_mpb = m0
        else:
            # Rare fallback for rows with missing MPB mass history.
            sel_gc = gc["hid_z0"].to_numpy() == int(hid)
            if np.any(sel_gc):
                m_halo_z0_mpb = float(np.power(10.0, gc.loc[sel_gc, "logMh_z0"].iloc[0]))
            else:
                m_halo_z0_mpb = np.nan
            snap_hm = int(g["SnapNum"].min())
            m_hm = m_halo_z0_mpb
            idx_final = 0
            hm_idx = 0

        if np.any(valid):
            z_hm = float(z_hist[hm_idx])
        else:
            z_hm = float(np.interp(snap_hm, snap_arr, z_arr, left=z_arr[0], right=z_arr[-1]))
        spin_mag = float(g["spin_mag"].iloc[idx_final]) if "spin_mag" in g.columns else 500.0
        if not np.isfinite(spin_mag):
            spin_mag = 500.0

        rows.append(
            {
                "hid_z0": int(hid),
                "z_hm": z_hm,
                "snap_hm": snap_hm,
                "snap_final": int(g["SnapNum"].iloc[idx_final]),
                "z_final_used": float(z_hist[idx_final]) if np.any(valid) else np.nan,
                "M_halo_hm": m_hm,
                "spin_mag": spin_mag,
                "M_halo_z0_mpb": m_halo_z0_mpb,
                "M_halo_final_mpb": m_halo_z0_mpb,
            }
        )

    halo_meta = pd.DataFrame(rows).set_index("hid_z0").sort_index()
    return halo_meta


def lookback_time_gyr(z: np.ndarray) -> np.ndarray:
    """Approximate lookback time in Gyr.

    Uses Astropy when available; otherwise falls back to a smooth analytic
    approximation that is accurate enough for the plotting workflow here.
    """

    z = np.asarray(z, dtype=float)
    try:
        from astropy.cosmology import Planck18  # type: ignore

        return Planck18.lookback_time(z).value
    except Exception:
        return 13.8 * (1.0 - 1.0 / np.sqrt(1.0 + np.clip(z, 0.0, None)))


def _ns_tag(ns_value: float) -> str:
    """Convert N_s value into filename tag, e.g. 0.5 -> '0p5'."""

    return f"{float(ns_value):.1f}".replace(".", "p")


def _model_output_root_from_allcat_path(allcat_path: Path) -> Path:
    """Infer the Gao output root from either a root template or an ns subdir file."""

    parent = allcat_path.parent
    if re.fullmatch(r"ns[0-9]+p[0-9]+", parent.name):
        return parent.parent
    return parent


def _load_run_metadata(allcat_path: Path) -> Dict[str, object]:
    """Load run metadata emitted by `my/run.py`, if present."""

    path = _model_output_root_from_allcat_path(allcat_path) / RUN_METADATA_NAME
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_ns_allcat_path(allcat_template_path: Path, ns_value: float) -> Path:
    """Build allcat path for one N_s from a template/root allcat filename."""

    model_output_root = _model_output_root_from_allcat_path(allcat_template_path)
    name = allcat_template_path.name
    m = re.match(r"^(?P<prefix>.+?)(?P<suffix>_s-.*\.txt)$", name)
    if m is None:
        raise ValueError(
            "Cannot infer N_s allcat filenames from template. "
            f"Expected name like 'allcat_s-...txt' or 'allcat_nsXpY_s-...txt', got: {name}"
        )

    prefix = re.sub(r"_ns[0-9p]+$", "", m.group("prefix"))
    suffix = m.group("suffix")
    ns_tag = _ns_tag(ns_value)
    return model_output_root / f"ns{ns_tag}" / f"{prefix}_ns{ns_tag}{suffix}"


def _resolve_reference_allcat_path(allcat_template_path: Path, ns_values: Sequence[float]) -> Path:
    """Pick one existing allcat path used as reference row ordering."""

    candidates = [allcat_template_path]
    for ns in ns_values:
        try:
            candidates.append(_build_ns_allcat_path(allcat_template_path, float(ns)))
        except ValueError:
            continue
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Cannot find any allcat file for plotting. Checked:\n- "
        + "\n- ".join(str(p) for p in candidates)
    )


def _find_final_gcs_file(allcat_ns_path: Path) -> Path:
    """Return the published merged final-GC file for one `N_s`."""

    m = re.search(r"_ns([0-9]+p[0-9]+)", allcat_ns_path.stem)
    if m is None:
        raise ValueError(f"Could not infer N_s tag from {allcat_ns_path.name}")
    ns_tag = m.group(1)
    path = allcat_ns_path.parent / f"finalGCs_ns{ns_tag}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing finalGCs file for {allcat_ns_path.name}: {path}")
    return path


def _find_halo_summary_file(allcat_ns_path: Path) -> Path | None:
    """Return the published halo-summary file for one `N_s`, if present."""

    m = re.search(r"_ns([0-9]+p[0-9]+)", allcat_ns_path.stem)
    if m is None:
        raise ValueError(f"Could not infer N_s tag from {allcat_ns_path.name}")
    ns_tag = m.group(1)
    path = allcat_ns_path.parent / f"haloSummary_ns{ns_tag}.csv"
    return path if path.exists() else None


def _read_comment_columns(path: Path) -> List[str]:
    """Return whitespace-delimited column names from the first header line."""

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                continue
            text = line[1:].strip()
            if not text:
                continue
            return text.split()
    raise ValueError(f"Cannot find header columns in {path}.")


def _load_final_gcs_table(
    path: Path,
    expected_len: int,
    expected_halo_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load final GC masses/radii from the published merged finalGCs table."""

    columns = _read_comment_columns(path)
    col_index = {name: idx for idx, name in enumerate(columns)}
    for required in ["halo_id_z0", "gc_index_halo", "m_final_msun", "r_final_kpc"]:
        if required not in col_index:
            raise ValueError(f"Missing required column '{required}' in {path}.")

    arr = np.asarray(np.loadtxt(path, ndmin=2), dtype=float)
    if len(arr) != expected_len:
        raise ValueError(
            f"Length mismatch for finalGCs table: {path} has {len(arr)} rows, expected {expected_len}."
        )

    halo_ids = np.asarray(arr[:, col_index["halo_id_z0"]], dtype=int)
    if not np.array_equal(halo_ids, np.asarray(expected_halo_ids, dtype=int)):
        raise ValueError(
            f"Row-order mismatch between {path} and the matching allcat_ns file "
            "when comparing halo_id_z0."
        )

    expected_gc_index = np.empty(expected_len, dtype=int)
    for hid in np.unique(expected_halo_ids):
        idx = np.where(np.asarray(expected_halo_ids, dtype=int) == int(hid))[0]
        # Within each halo the merged finalGCs table preserves the local
        # 1-based GC numbering used by the per-halo evolution outputs.
        expected_gc_index[idx] = np.arange(1, len(idx) + 1, dtype=int)
    gc_index_halo = np.asarray(arr[:, col_index["gc_index_halo"]], dtype=int)
    if not np.array_equal(gc_index_halo, expected_gc_index):
        raise ValueError(
            f"Row-order mismatch between {path} and the matching allcat_ns file "
            "when comparing gc_index_halo."
        )

    m_final = np.asarray(arr[:, col_index["m_final_msun"]], dtype=float)
    r_final = np.asarray(arr[:, col_index["r_final_kpc"]], dtype=float)
    m_final = np.where(np.isfinite(m_final) & (m_final > 0), m_final, 0.0)
    r_final = np.where(np.isfinite(r_final) & (r_final > 0), r_final, np.nan)
    return m_final, r_final


def _load_halo_summary(path: Path) -> pd.DataFrame:
    """Load one per-N_s halo summary table."""

    df = pd.read_csv(path)
    required = ["hid_z0", "m_imbh_seed_total_msun", "m_smbh_est_msun", "n_sunk"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}.")
    df["hid_z0"] = pd.to_numeric(df["hid_z0"], errors="coerce").astype(int)
    for col in df.columns:
        if col == "hid_z0":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("hid_z0").reset_index(drop=True)


def _build_halo_summary_from_final_gcs(
    path: Path,
    gc_ns: pd.DataFrame,
    expected_halo_ids: np.ndarray,
) -> pd.DataFrame:
    """Reconstruct the halo summary when a published CSV is absent.

    Older Gao output trees may have the merged ``finalGCs_ns*.dat`` tables but
    not the later ``haloSummary_ns*.csv`` files. The BH statistics needed by
    Figures 10 and 11 can still be derived exactly from the per-GC final table
    because it already stores the halo id, status code, initial/final GC mass,
    and the IMBH seed mass for each formed cluster.
    """

    columns = _read_comment_columns(path)
    col_index = {name: idx for idx, name in enumerate(columns)}
    required = [
        "halo_id_z0",
        "status",
        "m_final_msun",
        "m_init_msun",
        "imbh_mass_msun",
    ]
    for required_col in required:
        if required_col not in col_index:
            raise ValueError(f"Missing required column '{required_col}' in {path}.")

    arr = np.asarray(np.loadtxt(path, ndmin=2), dtype=float)
    if len(arr) != len(expected_halo_ids):
        raise ValueError(
            f"Length mismatch for finalGCs table: {path} has {len(arr)} rows, "
            f"expected {len(expected_halo_ids)}."
        )

    halo_ids = np.asarray(arr[:, col_index["halo_id_z0"]], dtype=int)
    if not np.array_equal(halo_ids, np.asarray(expected_halo_ids, dtype=int)):
        raise ValueError(
            f"Row-order mismatch between {path} and the matching allcat_ns file "
            "when rebuilding haloSummary."
        )

    status = np.asarray(arr[:, col_index["status"]], dtype=int)
    m_final = np.asarray(arr[:, col_index["m_final_msun"]], dtype=float)
    m_final = np.where(np.isfinite(m_final) & (m_final > 0.0), m_final, 0.0)
    m_init = np.asarray(arr[:, col_index["m_init_msun"]], dtype=float)
    m_init = np.where(np.isfinite(m_init) & (m_init > 0.0), m_init, 0.0)
    imbh = np.asarray(arr[:, col_index["imbh_mass_msun"]], dtype=float)
    imbh = np.where(np.isfinite(imbh) & (imbh > 0.0), imbh, 0.0)

    gc_tmp = gc_ns[["hid_z0", "logMh_z0"]].copy()
    gc_tmp["status"] = status
    gc_tmp["m_init_msun"] = m_init
    gc_tmp["m_final_msun"] = m_final
    gc_tmp["imbh_mass_msun"] = imbh

    rows: List[dict] = []
    for hid, grp in gc_tmp.groupby("hid_z0", sort=True):
        s = grp["status"].to_numpy(dtype=int)
        seed_mass = grp["imbh_mass_msun"].to_numpy(dtype=float)
        rows.append(
            {
                "hid_z0": int(hid),
                "logMh_z0": float(grp["logMh_z0"].iloc[0]),
                "n_gc_total": int(len(grp)),
                "n_alive": int(np.sum(s == 1)),
                "n_exhausted": int(np.sum(s == -1)),
                "n_torn": int(np.sum(s == -2)),
                "n_sunk": int(np.sum(s == -3)),
                "m_gc_init_total_msun": float(grp["m_init_msun"].sum()),
                "m_gc_final_total_msun": float(grp["m_final_msun"].sum()),
                "m_imbh_seed_total_msun": float(seed_mass.sum()),
                "m_smbh_est_msun": float(seed_mass[s == -3].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("hid_z0").reset_index(drop=True)


def _find_deposit_file(allcat_ns_path: Path) -> Path | None:
    """Return the published merged deposit file for one `N_s`."""

    m = re.search(r"_ns([0-9]+p[0-9]+)", allcat_ns_path.stem)
    if m is None:
        return None
    ns_tag = m.group(1)
    path = allcat_ns_path.parent / f"depos_ns{ns_tag}.dat"
    return path if path.exists() else None


def _load_deposit_profile(allcat_ns_path: Path) -> DepositProfile | None:
    """Build a deposited-profile table from the merged per-`N_s` deposit file."""

    path = _find_deposit_file(allcat_ns_path)
    if path is None:
        return None

    arr = np.asarray(np.loadtxt(path, ndmin=2), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Unexpected combined deposit-file shape in {path}: {arr.shape}")

    halo_ids: List[int] = []
    r_inner_rows: List[np.ndarray] = []
    r_outer_rows: List[np.ndarray] = []
    shell_rows: List[np.ndarray] = []
    cum_rows: List[np.ndarray] = []

    ordered_halos = [int(h) for h in pd.unique(arr[:, 0].astype(int))]
    for hid in ordered_halos:
        halo_block = arr[arr[:, 0].astype(int) == hid]
        if len(halo_block) == 0:
            continue
        last_time = float(halo_block[-1, 1])
        # Each deposit file stores one full radial profile per coarse time
        # block. For the figure suite we want only the final z=0 profile.
        block = halo_block[np.isclose(halo_block[:, 1], last_time)]
        if len(block) == 0:
            raise ValueError(f"Cannot find final-time deposit block in {path} for halo {hid}")
        order = np.argsort(block[:, 2])
        block = block[order]

        halo_ids.append(hid)
        r_inner_rows.append(np.asarray(block[:, 3], dtype=float))
        r_outer_rows.append(np.asarray(block[:, 4], dtype=float))
        shell = np.asarray(block[:, 5], dtype=float)
        shell_rows.append(shell)
        cum_rows.append(np.cumsum(shell))

    if not halo_ids:
        return None

    return DepositProfile(
        halo_ids=np.asarray(halo_ids, dtype=int),
        r_inner_kpc=r_inner_rows,
        r_outer_kpc=r_outer_rows,
        shell_mass_msun=shell_rows,
        cumulative_mass_msun=cum_rows,
    )


def _assert_same_row_order(gc_ref: pd.DataFrame, gc_ns: pd.DataFrame, ns_path: Path) -> None:
    """Ensure per-Ns catalog rows align with the reference catalog."""

    if len(gc_ref) != len(gc_ns):
        raise ValueError(
            f"Row-count mismatch: {ns_path} has {len(gc_ns)} rows, "
            f"reference catalog has {len(gc_ref)} rows."
        )

    key_cols = ["hid_z0", "subfind_form", "snap_form", "isMPB"]
    for col in key_cols:
        if not np.array_equal(gc_ref[col].to_numpy(), gc_ns[col].to_numpy()):
            raise ValueError(
                f"Row-order mismatch in column '{col}' for {ns_path}. "
                "Use outputs generated from the same run setup/subhalo list."
            )


def simulate_models(
    gc: pd.DataFrame,
    allcat_template_path: Path,
    ns_values: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    seed: int = 7,
) -> Dict[float, ModelResult]:
    """Load initial/final GC states for each N_s directly from simulation outputs."""

    del seed  # Kept for API compatibility; no stochastic model is used here.

    results: Dict[float, ModelResult] = {}
    n_tot = len(gc)

    for ns in ns_values:
        ns_key = float(ns)
        allcat_ns_path = _build_ns_allcat_path(allcat_template_path, ns_key)
        if not allcat_ns_path.exists():
            raise FileNotFoundError(f"Missing N_s allcat file: {allcat_ns_path}")

        gc_ns = load_allcat(allcat_ns_path)
        # Later figure panels compare different N_s values GC-by-GC, so the
        # catalogs must be in identical row order before we trust those joins.
        _assert_same_row_order(gc_ref=gc, gc_ns=gc_ns, ns_path=allcat_ns_path)

        r_init = np.asarray(gc_ns[ALLCAT_OPTIONAL_RADIUS_COLUMN], dtype=float)
        r_init = np.where(np.isfinite(r_init) & (r_init > 0), r_init, np.nan)

        final_gcs_path = _find_final_gcs_file(allcat_ns_path)
        m_final, r_final = _load_final_gcs_table(
            final_gcs_path,
            expected_len=n_tot,
            expected_halo_ids=np.asarray(gc_ns["hid_z0"], dtype=int),
        )
        halo_summary_path = _find_halo_summary_file(allcat_ns_path)
        if halo_summary_path is not None:
            halo_summary = _load_halo_summary(halo_summary_path)
        else:
            halo_summary = _build_halo_summary_from_final_gcs(
                final_gcs_path,
                gc_ns=gc_ns,
                expected_halo_ids=np.asarray(gc_ns["hid_z0"], dtype=int),
            )

        deposit_profile = _load_deposit_profile(allcat_ns_path)

        results[ns_key] = ModelResult(
            ns_value=ns_key,
            r_init=r_init,
            r_final=r_final,
            m_final=m_final,
            deposit_profile=deposit_profile,
            halo_summary=halo_summary,
        )

    return results


def _surface_density_mean_by_halo(
    halo_ids: np.ndarray,
    radii: np.ndarray,
    bins: np.ndarray,
    *,
    extra_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean surface density profile over halos, including empty-bin zeros."""

    halo_ids = np.asarray(halo_ids)
    radii = np.asarray(radii, dtype=float)
    if extra_mask is None:
        valid = np.isfinite(radii) & (radii > 0)
    else:
        valid = np.asarray(extra_mask, dtype=bool) & np.isfinite(radii) & (radii > 0)

    centers = np.sqrt(bins[:-1] * bins[1:])
    unique_halos = np.unique(halo_ids)
    if len(unique_halos) == 0:
        return centers, np.full(len(centers), np.nan, dtype=float)

    area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    prof = np.zeros((len(unique_halos), len(centers)), dtype=float)
    for ii, hid in enumerate(unique_halos):
        counts, _ = np.histogram(radii[(halo_ids == hid) & valid], bins=bins)
        prof[ii] = counts / np.clip(area, 1e-20, None)

    density = np.mean(prof, axis=0)
    density = np.where(density > 0, density, np.nan)
    return centers, density


def _final_survivor_mask(model: ModelResult, extra_mask: np.ndarray | None = None) -> np.ndarray:
    """Return mask selecting only surviving GCs for final-state profiles."""

    mask = np.asarray(model.m_final, dtype=float) > 0
    if extra_mask is not None:
        mask &= np.asarray(extra_mask, dtype=bool)
    return mask


def _mass_histograms_by_halo(
    halo_ids: np.ndarray,
    masses: np.ndarray,
    bins: np.ndarray,
    *,
    extra_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-halo mass histograms on a common binning."""

    halo_ids = np.asarray(halo_ids, dtype=int)
    masses = np.asarray(masses, dtype=float)
    if extra_mask is None:
        valid = np.isfinite(masses) & (masses > 0)
    else:
        valid = np.asarray(extra_mask, dtype=bool) & np.isfinite(masses) & (masses > 0)

    centers = np.sqrt(bins[:-1] * bins[1:])
    unique_halos = np.unique(halo_ids)
    out = np.zeros((len(unique_halos), len(centers)), dtype=float)
    for ii, hid in enumerate(unique_halos):
        counts, _ = np.histogram(masses[(halo_ids == hid) & valid], bins=bins)
        out[ii] = counts
    return centers, out


def _cumulative_profile(radii: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Cumulative sum as a function of radius."""

    if len(radii) == 0 or len(values) == 0:
        return np.zeros_like(grid, dtype=float)

    radii = np.asarray(radii, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(radii) & np.isfinite(values) & (radii > 0) & (values > 0)
    if not np.any(valid):
        return np.zeros_like(grid, dtype=float)

    order = np.argsort(radii[valid])
    r = radii[valid][order]
    v = values[valid][order]
    c = np.cumsum(v)
    return np.interp(grid, r, c, left=c[0] if len(c) else 0.0, right=c[-1] if len(c) else 0.0)


def _cumulative_mean_by_halo(
    halo_ids: np.ndarray,
    radii: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    *,
    extra_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Average cumulative radial profile over halos."""

    halo_ids = np.asarray(halo_ids, dtype=int)
    radii = np.asarray(radii, dtype=float)
    values = np.asarray(values, dtype=float)
    if extra_mask is None:
        base_mask = np.ones(len(halo_ids), dtype=bool)
    else:
        base_mask = np.asarray(extra_mask, dtype=bool)

    unique_halos = np.unique(halo_ids[base_mask])
    if len(unique_halos) == 0:
        return np.full(len(grid), np.nan, dtype=float)

    prof = np.zeros((len(unique_halos), len(grid)), dtype=float)
    for ii, hid in enumerate(unique_halos):
        hmask = base_mask & (halo_ids == hid)
        prof[ii] = _cumulative_profile(radii[hmask], values[hmask], grid)

    return np.mean(prof, axis=0)


def _deposit_mean_profile(
    profile: DepositProfile,
    *,
    grid_kpc: np.ndarray | None = None,
    halo_ids: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average deposited cumulative profile for a halo subset.

    Per-halo deposit files carry halo-specific radial bins, so each halo is
    interpolated onto a shared grid before averaging.
    """

    if halo_ids is None:
        use = np.ones(len(profile.halo_ids), dtype=bool)
    else:
        use = np.isin(profile.halo_ids, np.asarray(halo_ids, dtype=int))
    idx = np.where(use)[0]
    if len(idx) == 0:
        grid = np.asarray(grid_kpc if grid_kpc is not None else np.array([1.0e-3]), dtype=float)
        return grid, np.full(len(grid), np.nan, dtype=float)

    if grid_kpc is None:
        r_min = min(max(float(profile.r_outer_kpc[ii][0]), 1.0e-6) for ii in idx)
        r_max = max(float(profile.r_outer_kpc[ii][-1]) for ii in idx)
        grid = np.logspace(np.log10(r_min), np.log10(r_max), 256)
    else:
        grid = np.asarray(grid_kpc, dtype=float)

    prof = np.zeros((len(idx), len(grid)), dtype=float)
    for jj, ii in enumerate(idx):
        radii = np.asarray(profile.r_outer_kpc[ii], dtype=float)
        cum = np.asarray(profile.cumulative_mass_msun[ii], dtype=float)
        # Deposit tables are already cumulative in radius, so only a 1D radial
        # interpolation is needed before taking the halo-average profile.
        prof[jj] = np.interp(grid, radii, cum, left=0.0, right=cum[-1])
    return grid, np.mean(prof, axis=0)


def _deposit_mass_within_radius(
    profile: DepositProfile,
    radius_kpc: float,
    *,
    halo_ids: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deposited cumulative mass evaluated at `radius_kpc` for selected halos."""

    if halo_ids is None:
        use = np.ones(len(profile.halo_ids), dtype=bool)
    else:
        use = np.isin(profile.halo_ids, np.asarray(halo_ids, dtype=int))
    halo_use = profile.halo_ids[use]
    if len(halo_use) == 0:
        return halo_use, np.array([], dtype=float)

    vals = np.zeros(len(halo_use), dtype=float)
    use_idx = np.where(use)[0]
    for jj, ii in enumerate(use_idx):
        radii = np.asarray(profile.r_outer_kpc[ii], dtype=float)
        cum = np.asarray(profile.cumulative_mass_msun[ii], dtype=float)
        vals[jj] = float(np.interp(radius_kpc, radii, cum, left=0.0, right=cum[-1]))
    return halo_use, vals


def _select_halos_by_logmh(
    gc: pd.DataFrame,
    logmh_min: float,
    logmh_max: float,
    *,
    fallback_n: int = 3,
) -> np.ndarray:
    """Select halo IDs in a mass window, with nearest-mass fallback.

    Small demo subsets may not contain halos in the exact Gao+2023 mass
    windows; when that happens, use the nearest available halo masses so
    downstream figures remain reproducible.
    """

    halo_mass = gc.groupby("hid_z0", sort=True)["logMh_z0"].first()
    mask = (halo_mass >= logmh_min) & (halo_mass < logmh_max)
    selected = halo_mass.index[mask].to_numpy(dtype=int)
    if len(selected) > 0:
        return selected

    center = 0.5 * (logmh_min + logmh_max)
    nearest = (halo_mass - center).abs().sort_values().index.to_numpy(dtype=int)
    n_keep = min(max(1, int(fallback_n)), len(nearest))
    return nearest[:n_keep]


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient for finite inputs."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _fit_band(
    x: np.ndarray,
    y: np.ndarray,
    *,
    logx: bool = False,
    logy: bool = False,
    n_grid: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best-fit line with a symmetric 1-sigma band in transformed space."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if logx:
        mask &= x > 0
    if logy:
        mask &= y > 0
    if np.sum(mask) < 2:
        xx = np.full(n_grid, np.nan, dtype=float)
        return xx, np.full_like(xx, np.nan), np.full_like(xx, np.nan), np.full_like(xx, np.nan)

    xt = np.log10(x[mask]) if logx else x[mask]
    yt = np.log10(y[mask]) if logy else y[mask]

    aa, bb = np.polyfit(xt, yt, 1)
    xx_t = np.linspace(np.nanmin(xt), np.nanmax(xt), n_grid)
    yy_t = aa * xx_t + bb
    resid = yt - (aa * xt + bb)
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else 0.0

    xx = np.power(10.0, xx_t) if logx else xx_t
    yy = np.power(10.0, yy_t) if logy else yy_t
    lo = np.power(10.0, yy_t - sigma) if logy else yy_t - sigma
    hi = np.power(10.0, yy_t + sigma) if logy else yy_t + sigma
    return xx, yy, lo, hi


def _mean_and_std(values: np.ndarray) -> Tuple[float, float]:
    """Mean and sample scatter for finite inputs."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def _build_discrete_ns_style(
    ns_values: Sequence[float],
) -> Tuple[np.ndarray, mpl.colors.ListedColormap, mpl.colors.BoundaryNorm, np.ndarray, Dict[float, np.ndarray]]:
    """Return a discrete N_s colormap setup matching the paper-style bars."""

    ns_levels = np.asarray([float(v) for v in ns_values], dtype=float)
    if ns_levels.ndim != 1 or len(ns_levels) == 0:
        raise ValueError("Need at least one N_s value to build the colorbar.")

    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(ns_levels)))
    cmap = mpl.colors.ListedColormap(colors)
    if len(ns_levels) == 1:
        boundaries = np.array([ns_levels[0] - 0.5, ns_levels[0] + 0.5], dtype=float)
    else:
        mid = 0.5 * (ns_levels[:-1] + ns_levels[1:])
        first = ns_levels[0] - 0.5 * (ns_levels[1] - ns_levels[0])
        last = ns_levels[-1] + 0.5 * (ns_levels[-1] - ns_levels[-2])
        boundaries = np.concatenate(([first], mid, [last]))
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
    color_lookup = {float(ns): colors[ii] for ii, ns in enumerate(ns_levels)}
    return ns_levels, cmap, norm, boundaries, color_lookup


def _add_discrete_ns_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    ns_levels: np.ndarray,
    cmap: mpl.colors.ListedColormap,
    norm: mpl.colors.BoundaryNorm,
    boundaries: np.ndarray,
) -> mpl.colorbar.Colorbar:
    """Add a discrete N_s colorbar with labels only and minimal gap."""

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        boundaries=boundaries,
        ticks=ns_levels,
        spacing="proportional",
        drawedges=False,
        pad=0.0,
        fraction=0.05,
    )
    cbar.set_label(r"$N_{\rm S}$")
    cbar.set_ticklabels([f"{ns:.1f}" for ns in ns_levels])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(length=0, width=0, pad=1.5)
    cbar.outline.set_linewidth(0.5) # shrink border thickness
    return cbar


def _halo_level_table(
    gc: pd.DataFrame,
    halo_meta: pd.DataFrame,
    model: ModelResult,
    *,
    nsc_radius_kpc: float | None = None,
) -> pd.DataFrame:
    """Build one halo-level summary table used by correlation plots."""

    temp = gc[["hid_z0", "logMh_z0"]].copy()
    temp["M_form"] = gc["M_form"].to_numpy()
    temp["M_final"] = model.m_final
    temp["r_final"] = model.r_final

    rows = []
    for hid, g in temp.groupby("hid_z0", sort=True):
        rows.append(
            {
                "hid_z0": int(hid),
                "M_halo": float(np.power(10.0, g["logMh_z0"].iloc[0])),
                "M_gc_init": float(g["M_form"].sum()),
                "M_gc_final": float(g["M_final"].sum()),
                "M_nsc": 0.0,
            }
        )

    out = pd.DataFrame(rows).set_index("hid_z0").sort_index()
    if model.deposit_profile is not None and nsc_radius_kpc is not None:
        # Preferred NSC proxy: deposited mass within the adopted NSC radius.
        halo_ids_dep, m_dep = _deposit_mass_within_radius(
            model.deposit_profile,
            float(nsc_radius_kpc),
        )
        if len(halo_ids_dep) > 0:
            out.loc[halo_ids_dep, "M_nsc"] = m_dep
    else:
        # Fallback for older outputs without deposit tables: use surviving GC
        # mass inside a fixed central aperture.
        out["M_nsc"] = [
            float(temp.loc[(temp["hid_z0"] == hid) & (temp["r_final"] <= 0.3), "M_final"].sum())
            for hid in out.index.to_numpy(dtype=int)
        ]
    if model.halo_summary is not None and len(model.halo_summary) > 0:
        hs = model.halo_summary[
            ["hid_z0", "m_imbh_seed_total_msun", "m_smbh_est_msun", "n_sunk"]
        ].copy()
        hs["hid_z0"] = hs["hid_z0"].astype(int)
        hs = hs.set_index("hid_z0").sort_index()
        out = out.join(
            hs.rename(
                columns={
                    "m_imbh_seed_total_msun": "M_bh_total",
                    "m_smbh_est_msun": "M_smbh",
                }
            ),
            how="left",
        )
    else:
        out["M_bh_total"] = 0.0
        out["M_smbh"] = 0.0
        out["n_sunk"] = 0
    out["M_bh_total"] = pd.to_numeric(out["M_bh_total"], errors="coerce").fillna(0.0)
    out["M_smbh"] = pd.to_numeric(out["M_smbh"], errors="coerce").fillna(0.0)
    out["n_sunk"] = pd.to_numeric(out["n_sunk"], errors="coerce").fillna(0).astype(int)
    out = out.join(halo_meta[["z_hm"]], how="left")
    out["logM_halo"] = _safe_log10(out["M_halo"].to_numpy())
    out["logM_gc_init"] = _safe_log10(out["M_gc_init"].to_numpy())
    out["logM_gc_final"] = _safe_log10(out["M_gc_final"].to_numpy())
    out["logM_nsc"] = _safe_log10(np.clip(out["M_nsc"].to_numpy(), 1.0, None))
    out["logM_bh_total"] = _safe_log10(np.clip(out["M_bh_total"].to_numpy(), 1.0, None))
    out["logM_smbh"] = _safe_log10(np.clip(out["M_smbh"].to_numpy(), 1.0, None))
    return out


def build_reproduction(
    allcat_path: Path,
    mpb_path: Path,
    output_dir: Path,
    ns_values: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    seed: int = 7,
    include_observables: bool = True,
    final_redshift: float | None = None,
) -> pd.DataFrame:
    """Main reproduction entry point.

    Returns
    -------
    manifest:
        DataFrame with figure number and absolute output path.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_plot_settings_from_data()
    ns_values = [float(v) for v in ns_values]
    run_meta = _load_run_metadata(allcat_path)
    if final_redshift is None:
        final_redshift = float(run_meta.get("final_redshift", 0.0))
    else:
        final_redshift = float(final_redshift)

    allcat_ref_path = _resolve_reference_allcat_path(allcat_path, ns_values)
    gc, mpb = load_inputs(allcat_path=allcat_ref_path, mpb_path=mpb_path)
    halo_meta = estimate_zhm(gc=gc, mpb=mpb, final_redshift=final_redshift)
    gc = gc.join(halo_meta[["z_hm"]], on="hid_z0")
    mw_obs, m31_obs = _get_mw_m31_observations()
    obs_overlay = _load_observational_overlays() if include_observables else {}
    if include_observables and final_redshift > 1.0e-12:
        print(
            "WARNING observational overlays are z=0 references while this run "
            f"ends at final_redshift={final_redshift:g}"
        )

    models = simulate_models(
        gc=gc,
        allcat_template_path=allcat_path,
        ns_values=ns_values,
        seed=seed,
    )

    # Common bins/grids used by multiple figures.
    r_bins = np.logspace(-2.0, 2.0, 24)

    q1, q2 = halo_meta["z_hm"].quantile([1 / 3, 2 / 3]).to_numpy()
    halo_meta = halo_meta.copy()
    # Several figure panels split halos into low/mid/high assembly bins.
    halo_meta["zhm_bin"] = np.where(
        halo_meta["z_hm"] <= q1,
        "low",
        np.where(halo_meta["z_hm"] <= q2, "mid", "high"),
    )

    manifest_rows: List[dict] = []

    def save(fig_num: int, stem: str) -> Path:
        path = output_dir / f"Fig.{fig_num:02d}_{stem}.png"
        plt.savefig(path, dpi=STD_DPI, bbox_inches="tight")
        manifest_rows.append(
            {
                "figure": int(fig_num),
                "path": str(path),
                "observables": bool(include_observables),
            }
        )
        plt.close()
        return path

    # Figure 2: final M_GC/M_halo vs Ns with median and 25--75 percentile spread.
    x = np.linspace(0.0, 7.0, 500)
    y = []
    yerr_low = []
    yerr_high = []
    ns_values = np.array(ns_values, dtype=float)
    for ns in ns_values:
        halo_table = _halo_level_table(
            gc=gc,
            halo_meta=halo_meta,
            model=models[float(ns)],
        )
        halo_ratio = (
            halo_table["M_gc_final"].to_numpy(dtype=float)
            / np.clip(halo_table["M_halo"].to_numpy(dtype=float), 1e-30, None)
        )
        y_med = float(np.median(halo_ratio))
        q25, q75 = np.quantile(halo_ratio, [0.25, 0.75])
        y.append(y_med / 1.0e-5)
        yerr_low.append(max((y_med - float(q25)) / 1.0e-5, 0.0))
        yerr_high.append(max((float(q75) - y_med) / 1.0e-5, 0.0))
    plt.figure(constrained_layout=True, dpi=STD_DPI, figsize=(4.0, 3.0))
    plt.errorbar(
        ns_values,
        y,
        yerr=[yerr_low, yerr_high],
        marker="D",
        color="black",
        mfc="black",
        mec="black",
        capsize=4,
        lw=1.0,
    )
    if include_observables:
        ref_colors = {"S09": "blue", "G10": "red", "H14": "c", "H17": "green"}
        for label, ratio in obs_overlay["fig2_ratio_refs"].items():
            plt.axhline(
                ratio / 1.0e-5,
                lw=1.0,
                ls="--",
                alpha=0.9,
                color=ref_colors.get(label, "gray"),
                label=label,
            )
    plt.xlabel(r"$N_{\rm S}$")
    plt.ylabel(r"$M_{\rm GC} \, / \, M_{\rm halo} \times 10^{-5}$")
    plt.xlim(0.0, 4.5)
    plt.ylim(1.0, 9.0)
    plt.xticks([0, 1, 2, 3, 4])
    if include_observables:
        plt.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    save(2, "mgc_mhalo_ratio")

    ns_levels, ns_cmap, ns_norm, ns_boundaries, ns_color_lookup = _build_discrete_ns_style(ns_values)

    # Figure 3: global radial number-density profile, initial vs final.
    fig, ax = plt.subplots(constrained_layout=True, dpi=STD_DPI, figsize=(4.2, 3.0))
    halo_ids = gc["hid_z0"].to_numpy(dtype=int)
    for i_ns, ns in enumerate(ns_values):
        model = models[float(ns)]
        c0, d0 = _surface_density_mean_by_halo(halo_ids, model.r_init, r_bins)
        c1, d1 = _surface_density_mean_by_halo(
            halo_ids,
            model.r_final,
            r_bins,
            extra_mask=_final_survivor_mask(model),
        )
        color = ns_color_lookup[float(ns)]
        ax.plot(c0, d0, "--", lw=1.0, color=color)
        ax.plot(c1, d1, "-", lw=1.0, color=color)
    if include_observables:
        f3_g14 = obs_overlay["fig3_g14"]
        f3_b21 = obs_overlay["fig3_b21"]
        f3_rbc = obs_overlay["fig3_rbc"]
        ax.plot(
            f3_g14["r_kpc"],
            f3_g14["sigma"],
            color="red",
            lw=1.0,
            ls=":",
            label="G14",
        )
        ax.errorbar(
            f3_b21["r_kpc"],
            f3_b21["sigma"],
            xerr=f3_b21["xerr"],
            yerr=f3_b21["yerr"],
            fmt="o",
            color="#d12ad1",
            ms=2.0,
            capsize=1.0,
            lw=0.5,
            label="B21",
        )
        ax.errorbar(
            f3_rbc["r_kpc"],
            f3_rbc["sigma"],
            xerr=f3_rbc["xerr"],
            yerr=f3_rbc["yerr"],
            fmt="o",
            color="black",
            ms=2.0,
            capsize=1.0,
            lw=0.5,
            label="RBCver.5",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r~[\mathrm{kpc}]$")
    ax.set_ylabel(r"$\Sigma~[\mathrm{kpc}^{-2}]$")
    ax.set_xlim(0.01, 200.0)
    ax.set_ylim(1.0e-4, 2.0e4)
    ax.set_xticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticks([1.0e-4, 1.0e-2, 1.0, 100.0, 10000.0])
    if include_observables:
        plt.legend(frameon=False, loc="upper right", ncol=1)
    _add_discrete_ns_colorbar(
        fig,
        ax,
        ns_levels=ns_levels,
        cmap=ns_cmap,
        norm=ns_norm,
        boundaries=ns_boundaries,
    )
    save(3, "surface_number_density")

    # Figure 6: z_hm histogram.
    plt.figure(constrained_layout=True, dpi=STD_DPI, figsize=(6.0, 4.2))
    plt.hist(halo_meta["z_hm"].to_numpy(), bins=22, alpha=0.85)
    plt.xlabel(r"$z_{\rm h}$")
    plt.ylabel(r"\# halos")
    plt.xlim(0.0, 3.75)
    plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    save(6, "z_h_hist")

    # Figure 7: radial profiles split by z_hm terciles.
    ref_ns = min(ns_values, key=lambda x: abs(float(x) - 2.0))
    model_ref = models[float(ref_ns)]
    plt.figure(constrained_layout=True, dpi=STD_DPI, figsize=(4.8, 3.6))
    zhm_style = {
        "low": ("tab:blue", r"$z_{\rm h}<%.1f$" % q1),
        "mid": ("tab:green", r"$z_{\rm h}\in[%.1f,%.1f]$" % (q1, q2)),
        "high": ("tab:red", r"$z_{\rm h}>%.1f$" % q2),
    }
    for lbl in ["low", "mid", "high"]:
        hid_sel = halo_meta.index[halo_meta["zhm_bin"] == lbl].to_numpy()
        mask = gc["hid_z0"].isin(hid_sel).to_numpy()
        c0, d0 = _surface_density_mean_by_halo(
            halo_ids[mask],
            model_ref.r_init[mask],
            r_bins,
        )
        c1, d1 = _surface_density_mean_by_halo(
            halo_ids[mask],
            model_ref.r_final[mask],
            r_bins,
            extra_mask=_final_survivor_mask(model_ref, mask)[mask],
        )
        color, label = zhm_style[lbl]
        plt.plot(c0, d0, "--", lw=1.5, color=color, label=f"{label}")
        plt.plot(c1, d1, "-", lw=1.8, color=color)
    if include_observables:
        f3_b21 = obs_overlay["fig3_b21"]
        f3_rbc = obs_overlay["fig3_rbc"]
        plt.errorbar(
            f3_b21["r_kpc"],
            f3_b21["sigma"],
            xerr=f3_b21["xerr"],
            yerr=f3_b21["yerr"],
            fmt="o",
            color="#d12ad1",
            ms=3.5,
            capsize=3.0,
            lw=1.0,
            label="B21",
        )
        plt.errorbar(
            f3_rbc["r_kpc"],
            f3_rbc["sigma"],
            xerr=f3_rbc["xerr"],
            yerr=f3_rbc["yerr"],
            fmt="o",
            color="black",
            ms=3.5,
            capsize=3.0,
            lw=1.0,
            label="RBCver.5",
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$r~[\mathrm{kpc}]$")
    plt.ylabel(r"$\Sigma~[\mathrm{kpc}^{-2}]$")
    plt.xlim(0.1, 200.0)
    plt.ylim(1.0e-4, 2.0e2)
    plt.yticks([1.0e-4, 1.0e-2, 1.0, 100.0])
    plt.legend(fontsize=10, frameon=False)
    save(7, "number_density_by_z_h")

    # Figure 8: MW-like in-situ final mass function.
    insitu = gc["isMPB"].to_numpy().astype(bool)
    mw_hid_nominal = _select_halos_by_logmh(gc, 11.7, 11.9, fallback_n=3)
    mw_hid = (
        mw_hid_nominal
        if len(mw_hid_nominal) >= 8
        else halo_meta.index.to_numpy(dtype=int)
    )
    mw_mask = gc["hid_z0"].isin(mw_hid).to_numpy() & insitu
    m_bins = np.logspace(3.8, 7.1, 12)
    fig, ax = plt.subplots(constrained_layout=True, dpi=STD_DPI, figsize=(4.5, 3.0))
    for ns in ns_values:
        model = models[float(ns)]
        extra_mask = mw_mask & _final_survivor_mask(model)
        cent, hist = _mass_histograms_by_halo(
            halo_ids[mw_mask],
            model.m_final[mw_mask],
            m_bins,
            extra_mask=extra_mask[mw_mask],
        )
        med = np.median(hist, axis=0)
        ax.plot(cent, med, lw=1.0)
        if abs(float(ns) - 2.0) < 1.0e-8:
            q25, q75 = np.quantile(hist, [0.25, 0.75], axis=0)
            ax.fill_between(cent, q25, q75, color="lightsteelblue", alpha=0.55, lw=0.0, label=r"$N_{\rm S}=2.0$ \\ 25\%-75\% quantile")
    if include_observables:
        f8 = obs_overlay["fig8_mass_obs"]
        ax.plot(f8["mass_msun"], f8["count"], color="black", lw=1.0, label="B21")
    ax.set_xscale("log")
    ax.set_ylim(-0.5, 48.0)
    ax.set_xlabel(r"$M_{\rm GC}~[M_\odot]$")
    ax.set_ylabel(r"\# GCs per halo")
    if include_observables:
        ax.legend(frameon=False, loc="upper right", ncol=1)
    _add_discrete_ns_colorbar(
        fig,
        ax,
        ns_levels=ns_levels,
        cmap=ns_cmap,
        norm=ns_norm,
        boundaries=ns_boundaries,
    )
    save(8, "mass_function_MW_insitu_GCs")

    # Figure 10: cumulative initial / final mass profiles averaged over all candidate halos.
    r_grid_pc = np.logspace(0.0, 4.0, 120)
    fig, ax = plt.subplots(constrained_layout=True, dpi=STD_DPI, figsize=(5.1, 3.6))
    smbh_x_fig10 = np.geomspace(2.05, 3.05, len(ns_values))
    bh_total_x_fig10 = np.geomspace(2.05, 3.05, len(ns_values))
    smbh_means_fig10: List[float] = []
    bh_total_means_fig10: List[float] = []
    for i_ns, ns in enumerate(ns_values):
        model = models[float(ns)]
        color = ns_color_lookup[float(ns)]
        c_init = _cumulative_mean_by_halo(
            halo_ids,
            1000.0 * model.r_init,
            gc["M_form"].to_numpy(),
            r_grid_pc,
        )
        if model.deposit_profile is not None:
            # Newer outputs provide the deposited cumulative profile directly,
            # which is the quantity intended for the central-mass panels.
            r_dep_kpc, c_final = _deposit_mean_profile(model.deposit_profile, grid_kpc=r_grid_pc / 1000.0)
            r_dep_pc = 1000.0 * r_dep_kpc
        else:
            r_dep_pc = r_grid_pc
            c_final = _cumulative_mean_by_halo(
                halo_ids,
                1000.0 * model.r_final,
                model.m_final,
                r_grid_pc,
                extra_mask=_final_survivor_mask(model),
            )
        ax.plot(r_grid_pc, c_init, "--", color=color, lw=1.0)
        ax.plot(r_dep_pc, c_final, "-", color=color, lw=1.0)
        halo_table_ns = _halo_level_table(
            gc=gc,
            halo_meta=halo_meta,
            model=model,
            nsc_radius_kpc=float(mw_obs.r_nsc_pc) / 1000.0,
        )
        mean_smbh, std_smbh = _mean_and_std(halo_table_ns["M_smbh"].to_numpy(dtype=float))
        mean_bh_total, std_bh_total = _mean_and_std(
            halo_table_ns["M_bh_total"].to_numpy(dtype=float)
        )
        if np.isfinite(mean_smbh) and mean_smbh > 0.0:
            smbh_means_fig10.append(mean_smbh)
            ax.errorbar(
                [smbh_x_fig10[i_ns]],
                [mean_smbh],
                yerr=std_smbh if std_smbh > 0.0 else None,
                fmt="^",
                ms=3.5,
                mfc="white",
                mec=color,
                color=color,
                capsize=3.0,
                zorder=8,
                label="sunk BHs" if len(smbh_means_fig10) == 1 else None,
            )
        if np.isfinite(mean_bh_total) and mean_bh_total > 0.0:
            bh_total_means_fig10.append(mean_bh_total)
            ax.errorbar(
                [bh_total_x_fig10[i_ns]],
                [mean_bh_total],
                yerr=std_bh_total if std_bh_total > 0.0 else None,
                fmt="D",
                ms=3.5,
                mfc="white",
                mec=color,
                color=color,
                capsize=3.0,
                zorder=8,
                label="total BHs" if len(bh_total_means_fig10) == 1 else None,
            )
    _add_nsc_smbh_points_pc(ax, mw_obs, show_labels=True)
    _add_nsc_smbh_points_pc(ax, m31_obs, show_labels=True)
    ax.plot([], [], "--", color="gray", lw=1.2, label="init")
    ax.plot([], [], "-", color="gray", lw=1.6, label="depo")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r~[\mathrm{pc}]$")
    ax.set_ylabel(r"$M_{\rm encl}~[M_\odot]$")
    ax.set_xlim(2.0, 1.0e4)
    ylo10 = 1.0e6
    bh_means_fig10 = smbh_means_fig10 + bh_total_means_fig10
    if bh_means_fig10:
        ylo10 = min(ylo10, 10.0 ** np.floor(np.log10(max(min(bh_means_fig10), 1.0e-6))))
    ax.set_ylim(ylo10, 3.0e8)
    ax.legend(fontsize=10, frameon=False, loc="lower right", ncol=2)
    _add_discrete_ns_colorbar(
        fig,
        ax,
        ns_levels=ns_levels,
        cmap=ns_cmap,
        norm=ns_norm,
        boundaries=ns_boundaries,
    )
    save(10, "cum_mass")

    halo_table_ref_all = _halo_level_table(
        gc=gc,
        halo_meta=halo_meta,
        model=model_ref,
        nsc_radius_kpc=float(mw_obs.r_nsc_pc) / 1000.0,
    )

    # Figure 11: cumulative mass profiles by z_hm terciles, with all halo
    # profiles shown faintly in the background and tercile averages on top.
    fig, ax = plt.subplots(constrained_layout=True, dpi=STD_DPI, figsize=(6.6, 4.8))
    bg_init_color = "0.78"
    bg_dep_color = "0.62"
    bg_lw = 0.45
    bg_alpha = 0.10
    mass_form_all = gc["M_form"].to_numpy(dtype=float)
    for hid in np.unique(halo_ids):
        hmask = halo_ids == hid
        c_init_h = _cumulative_profile(
            1000.0 * model_ref.r_init[hmask],
            mass_form_all[hmask],
            r_grid_pc,
        )
        ax.plot(r_grid_pc, c_init_h, "--", lw=bg_lw, color=bg_init_color, alpha=bg_alpha, zorder=1)
    if model_ref.deposit_profile is not None:
        grid_kpc = r_grid_pc / 1000.0
        for radii, cum in zip(
            model_ref.deposit_profile.r_outer_kpc,
            model_ref.deposit_profile.cumulative_mass_msun,
        ):
            radii = np.asarray(radii, dtype=float)
            cum = np.asarray(cum, dtype=float)
            if len(radii) == 0 or len(cum) == 0:
                continue
            c_dep_h = np.interp(grid_kpc, radii, cum, left=0.0, right=cum[-1])
            ax.plot(r_grid_pc, c_dep_h, "-", lw=bg_lw, color=bg_dep_color, alpha=bg_alpha, zorder=1)
    else:
        final_mask_all = _final_survivor_mask(model_ref)
        for hid in np.unique(halo_ids):
            hmask = (halo_ids == hid) & final_mask_all
            c_dep_h = _cumulative_profile(
                1000.0 * model_ref.r_final[hmask],
                model_ref.m_final[hmask],
                r_grid_pc,
            )
            ax.plot(r_grid_pc, c_dep_h, "-", lw=bg_lw, color=bg_dep_color, alpha=bg_alpha, zorder=1)
    for lbl in ["low", "mid", "high"]:
        hid_sel = halo_meta.index[halo_meta["zhm_bin"] == lbl].to_numpy()
        sel = gc["hid_z0"].isin(hid_sel).to_numpy()
        c_init = _cumulative_mean_by_halo(
            halo_ids,
            1000.0 * model_ref.r_init,
            gc["M_form"].to_numpy(),
            r_grid_pc,
            extra_mask=sel,
        )
        if model_ref.deposit_profile is not None:
            r_dep_kpc, c_final = _deposit_mean_profile(
                model_ref.deposit_profile,
                grid_kpc=r_grid_pc / 1000.0,
                halo_ids=hid_sel,
            )
            r_dep_pc = 1000.0 * r_dep_kpc
        else:
            r_dep_pc = r_grid_pc
            c_final = _cumulative_mean_by_halo(
                halo_ids,
                1000.0 * model_ref.r_final,
                model_ref.m_final,
                r_grid_pc,
                extra_mask=sel & _final_survivor_mask(model_ref),
            )
        color, label = zhm_style[lbl]
        ax.plot(r_grid_pc, c_init, "--", lw=1.8, color=color, label=label, zorder=3)
        ax.plot(r_dep_pc, c_final, "-", lw=2.0, color=color, zorder=4)
    _add_nsc_smbh_points_pc(ax, mw_obs, show_labels=True)
    _add_nsc_smbh_points_pc(ax, m31_obs, show_labels=True)
    ax.plot([], [], "--", color="0.45", lw=1.2, label="init")
    ax.plot([], [], "-", color="0.45", lw=1.6, label="depo")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r~[\mathrm{pc}]$")
    ax.set_ylabel(r"$M_{\rm encl}~[M_\odot]$")
    ax.set_xlim(2.0, 1.0e4)
    ax.set_ylim(4.0e4, 2.0e9)
    ax.legend(fontsize=9, frameon=False, loc="lower right", ncol=2)
    save(11, "cum_mass_by_zhm")

    # Figure 16: z_hm correlation panels for MW-like halos.
    halo_table_all = halo_table_ref_all
    halo_table = halo_table_all.loc[halo_table_all.index.intersection(mw_hid)].dropna(subset=["z_hm"])
    fig, axs = plt.subplots(4, 1, figsize=(5.4, 10.2), sharex=True)
    x = halo_table["z_hm"].to_numpy()

    for ax, yv, ylabel in [
        (axs[0], halo_table["M_halo"].to_numpy(), r"$m_{halo}\,(M_{\odot})$"),
        (axs[1], halo_table["M_gc_init"].to_numpy(), r"$m_{GCi}\,(M_{\odot})$"),
        (axs[2], halo_table["M_nsc"].to_numpy(), r"$m_{NSC}\,(M_{\odot})$"),
        (axs[3], halo_table["M_gc_final"].to_numpy(), r"$m_{GCf}\,(M_{\odot})$"),
    ]:
        ax.scatter(x, yv, s=7, alpha=0.75, color="black", linewidths=0)
        fx, fy, flo, fhi = _fit_band(x, yv)
        if np.any(np.isfinite(fx)):
            ax.plot(fx, fy, color="gray", lw=1.3)
            ax.fill_between(fx, flo, fhi, color="lightgray", alpha=0.75, lw=0.0)
        ax.text(0.05, 0.82, f"r={_pearson_r(x, yv):.2f}", transform=ax.transAxes)
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[-1].set_xlabel(r"$z_{hm}$")
    save(16, "corr_zhm_panels")

    # Figure 17: halo-level initial vs final GC masses for MW-like halos.
    fig = plt.figure(constrained_layout=True, dpi=STD_DPI, figsize=(6.0, 4.5))
    ax = fig.add_subplot(111)
    x_all = halo_table["M_gc_init"].to_numpy()
    y_all = np.clip(halo_table["M_gc_final"].to_numpy(), 1.0, None)
    zhm_all = halo_table["z_hm"].to_numpy()
    sc = ax.scatter(
        x_all,
        y_all,
        c=zhm_all,
        s=12,
        alpha=0.9,
        cmap="jet",
        linewidths=0,
    )
    fx, fy, flo, fhi = _fit_band(x_all, y_all, logx=True, logy=True)
    ax.fill_between(fx, flo, fhi, color="lightgray", alpha=0.75, lw=0.0)
    ax.plot(fx, fy, color="black", lw=1.3)
    ax.text(0.20, 0.73, f"r={_pearson_r(np.log10(x_all), np.log10(y_all)):.2f}", transform=ax.transAxes)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$z_{hm}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{GCi}\,(M_{\odot})$")
    ax.set_ylabel(r"$m_{GCf}\,(M_{\odot})$")
    save(17, "m_init_vs_m_final")

    # Figure 18: NSC-mass correlations against halo / GC mass metrics for MW-like halos.
    fig, axs = plt.subplots(3, 1, figsize=(5.0, 9.0), sharey=True)
    y_nsc = halo_table["M_nsc"].to_numpy()
    x1 = halo_table["M_halo"].to_numpy() / 1.0e12
    x2 = np.clip(halo_table["M_gc_init"].to_numpy(), 1.0, None)
    x3 = np.clip(halo_table["M_gc_final"].to_numpy(), 1.0, None)

    configs = [
        (axs[0], x1, r"$m_{halo}\,(10^{12}M_{\odot})$", False, True, _pearson_r(x1, _log10_positive_or_nan(y_nsc))),
        (axs[1], x2, r"$m_{GCi}\,(M_{\odot})$", True, True, _pearson_r(_log10_positive_or_nan(x2), _log10_positive_or_nan(y_nsc))),
        (axs[2], x3, r"$m_{GCf}\,(M_{\odot})$", True, True, _pearson_r(_log10_positive_or_nan(x3), _log10_positive_or_nan(y_nsc))),
    ]
    for ax, xx, xlabel, logx, logy, rr in configs:
        scatter_mask = np.isfinite(xx) & np.isfinite(y_nsc)
        if logy:
            scatter_mask &= y_nsc > 0
        if logx:
            scatter_mask &= xx > 0
        ax.scatter(xx[scatter_mask], y_nsc[scatter_mask], s=8, alpha=0.75, color="black", linewidths=0)
        fx, fy, flo, fhi = _fit_band(xx, y_nsc, logx=logx, logy=logy)
        if np.any(np.isfinite(fx)):
            ax.plot(fx, fy, color="gray", lw=1.3)
            ax.fill_between(fx, flo, fhi, color="lightgray", alpha=0.75, lw=0.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$m_{NSC}\,(M_{\odot})$")
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.text(0.82, 0.12, f"r={rr:.2f}", transform=ax.transAxes)
    save(18, "corr_nsc_panels")

    manifest = pd.DataFrame(manifest_rows).sort_values("figure").reset_index(drop=True)
    manifest_path = output_dir / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest


def _parse_ns_values_arg(text: str) -> List[float]:
    """Parse comma-separated N_s values (optionally wrapped by brackets)."""

    cleaned = text.strip().strip("[]")
    if not cleaned:
        raise ValueError("Empty --ns-values string.")
    out = []
    for token in cleaned.split(","):
        tok = token.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("No valid N_s values parsed from --ns-values.")
    return out


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Reproduce Gao+2023 figure suite.")
    parser.add_argument(
        "--allcat",
        type=Path,
        default=Path("/lingshan/disk3/subonan/_outputs/Gao+2023/allcat_s-0_p2-6.75_p3-0.5.txt"),
        help=(
            "Root template allcat path or one per-N_s allcat path. "
            "Per-N_s files are resolved from ns*/ subdirectories."
        ),
    )
    parser.add_argument(
        "--mpb",
        type=Path,
        default=Path("/lingshan/disk3/subonan/_outputs/Gao+2023/mpb_from_fixed_trees.csv"),
        help="Path to MPB table CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/lingshan/disk3/subonan/_outputs/Gao+2023/_plots"),
        help="Output directory for reproduced figures.",
    )
    parser.add_argument(
        "--ns-values",
        type=str,
        default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0",
        help="Comma-separated N_s values, e.g. '0.5,1.0,1.5'.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for stochastic scatter.")
    parser.add_argument("--final-z", "--final-redshift", dest="final_z", type=float, default=None)
    parser.add_argument(
        "--no-observables",
        action="store_true",
        help="Disable observational overlays.",
    )
    args = parser.parse_args()
    ns_values = _parse_ns_values_arg(args.ns_values)

    manifest = build_reproduction(
        allcat_path=args.allcat,
        mpb_path=args.mpb,
        output_dir=args.output,
        ns_values=ns_values,
        seed=args.seed,
        include_observables=not args.no_observables,
        final_redshift=args.final_z,
    )
    print(f"FIGURES_WRITTEN {len(manifest)}")
    print(manifest.to_string(index=False))
    print(f"MANIFEST {args.output / 'figure_manifest.csv'}")


if __name__ == "__main__":
    main()
