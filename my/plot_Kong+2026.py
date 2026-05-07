#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""
Plot IMBH seed diagnostics and sunk-BH summaries from one finished run.

This script reads one per-``N_s`` formation catalogue for Fig.01 and Fig.02:
initial cluster mass versus radius, and initial surface density versus
metallicity with IMBH-mass threshold contours. It also reads the
redshift-resolved halo summary and root ``allcat`` table written by
``my/run.py`` so Fig.03 plots sunk BH mass against descendant z=0 halo mass.
Fig.04 converts the run-output same-redshift MPB halo masses to stellar
masses with the project SMHM helper for comparison to observed ``M_BH-M_*``
points.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, Iterable, List, Tuple

THREAD_CAP_DEFAULT = str(min(64, max(1, os.cpu_count() or 1)))
for env_name in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ.setdefault(env_name, THREAD_CAP_DEFAULT)

import matplotlib as mpl
mpl.use("Agg")

from matplotlib import colors, font_manager
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from IMBH import IMBHModel, IMBHModelConfig  # noqa: E402
import smhm  # noqa: E402

DEFAULT_OUT_DIR = Path("/lingshan/disk3/subonan/_outputs/High-z_SMBHs_Orig_R0.5_z0")
DEFAULT_CLIFF_DATA_DIR = PROJECT_ROOT / "data" / "TheCliff+2026"
STD_DPI = 512
NS_VALUE_DEFAULT = 2.0
FIGURE_01_FILENAME = "Fig.01_initial_cluster_mass_radius_imbh_seeds.png"
FIGURE_02_FILENAME = "Fig.02_initial_surface_density_metallicity_imbh_thresholds.png"
FIGURE_03_FILENAME = "Fig.03_sunk_bh_mass_vs_halo_mass.png"
FIGURE_04_FILENAME = "Fig.04_sunk_bh_mass_vs_stellar_mass_at_redshift.png"
HALO_MASS_UNIT_LABEL = r"M_{\odot}"
SMHM_TOP_AXIS_DEFAULT = True
CLIFF_OBS_FILENAME = "cliff_fig14_mbh_mstar_points.csv"
BH_TO_STELLAR_MASS_RATIOS = (0.01, 0.1, 1.0)
FH_VALUES = (0.125, 0.184, 0.269, 0.395, 0.580)
IMBH_CONTOUR_LEVELS = (100.0, 300.0, 1000.0, 3000.0)
IMBH_SIZE_REFERENCE_MASSES = (100.0, 300.0, 1000.0, 3000.0)
MIN_IMBH_PLOT_MASS_MSUN = 0.0
TIMES_COMPATIBLE_FONT_PATHS = (
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Bold.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Italic.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-BoldItalic.otf"),
)
REQUIRED_SUMMARY_COLUMNS = [
    "hid_z0",
    "z_out",
    "lookback_to_z0_gyr",
    "halo_mass_available",
    "logMh_z_msun",
    "m_smbh_gc_sunk_msun",
    "m_smbh_wanderer_sunk_msun",
    "m_smbh_est_msun",
]
REQUIRED_ALLCAT_COLUMNS = [
    "hid_z0",
    "logMh_z0",
]
REQUIRED_FORMATION_COLUMNS = [
    "logM_form",
    "zform",
    "feh",
    "gc_radius_pc",
    "sigma_h_msun_pc2",
    "imbh_mass_msun",
]
REQUIRED_CLIFF_OBS_COLUMNS = [
    "name",
    "sample",
    "reference",
    "z",
    "logMstar",
    "logMstar_err_lo",
    "logMstar_err_hi",
    "logMstar_upper_limit",
    "logMBH",
    "logMBH_err_lo",
    "logMBH_err_hi",
    "logMBH_upper_limit",
    "plot_group",
    "marker",
    "color",
    "source_note",
]


def _ns_tag(ns_value: float) -> str:
    return f"{float(ns_value):.1f}".replace(".", "p")


def _available_times_serif_fonts() -> List[str]:
    for font_path in TIMES_COMPATIBLE_FONT_PATHS:
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))

    serif_fonts: List[str] = []
    for family in ["Times New Roman", "Nimbus Roman", "Nimbus Roman No9 L", "Times"]:
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        serif_fonts.append(family)
    return serif_fonts or ["DejaVu Serif"]


def _apply_plot_style() -> None:
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": _available_times_serif_fonts(),
            "font.size": 10,
            "mathtext.default": "regular",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": False,
            "text.usetex": use_tex,
        }
    )
    if use_tex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{bm}"


def _load_summary_table(out_dir: Path, ns_value: float) -> pd.DataFrame:
    ns_tag = _ns_tag(ns_value)
    path = out_dir / f"ns{ns_tag}" / f"haloSummaryByZ_ns{ns_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing redshift-resolved halo summary: {path}")
    table = pd.read_csv(path)
    missing = [name for name in REQUIRED_SUMMARY_COLUMNS if name not in table.columns]
    if missing:
        if "halo_mass_available" in missing or "logMh_z_msun" in missing:
            raise ValueError(
                f"{path} is missing required same-redshift halo-mass columns: {missing}. "
                "This output was produced before halo masses at z_out were stored. "
                "Regenerate the run output with the updated my/run.py."
            )
        raise ValueError(f"{path} is missing required columns: {missing}")
    return table.copy()


def _read_comment_columns(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                text = line[1:].strip()
                if text:
                    return text.split()
    raise ValueError(f"Cannot find header columns in {path}")


def _load_ns_formation_table(out_dir: Path, ns_value: float) -> pd.DataFrame:
    ns_tag = _ns_tag(ns_value)
    ns_dir = out_dir / f"ns{ns_tag}"
    candidates = sorted(ns_dir.glob(f"allcat_ns{ns_tag}_s-*_p2-*_p3-*.txt"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"Missing per-N_s formation catalogue in {ns_dir}. Expected allcat_ns{ns_tag}_s-*_p2-*_p3-*.txt.")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(f"Found multiple per-N_s formation catalogues in {ns_dir}; expected exactly one: {names}")

    path = candidates[0]
    columns = _read_comment_columns(path)
    raw = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")
    raw = raw.iloc[:, : len(columns)].copy()
    raw.columns = columns[: raw.shape[1]]
    missing = [name for name in REQUIRED_FORMATION_COLUMNS if name not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    for col in REQUIRED_FORMATION_COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw.copy()


def _imbh_marker_sizes(imbh_mass_msun: np.ndarray) -> np.ndarray:
    mass = np.asarray(imbh_mass_msun, dtype=float)
    sizes = np.full(mass.shape, 18.0, dtype=float)
    valid = np.isfinite(mass) & (mass > 0.0)
    if not np.any(valid):
        return sizes

    log_mass = np.log10(mass[valid])
    lo = float(np.nanmin(log_mass))
    hi = float(np.nanmax(log_mass))
    if hi <= lo:
        sizes[valid] = 55.0
    else:
        sizes[valid] = np.interp(log_mass, (lo, hi), (12.0, 95.0))
    return sizes


def _load_z0_halo_mass_lookup(out_dir: Path) -> Dict[int, float]:
    allcat_candidates = sorted(out_dir.glob("allcat_s-*.txt"))
    if len(allcat_candidates) == 0:
        raise FileNotFoundError(f"Missing root allcat file in {out_dir}. Expected one file matching allcat_s-*.txt.")
    if len(allcat_candidates) > 1:
        names = ", ".join(path.name for path in allcat_candidates)
        raise RuntimeError(f"Found multiple root allcat files in {out_dir}; expected exactly one: {names}")

    path = allcat_candidates[0]
    columns = _read_comment_columns(path)
    raw = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")
    raw = raw.iloc[:, : len(columns)].copy()
    raw.columns = columns[: raw.shape[1]]
    missing = [name for name in REQUIRED_ALLCAT_COLUMNS if name not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    for col in REQUIRED_ALLCAT_COLUMNS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    table = raw.dropna(subset=REQUIRED_ALLCAT_COLUMNS).copy()
    table["hid_z0"] = table["hid_z0"].astype(int)
    grouped = table.groupby("hid_z0", sort=True)["logMh_z0"].first()
    return {int(hid): float(logmh) for hid, logmh in grouped.items()}


def _stellar_mass_from_halo_mass_at_redshift(
    halo_mass: np.ndarray | float,
    redshift: np.ndarray | float,
) -> np.ndarray | float:
    mh = np.asarray(halo_mass, dtype=float)
    z = np.asarray(redshift, dtype=float)
    mh_broadcast, z_broadcast = np.broadcast_arrays(mh, z)
    out = np.full(mh_broadcast.shape, np.nan, dtype=float)
    flat_mh = mh_broadcast.ravel()
    flat_z = z_broadcast.ravel()
    flat_out = out.ravel()
    for i, (mass, z_val) in enumerate(zip(flat_mh, flat_z)):
        if np.isfinite(mass) and mass > 0.0 and np.isfinite(z_val):
            flat_out[i] = smhm.SMHM(float(mass), float(z_val), k=True, scatter=False, mdef="mvir")
    if np.isscalar(halo_mass) and np.isscalar(redshift):
        return float(out.reshape(-1)[0])
    return out


def _attach_plot_masses(summary: pd.DataFrame, z0_halo_mass_lookup: Dict[int, float]) -> pd.DataFrame:
    out = summary.copy()
    out["hid_z0"] = pd.to_numeric(out["hid_z0"], errors="coerce").astype(int)
    out["z_out"] = pd.to_numeric(out["z_out"], errors="coerce")
    out["logMh_z0_msun"] = out["hid_z0"].map(z0_halo_mass_lookup)
    missing_z0 = out.loc[~np.isfinite(out["logMh_z0_msun"].to_numpy(dtype=float)), "hid_z0"].unique()
    if len(missing_z0) > 0:
        missing_text = ", ".join(str(int(hid)) for hid in sorted(missing_z0)[:8])
        raise KeyError(f"Root allcat is missing descendant z=0 halo mass for halo(s): {missing_text}")

    out["mhalo_z0_msun"] = np.power(10.0, out["logMh_z0_msun"].to_numpy(dtype=float))
    out["halo_mass_available"] = pd.to_numeric(out["halo_mass_available"], errors="coerce").fillna(0).astype(int)
    out["logMh_z_msun"] = pd.to_numeric(out["logMh_z_msun"], errors="coerce")
    valid_halo_z = (
        (out["halo_mass_available"].to_numpy(dtype=int) == 1)
        & np.isfinite(out["logMh_z_msun"].to_numpy(dtype=float))
    )
    out["mhalo_z_msun"] = np.nan
    out.loc[valid_halo_z, "mhalo_z_msun"] = np.power(
        10.0,
        out.loc[valid_halo_z, "logMh_z_msun"].to_numpy(dtype=float),
    )
    out["mstar_z_smhm_msun"] = _stellar_mass_from_halo_mass_at_redshift(
        out["mhalo_z_msun"].to_numpy(dtype=float),
        out["z_out"].to_numpy(dtype=float),
    )
    mstar = out["mstar_z_smhm_msun"].to_numpy(dtype=float)
    out["logMstar_z_smhm_msun"] = np.where(
        np.isfinite(mstar) & (mstar > 0.0),
        np.log10(mstar),
        np.nan,
    )
    return out


def _regular_log_bin_edges(values: Iterable[float], step_dex: float) -> np.ndarray:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([0.0, float(step_dex)], dtype=float)
    lo = float(step_dex) * math.floor(float(vals.min()) / float(step_dex))
    hi = float(step_dex) * math.ceil(float(vals.max()) / float(step_dex))
    if hi <= lo:
        hi = lo + float(step_dex)
    edges = np.arange(lo, hi + 0.5 * float(step_dex), float(step_dex), dtype=float)
    if len(edges) < 2:
        edges = np.array([lo, lo + float(step_dex)], dtype=float)
    return edges


def _bin_track(track: pd.DataFrame, edges: np.ndarray, x_log_col: str) -> pd.DataFrame:
    x_log = track[x_log_col].to_numpy(dtype=float)
    y = track["m_smbh_est_msun"].to_numpy(dtype=float)
    rows = []
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        include_right = idx == len(edges) - 2
        mask = np.isfinite(x_log) & np.isfinite(y) & (x_log >= left)
        mask &= (x_log <= right) if include_right else (x_log < right)
        if int(mask.sum()) == 0:
            continue
        y_sel = y[mask]
        rows.append(
            {
                "logx_center": 0.5 * (left + right),
                "count": int(mask.sum()),
                "mean_mass": float(np.mean(y_sel)),
                "std_mass": float(np.std(y_sel)),
            }
        )
    return pd.DataFrame(rows)


def _present_day_stellar_mass_from_halo_mass(halo_mass: np.ndarray | float) -> np.ndarray | float:
    return _stellar_mass_from_halo_mass_at_redshift(halo_mass, 0.0)


_PRESENT_DAY_SMHM_AXIS_CACHE: Tuple[np.ndarray, np.ndarray] | None = None


def _present_day_halo_mass_from_stellar_mass(stellar_mass: np.ndarray | float) -> np.ndarray | float:
    global _PRESENT_DAY_SMHM_AXIS_CACHE
    if _PRESENT_DAY_SMHM_AXIS_CACHE is None:
        log_mh_grid = np.linspace(8.0, 16.0, 4096)
        mh_grid = np.power(10.0, log_mh_grid)
        mstar_grid = _present_day_stellar_mass_from_halo_mass(mh_grid)
        valid_grid = np.isfinite(mstar_grid) & (mstar_grid > 0.0)
        log_mstar_grid = np.log10(mstar_grid[valid_grid])
        log_mh_grid = log_mh_grid[valid_grid]
        order = np.argsort(log_mstar_grid)
        log_mstar_sorted = log_mstar_grid[order]
        log_mh_sorted = log_mh_grid[order]
        log_mstar_unique, unique_idx = np.unique(log_mstar_sorted, return_index=True)
        _PRESENT_DAY_SMHM_AXIS_CACHE = (log_mstar_unique, log_mh_sorted[unique_idx])

    log_mstar_grid, log_mh_grid = _PRESENT_DAY_SMHM_AXIS_CACHE
    sm = np.asarray(stellar_mass, dtype=float)
    out = np.full(sm.shape, np.nan, dtype=float)
    valid = np.isfinite(sm) & (sm > 0.0)
    out[valid] = np.power(
        10.0,
        np.interp(np.log10(sm[valid]), log_mstar_grid, log_mh_grid, left=np.nan, right=np.nan),
    )
    if np.isscalar(stellar_mass):
        return float(out.reshape(-1)[0])
    return out


def _add_smhm_top_axis(ax: plt.Axes) -> None:
    secax = ax.secondary_xaxis(
        "top",
        functions=(_present_day_stellar_mass_from_halo_mass, _present_day_halo_mass_from_stellar_mass),
    )
    secax.set_xlabel(r"Corresponding $z=0$ stellar mass from SMHM [$M_{\odot}$]")
    secax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    secax.tick_params(direction="in", top=True, which="both")


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _load_cliff_fig14_observations(data_dir: Path) -> pd.DataFrame:
    path = data_dir / CLIFF_OBS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing Cliff Fig.14 observation table: {path}")
    table = pd.read_csv(path)
    missing = [name for name in REQUIRED_CLIFF_OBS_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    numeric_columns = [
        "z",
        "logMstar",
        "logMstar_err_lo",
        "logMstar_err_hi",
        "logMBH",
        "logMBH_err_lo",
        "logMBH_err_hi",
    ]
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    for column in ["logMstar_upper_limit", "logMBH_upper_limit"]:
        table[column] = table[column].map(_as_bool)
    for column in ["name", "sample", "reference", "plot_group", "marker", "color", "source_note"]:
        table[column] = table[column].fillna("").astype(str)

    valid = np.isfinite(table["logMstar"].to_numpy(dtype=float)) & np.isfinite(table["logMBH"].to_numpy(dtype=float))
    return table.loc[valid].copy()


def _log_error_to_linear(log_value: float, err_lo: float, err_hi: float) -> np.ndarray | None:
    if not np.isfinite(log_value):
        return None
    value = 10.0**log_value
    lo = float(err_lo) if np.isfinite(err_lo) and err_lo > 0.0 else 0.0
    hi = float(err_hi) if np.isfinite(err_hi) and err_hi > 0.0 else 0.0
    if lo <= 0.0 and hi <= 0.0:
        return None
    return np.array([[value - 10.0 ** (log_value - lo)], [10.0 ** (log_value + hi) - value]], dtype=float)


def _plot_mbh_mstar_observations(ax: plt.Axes, observations: pd.DataFrame) -> None:
    seen_labels: set[str] = set()
    for _, row in observations.iterrows():
        log_mstar = float(row["logMstar"])
        log_mbh = float(row["logMBH"])
        x = 10.0**log_mstar
        y = 10.0**log_mbh
        colour = str(row["color"]).strip() or "0.45"
        marker = str(row["marker"]).strip() or "o"
        group = str(row["plot_group"]).strip().lower()
        if group == "cliff":
            label_base = str(row["name"]).strip() or "The Cliff"
            zorder = 7
            alpha = 1.0
            marker_size = 6.5
        else:
            label_base = "Cliff Fig.14 comparison points"
            colour = colour if colour else "0.55"
            zorder = 5
            alpha = 0.75
            marker_size = 5.5
        label = None if label_base in seen_labels else label_base
        seen_labels.add(label_base)

        yerr = None
        if not _as_bool(row["logMBH_upper_limit"]):
            yerr = _log_error_to_linear(log_mbh, float(row["logMBH_err_lo"]), float(row["logMBH_err_hi"]))

        xerr = None
        if not _as_bool(row["logMstar_upper_limit"]):
            xerr = _log_error_to_linear(log_mstar, float(row["logMstar_err_lo"]), float(row["logMstar_err_hi"]))

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            ms=marker_size,
            mfc=colour,
            mec="white" if group == "cliff" else colour,
            mew=0.7 if group == "cliff" else 0.0,
            ecolor=colour,
            elinewidth=1.0,
            capsize=2.5,
            color=colour,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )
        if _as_bool(row["logMstar_upper_limit"]):
            ax.annotate(
                "",
                xy=(x * 0.55, y),
                xytext=(x * 0.95, y),
                arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.1, "alpha": alpha},
                zorder=zorder,
            )


def plot_fig01(formation: pd.DataFrame) -> plt.Figure:
    log_mass = formation["logM_form"].to_numpy(dtype=float)
    cluster_mass = np.power(10.0, log_mass)
    radius_pc = formation["gc_radius_pc"].to_numpy(dtype=float)
    z_form = formation["zform"].to_numpy(dtype=float)
    imbh_mass = formation["imbh_mass_msun"].to_numpy(dtype=float)
    valid = (
        np.isfinite(cluster_mass)
        & (cluster_mass > 0.0)
        & np.isfinite(radius_pc)
        & (radius_pc > 0.0)
        & np.isfinite(z_form)
        & np.isfinite(imbh_mass)
        & (imbh_mass > MIN_IMBH_PLOT_MASS_MSUN)
    )
    if not np.any(valid):
        raise ValueError("No finite IMBH-seeded formation rows are available for Fig.01.")

    cluster_mass = cluster_mass[valid]
    radius_pc = radius_pc[valid]
    z_form = z_form[valid]
    imbh_mass = imbh_mass[valid]
    reference_masses = np.asarray(IMBH_SIZE_REFERENCE_MASSES, dtype=float)
    combined_sizes = _imbh_marker_sizes(np.concatenate([imbh_mass, reference_masses]))
    marker_sizes = combined_sizes[: len(imbh_mass)]
    reference_sizes = combined_sizes[len(imbh_mass) :]

    x_min = cluster_mass.min() / 1.25
    x_max = cluster_mass.max() * 1.25
    mass_grid = np.logspace(np.log10(x_min), np.log10(x_max), 512)
    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))

    fh_colours = mpl.cm.cividis(np.linspace(0.15, 0.9, len(FH_VALUES)))
    curve_radii = []
    for fh, colour in zip(FH_VALUES, fh_colours):
        model = IMBHModel(IMBHModelConfig(enabled=True, fh=fh))
        radius_grid = np.asarray(model.radius_eq7(mass_grid), dtype=float)
        curve_radii.append(radius_grid)
        ax.plot(mass_grid, radius_grid, c=colour, lw=1.6, alpha=0.95, label=rf"$f_h={fh:.3f}$")

    if len(np.unique(z_form)) == 1:
        norm = colors.Normalize(vmin=float(z_form[0]) - 0.5, vmax=float(z_form[0]) + 0.5)
    else:
        norm = colors.Normalize(vmin=float(np.nanmin(z_form)), vmax=float(np.nanmax(z_form)))
    scatter = ax.scatter(cluster_mass, radius_pc, c=z_form, s=marker_sizes, cmap=mpl.cm.viridis, norm=norm, alpha=0.35, edgecolors="none", rasterized=True, zorder=5)
    colour_bar = fig.colorbar(scatter, ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Formation redshift z")

    curve_legend = ax.legend(loc="lower right", fontsize=7.0, frameon=False, title=r"Eq. (7)")
    ax.add_artist(curve_legend)
    size_handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor="0.45", markeredgecolor="none", markersize=float(np.sqrt(size)), lw=0, label=rf"${mass:g}\ M_\odot$")
        for mass, size in zip(reference_masses, reference_sizes)
    ]
    ax.legend(handles=size_handles, loc="upper left", fontsize=7.0, frameon=False, title=r"IMBH seed mass")

    y_values = np.concatenate([radius_pc, *curve_radii])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(float(np.nanmin(y_values)) / 1.25, float(np.nanmax(y_values)) * 1.25)
    ax.set_xlabel(r"Initial cluster mass $M_{\rm cl}$ [$M_{\odot}$]")
    ax.set_ylabel(r"Initial 3D half-mass radius $r_h$ [pc]")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig


def plot_fig02(formation: pd.DataFrame) -> plt.Figure:
    sigma_h = formation["sigma_h_msun_pc2"].to_numpy(dtype=float)
    feh = formation["feh"].to_numpy(dtype=float)
    z_form = formation["zform"].to_numpy(dtype=float)
    imbh_mass = formation["imbh_mass_msun"].to_numpy(dtype=float)
    valid = (
        np.isfinite(sigma_h)
        & (sigma_h > 0.0)
        & np.isfinite(feh)
        & np.isfinite(z_form)
        & np.isfinite(imbh_mass)
        & (imbh_mass > MIN_IMBH_PLOT_MASS_MSUN)
    )
    if not np.any(valid):
        raise ValueError("No finite IMBH-seeded formation rows are available for Fig.02.")

    sigma_h = sigma_h[valid]
    feh = feh[valid]
    z_form = z_form[valid]
    imbh_mass = imbh_mass[valid]
    reference_masses = np.asarray(IMBH_SIZE_REFERENCE_MASSES, dtype=float)
    combined_sizes = _imbh_marker_sizes(np.concatenate([imbh_mass, reference_masses]))
    marker_sizes = combined_sizes[: len(imbh_mass)]
    reference_sizes = combined_sizes[len(imbh_mass) :]

    log_sigma = np.log10(sigma_h)
    log_sigma_span = max(float(log_sigma.max() - log_sigma.min()), 0.3)
    log_sigma_min = float(log_sigma.min()) - 0.05 * log_sigma_span
    log_sigma_max = float(log_sigma.max()) + 0.05 * log_sigma_span
    feh_span = max(float(feh.max() - feh.min()), 0.3)
    feh_min = float(feh.min()) - 0.05 * feh_span
    feh_max = float(feh.max()) + 0.05 * feh_span
    sigma_grid_1d = np.logspace(log_sigma_min, log_sigma_max, 480)
    feh_grid_1d = np.linspace(feh_min, feh_max, 360)
    sigma_grid, feh_grid = np.meshgrid(sigma_grid_1d, feh_grid_1d)
    contour_model = IMBHModel(IMBHModelConfig(enabled=True, metallicity_kind="z_ratio"))
    imbh_grid = contour_model.imbh_mass_from_sigma_metallicity(sigma_grid, np.power(10.0, feh_grid))

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    if len(np.unique(z_form)) == 1:
        norm = colors.Normalize(vmin=float(z_form[0]) - 0.5, vmax=float(z_form[0]) + 0.5)
    else:
        norm = colors.Normalize(vmin=float(np.nanmin(z_form)), vmax=float(np.nanmax(z_form)))
    scatter = ax.scatter(sigma_h, feh, c=z_form, s=marker_sizes, cmap=mpl.cm.viridis, norm=norm, alpha=0.35, edgecolors="none", rasterized=True, zorder=5)
    finite_grid = imbh_grid[np.isfinite(imbh_grid)]
    if finite_grid.size > 0:
        levels = [level for level in IMBH_CONTOUR_LEVELS if float(finite_grid.min()) <= level <= float(finite_grid.max())]
        if levels:
            contours = ax.contour(sigma_grid, feh_grid, imbh_grid, levels=levels, colors="black", linewidths=1.2, zorder=7)
            ax.clabel(contours, inline=True, fmt=rf"%d $M_\odot$", fontsize=7.0)
    colour_bar = fig.colorbar(scatter, ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Formation redshift z")

    size_handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor="0.45", markeredgecolor="none", markersize=float(np.sqrt(size)), lw=0, label=rf"${mass:g}\ M_\odot$")
        for mass, size in zip(reference_masses, reference_sizes)
    ]
    ax.legend(handles=size_handles, loc="upper right", fontsize=7.0, frameon=False, title=r"IMBH seed mass")

    # Formation outputs store metallicity as [Fe/H]; the IMBH contour model uses Z/Zsun internally.
    ax.set_xscale("log")
    ax.set_xlim(10.0**log_sigma_min, 10.0**log_sigma_max)
    ax.set_ylim(feh_min, feh_max)
    ax.set_xlabel(r"Initial half-mass surface density $\Sigma_h$ [$M_{\odot}\,{\rm pc}^{-2}$]")
    ax.set_ylabel(r"Initial metallicity [Fe/H]")
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.tick_params(direction="in", right=True, top=True, which="both")
    return fig


def plot_fig03(joined: pd.DataFrame, mass_bin_width_dex: float, add_stellar_mass_axis: bool) -> plt.Figure:
    x_log_col = "logMh_z0_msun"
    plot_rows = joined[
        np.isfinite(joined[x_log_col].to_numpy(dtype=float))
        & np.isfinite(joined["m_smbh_est_msun"].to_numpy(dtype=float))
    ].copy()
    if len(plot_rows) == 0:
        raise ValueError("No finite rows are available for plotting.")

    z_values = np.sort(plot_rows["z_out"].unique())
    edges = _regular_log_bin_edges(plot_rows[x_log_col], step_dex=mass_bin_width_dex)

    if len(z_values) == 1:
        norm = colors.Normalize(vmin=float(z_values[0]) - 0.5, vmax=float(z_values[0]) + 0.5)
    else:
        norm = colors.Normalize(vmin=float(z_values.min()), vmax=float(z_values.max()))
    cmap = mpl.cm.viridis

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    n_tracks = 0
    for z_out in z_values:
        track = plot_rows[plot_rows["z_out"] == float(z_out)].copy()
        binned = _bin_track(track, edges, x_log_col=x_log_col)
        if len(binned) == 0:
            continue
        mean_mass = binned["mean_mass"].to_numpy(dtype=float)
        valid = np.isfinite(mean_mass) & (mean_mass > 0.0)
        if not np.any(valid):
            continue
        x = np.power(10.0, binned.loc[valid, "logx_center"].to_numpy(dtype=float))
        mean_mass = mean_mass[valid]
        std_mass = binned.loc[valid, "std_mass"].to_numpy(dtype=float)
        lower = np.maximum(mean_mass - std_mass, mean_mass * 1.0e-3)
        upper = np.maximum(mean_mass + std_mass, mean_mass * 1.0e-3)
        colour = cmap(norm(float(z_out)))
        ax.fill_between(x, lower, upper, color=colour, alpha=0.18, edgecolor="none")
        ax.plot(x, mean_mass, c=colour, lw=2.0)
        n_tracks += 1

    if n_tracks == 0:
        raise ValueError("All binned sunk-BH tracks are empty or non-positive.")

    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(z_values) == 1:
        colour_bar.set_ticks([float(z_values[0])])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"Descendant $z=0$ halo mass [${HALO_MASS_UNIT_LABEL}$]")
    ax.set_ylabel(r"Sunk BH mass [$M_{\odot}$]")
    ax.set_xlim(left=10.0**edges[0], right=10.0**edges[-1])
    ax.set_ylim(bottom=1.0e2, top=1.0e8)
    if add_stellar_mass_axis:
        _add_smhm_top_axis(ax)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    return fig


def plot_fig04(joined: pd.DataFrame, mass_bin_width_dex: float, observations: pd.DataFrame | None) -> plt.Figure:
    x_log_col = "logMstar_z_smhm_msun"
    plot_rows = joined[
        np.isfinite(joined[x_log_col].to_numpy(dtype=float))
        & np.isfinite(joined["m_smbh_est_msun"].to_numpy(dtype=float))
    ].copy()
    if len(plot_rows) == 0:
        raise ValueError("No finite rows are available for plotting.")

    z_values = np.sort(plot_rows["z_out"].unique())
    edges = _regular_log_bin_edges(plot_rows[x_log_col], step_dex=mass_bin_width_dex)
    x_limit_values = plot_rows[x_log_col].to_numpy(dtype=float)
    if observations is not None and len(observations) > 0:
        x_limit_values = np.concatenate([x_limit_values, observations["logMstar"].to_numpy(dtype=float)])
    x_limit_edges = _regular_log_bin_edges(x_limit_values, step_dex=mass_bin_width_dex)

    if len(z_values) == 1:
        norm = colors.Normalize(vmin=float(z_values[0]) - 0.5, vmax=float(z_values[0]) + 0.5)
    else:
        norm = colors.Normalize(vmin=float(z_values.min()), vmax=float(z_values.max()))
    cmap = mpl.cm.viridis

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    n_tracks = 0
    for z_out in z_values:
        track = plot_rows[plot_rows["z_out"] == float(z_out)].copy()
        binned = _bin_track(track, edges, x_log_col=x_log_col)
        if len(binned) == 0:
            continue
        mean_mass = binned["mean_mass"].to_numpy(dtype=float)
        valid = np.isfinite(mean_mass) & (mean_mass > 0.0)
        if not np.any(valid):
            continue
        x = np.power(10.0, binned.loc[valid, "logx_center"].to_numpy(dtype=float))
        mean_mass = mean_mass[valid]
        std_mass = binned.loc[valid, "std_mass"].to_numpy(dtype=float)
        lower = np.maximum(mean_mass - std_mass, mean_mass * 1.0e-3)
        upper = np.maximum(mean_mass + std_mass, mean_mass * 1.0e-3)
        colour = cmap(norm(float(z_out)))
        ax.fill_between(x, lower, upper, color=colour, alpha=0.18, edgecolor="none")
        ax.plot(x, mean_mass, c=colour, lw=2.0)
        n_tracks += 1

    if n_tracks == 0:
        raise ValueError("All binned sunk-BH tracks are empty or non-positive.")

    colour_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, aspect=30, pad=0.0)
    colour_bar.set_label("Redshift z")
    if len(z_values) == 1:
        colour_bar.set_ticks([float(z_values[0])])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rf"Stellar mass at redshift $z$ from MPB $M_h(z)$ and SMHM [${HALO_MASS_UNIT_LABEL}$]")
    ax.set_ylabel(r"Sunk BH mass [$M_{\odot}$]")
    ax.set_xlim(left=10.0**x_limit_edges[0], right=10.0**x_limit_edges[-1])
    ax.set_ylim(bottom=1.0e2, top=1.0e8)

    x_line = np.logspace(x_limit_edges[0], x_limit_edges[-1], 256)
    for ratio in BH_TO_STELLAR_MASS_RATIOS:
        ratio_label = f"{ratio:g}"
        ax.plot(
            x_line,
            ratio * x_line,
            c="#7b3294",
            ls="--",
            lw=1.1,
            alpha=0.8,
            label=rf"$M_{{\rm BH}}/M_\ast={ratio_label}$",
            zorder=2,
        )
    if observations is not None and len(observations) > 0:
        _plot_mbh_mstar_observations(ax, observations)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.85)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IMBH seed diagnostics, sunk-BH halo tracks, and SMHM comparison points from one local High-z SMBHs output directory.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Model output directory containing ns*/allcat_ns*.txt, allcat_s-*.txt, and ns*/haloSummaryByZ_ns*.csv.")
    parser.add_argument("--plot_dir", type=Path, default=None, help="Plot output directory. Default: <out_dir>/_plots_Kong+2026")
    parser.add_argument("--ns-value", type=float, default=NS_VALUE_DEFAULT, help="Single N_s value used for all four figures; Fig.01 and Fig.02 intentionally use only this per-N_s formation catalogue.")
    parser.add_argument("--mass-bin-width-dex", type=float, default=0.5, help="Log10 halo-mass bin width.")
    parser.add_argument("--no-smhm-top-axis", action="store_true", help="Do not add the SMHM stellar-mass top x-axis to Fig.03.")
    parser.add_argument("--cliff-data-dir", type=Path, default=DEFAULT_CLIFF_DATA_DIR, help="Directory containing Cliff Fig.14 observation CSV files.")
    parser.add_argument("--no-cliff-observations", action="store_true", help="Do not overlay Cliff Fig.14 observational points on Fig.04.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    plot_dir = args.plot_dir.resolve() if args.plot_dir is not None else (out_dir / "_plots_Kong+2026").resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    _apply_plot_style()
    summary = _load_summary_table(out_dir, args.ns_value)
    formation = _load_ns_formation_table(out_dir, args.ns_value)
    z0_halo_mass_lookup = _load_z0_halo_mass_lookup(out_dir)
    joined = _attach_plot_masses(summary, z0_halo_mass_lookup)
    cliff_observations = None
    if not args.no_cliff_observations:
        cliff_observations = _load_cliff_fig14_observations(args.cliff_data_dir.resolve())

    fig01 = plot_fig01(formation)
    path01 = plot_dir / FIGURE_01_FILENAME
    fig01.savefig(path01, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig01)
    print(f"Saved {path01}")

    fig02 = plot_fig02(formation)
    path02 = plot_dir / FIGURE_02_FILENAME
    fig02.savefig(path02, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig02)
    print(f"Saved {path02}")

    fig03 = plot_fig03(
        joined,
        mass_bin_width_dex=float(args.mass_bin_width_dex),
        add_stellar_mass_axis=SMHM_TOP_AXIS_DEFAULT and not args.no_smhm_top_axis,
    )
    path03 = plot_dir / FIGURE_03_FILENAME
    fig03.savefig(path03, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig03)
    print(f"Saved {path03}")

    # Fig.04 uses same-redshift MPB halo masses stored by my/run.py.
    fig04 = plot_fig04(
        joined,
        mass_bin_width_dex=float(args.mass_bin_width_dex),
        observations=cliff_observations,
    )
    path04 = plot_dir / FIGURE_04_FILENAME
    fig04.savefig(path04, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig04)
    print(f"Saved {path04}")


if __name__ == "__main__":
    main()
