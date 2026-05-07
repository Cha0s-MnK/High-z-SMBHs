#!/usr/bin/env python3

"""
Batch driver for the Python rewrite of Gao+2023 GC evolution.

This workflow uses the bundled project ``data/`` directory, with an optional
override for the fixed-tree input directory:

- ``fixed_trees_large_spin`` (halo trees)
- ``mass_loss.txt`` (stellar-evolution mass-loss table)
- ``snaps2redshifts.txt`` (snapshot-redshift table)

The script performs three major steps:
1. Run ``src/main_spatial.py`` per Sersic index ``N_s`` to build fresh GC formation catalogs from raw trees.
2. Evolve each catalog halo-by-halo with ``src/evo.py`` physics.
3. Write plotting-ready outputs consumed by the paper-specific plot scripts in ``my/``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
from functools import lru_cache
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MAIN_SPATIAL_PATH = SRC_DIR / "main_spatial.py"
EVO_PATH = SRC_DIR / "evo.py"
SMHM_PATH = SRC_DIR / "smhm.py"
PLOT_GAO2023_PATH = PROJECT_ROOT / "my" / "plot_Gao+2023.py"
PLOT_CHOKSI2018_PATH = PROJECT_ROOT / "my" / "plot_Choksi+2018.py"
PLOT_NEUMAYER2020_PATH = PROJECT_ROOT / "my" / "plot_Neumayer+2020.py"
PLOT_KONG2026_PATH = PROJECT_ROOT / "my" / "plot_Kong+2026.py"
DATA_DIR = PROJECT_ROOT / "data"
SNAPS_PATH = DATA_DIR / "snaps2redshifts.txt"
MASS_LOSS_PATH = DATA_DIR / "mass_loss.txt"
DEFAULT_TREE_DIR = DATA_DIR / "fixed_trees_large_spin"
TREE_LOOKUP_BASENAME = "id_lookup_large_dark.csv"
for _required_path, _label, _kind in (
    (SRC_DIR, "source directory", "dir"),
    (MAIN_SPATIAL_PATH, "formation script", "file"),
    (EVO_PATH, "evolution module", "file"),
    (SMHM_PATH, "SMHM module", "file"),
):
    if _kind == "dir" and (not _required_path.is_dir()):
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing {_label}: {_required_path}"
        )
    if _kind == "file" and (not _required_path.is_file()):
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing {_label}: {_required_path}"
        )
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evo import (  # noqa: E402
    Redshift2CosmicTimeGyr,
    TIDAL_STRIPPING_CHOICES,
    TIDAL_STRIPPING_FRAGIONE2019,
    evolve_single_halo,
    read_haloevo_mpb,
)
import smhm  # noqa: E402


NS_VALUES_DEFAULT = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
EXTRA_OUT_Z_LIST_DEFAULT = "1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0"
METAL_CHOKSI2018 = "Choksi+2018"
METAL_CHEN_GNEDIN2024 = "Chen&Gnedin2024"
METAL_CHOICES = (METAL_CHOKSI2018, METAL_CHEN_GNEDIN2024)
ACCRETED_BARYON_MURATOV_GNEDIN2010 = "Muratov&Gnedin2010"
ACCRETED_BARYON_CHEN_GNEDIN2023 = "Chen&Gnedin2023"
ACCRETED_BARYON_CHOICES = (ACCRETED_BARYON_MURATOV_GNEDIN2010, ACCRETED_BARYON_CHEN_GNEDIN2023)

FINAL_GC_HEADER = "\n".join([
    ("hid_z0 logMh_z0 subfind_form logMh_form logMstar_form logMgas_form "
     "logM_form zform feh r_galaxy_kpc gc_radius_pc sigma_h_msun_pc2 imbh_mass_msun"),
    "rows: one formed GC per row; this is the per-halo format evolution input table",])

ALLCAT_HEADER = "\n".join([
    ("hid_z0 logMh_z0 logMstar_z0 logMh_form logMstar_form logM_form "
     "zform feh isMPB subfind_form snap_form r_galaxy_kpc "
     "gc_radius_pc sigma_h_msun_pc2 imbh_mass_msun"),
    "rows: one formed GC per row; companion finalGCs_ns files use the same row ordering",])

COMBINED_FINAL_GC_HEADER = "\n".join(
    [("halo_id_z0 gc_index_halo status m_final_msun log10_m_final_msun "
      "m_init_msun lookback_time_final_gyr lookback_time_init_gyr "
      "r_final_kpc r_init_kpc gc_radius_pc sigma_h_msun_pc2 feh "
      "imbh_mass_msun"),
     ("rows: one GC row per allcat_ns row for this N_s; feh and "
      "the GC/IMBH columns are fixed at formation."),])

COMBINED_DEPOS_HEADER = "\n".join([
    "halo_id_z0 lookback_time_gyr bin_index r_inner_kpc r_outer_kpc m_depo_total_msun m_star_no_evo_msun m_star_with_evo_msun",
    "rows: one deposited radial-bin row from the per-halo Depos files for this N_s; halo_id_z0 identifies the source halo",])

GLOBAL_FINAL_GC_HEADER = "\n".join([
    ("ns halo_id_z0 gc_index_halo status m_final_msun log10_m_final_msun "
     "m_init_msun lookback_time_final_gyr lookback_time_init_gyr "
     "r_final_kpc r_init_kpc gc_radius_pc sigma_h_msun_pc2 feh "
     "imbh_mass_msun"),
    "rows: one GC row from the per-N_s finalGCs files; ns and halo_id_z0 identify the source run and halo",])

GLOBAL_DEPOS_HEADER = "\n".join([
    "ns halo_id_z0 lookback_time_gyr bin_index r_inner_kpc r_outer_kpc m_depo_total_msun m_star_no_evo_msun m_star_with_evo_msun",
    "rows: one deposited radial-bin row from the per-N_s combined Depos files; ns and halo_id_z0 identify the source run and halo",])

HALO_SUMMARY_COLUMNS = [
    "hid_z0",
    "logMh_z0",
    "n_gc_total",
    "n_alive",
    "n_wanderer",
    "n_exhausted",
    "n_torn",
    "n_sunk_gc",
    "n_sunk_wanderer",
    "n_sunk",
    "m_gc_init_total_msun",
    "m_gc_final_total_msun",
    "m_imbh_seed_total_msun",
    "m_smbh_gc_sunk_msun",
    "m_smbh_wanderer_sunk_msun",
    "m_smbh_est_msun",
]
HALO_SUMMARY_BY_Z_COLUMNS = [
    "hid_z0",
    "z_out",
    "lookback_to_z0_gyr",
    "halo_mass_available",
    "logMh_z_msun",
    "m_smbh_gc_sunk_msun",
    "m_smbh_wanderer_sunk_msun",
    "m_smbh_est_msun",
]

RUN_METADATA_NAME = "run_metadata.json"
HALO_TREE_LOOKUP_NAME = "halo_tree_lookup.csv"


def _ns_tag(ns: float) -> str:
    """Convert one Sersic index into the filename-safe `0p5` style tag."""

    return f"{float(ns):.1f}".replace(".", "p")


def _fmt_param_tag(value: float) -> str:
    """Compact float formatting for output filenames."""

    return f"{float(value):g}"


def _ns_output_dir(base_output_dir: Path, ns_value: float) -> Path:
    """Return the per-N_s output directory and create it if needed."""

    path = base_output_dir / f"ns{_ns_tag(ns_value)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _final_gcs_ns_name(ns_value: float) -> str:
    return f"finalGCs_ns{_ns_tag(ns_value)}.dat"


def _depos_ns_name(ns_value: float) -> str:
    return f"depos_ns{_ns_tag(ns_value)}.dat"


def _halo_summary_by_z_ns_name(ns_value: float) -> str:
    return f"haloSummaryByZ_ns{_ns_tag(ns_value)}.csv"


def _tmp_final_gcs_halo_path(work_dir: Path, hz0: int, ns_tag: str) -> Path:
    return work_dir / f"finalGCs_halo{int(hz0)}_ns{ns_tag}.tmp.dat"


def _tmp_depos_halo_path(work_dir: Path, hz0: int, ns_tag: str) -> Path:
    return work_dir / f"depos_halo{int(hz0)}_ns{ns_tag}.tmp.dat"


def _parse_ns_values(text: str) -> List[float]:
    out: List[float] = []
    for token in text.split(","):
        tok = token.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("No valid N_s values were provided.")
    return out


def _parse_extra_out_z_list(text: str) -> List[float]:
    out: List[float] = []
    seen: set[float] = set()
    for token in text.split(","):
        tok = token.strip()
        if not tok:
            continue
        value = float(tok)
        if (not math.isfinite(value)) or value < 0.0:
            raise ValueError(f"Invalid extra output redshift: {tok}")
        if abs(value) < 1.0e-12:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    out.sort()
    return out


def _clear_dir_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _check_project_layout(
    *,
    plot_gao2023_requested: bool,
    plot_choksi2018_requested: bool,
    plot_neumayer2020_requested: bool,
    plot_kong2026_requested: bool,
    tree_dir: Path | None,
) -> tuple[Path, Path]:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing data directory: {DATA_DIR}"
        )
    if not SNAPS_PATH.is_file():
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing snapshot-redshift table: {SNAPS_PATH}"
        )
    if not MASS_LOSS_PATH.is_file():
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing stellar-evolution mass-loss table: {MASS_LOSS_PATH}"
        )
    if plot_gao2023_requested and (not PLOT_GAO2023_PATH.is_file()):
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing Gao+2023 plot script: {PLOT_GAO2023_PATH}"
        )
    if plot_choksi2018_requested and (not PLOT_CHOKSI2018_PATH.is_file()):
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing Choksi+2018 plot script: {PLOT_CHOKSI2018_PATH}"
        )
    if plot_neumayer2020_requested and (not PLOT_NEUMAYER2020_PATH.is_file()):
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing Neumayer+2020 plot script: {PLOT_NEUMAYER2020_PATH}"
        )
    if plot_kong2026_requested and (not PLOT_KONG2026_PATH.is_file()):
        raise FileNotFoundError(
            f"Expected bundled High-z SMBHs repository layout under {PROJECT_ROOT}; "
            f"missing Kong+2026 plot script: {PLOT_KONG2026_PATH}"
        )
    effective_tree_dir = tree_dir.resolve() if tree_dir is not None else DEFAULT_TREE_DIR
    if not effective_tree_dir.is_dir():
        raise FileNotFoundError(
            f"Expected fixed-tree input directory not found: {effective_tree_dir}. "
            f"The runner uses the bundled repository layout under {PROJECT_ROOT}."
        )
    _tree_file_map(str(effective_tree_dir.resolve()))
    return DATA_DIR, effective_tree_dir


def _confirm_clear_output(path: Path) -> None:
    """Confirm clearing only when the output directory already has contents."""

    path.mkdir(parents=True, exist_ok=True)
    try:
        next(path.iterdir())
    except StopIteration:
        return

    prompt = (
        f"--clear-output will remove all existing contents under:\n"
        f"{path}\n"
        "Continue? [y/N]: "
    )
    try:
        reply = input(prompt).strip().lower()
    except EOFError as exc:
        raise SystemExit("Aborted: no confirmation received for --clear-output.") from exc
    if reply not in {"y", "yes"}:
        raise SystemExit("Aborted: output directory was not cleared.")


def _iter_numeric_text_lines(path: Path) -> Sequence[str]:
    """Yield non-comment, non-empty lines from a text table."""

    out: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            out.append(s)
    return out


def _format_combined_gcfin_row(hid: int, row: str, formation_row: np.ndarray | None = None) -> str:
    """Reformat one temporary per-halo GC row into the published finalGCs schema."""

    parts = row.split()
    if len(parts) < 8:
        raise ValueError(f"Expected at least 8 final GC columns, got {len(parts)} in row: {row}")

    gc_index_halo = int(float(parts[0]))
    status = int(float(parts[1]))
    m_final_msun = float(parts[2])
    log10_m_final_msun = math.log10(m_final_msun) if m_final_msun > 0.0 else -1.0
    m_init_msun = float(parts[3])
    lookback_time_final_gyr = float(parts[4])
    lookback_time_init_gyr = float(parts[5])
    r_final_kpc = float(parts[6])
    r_init_kpc = float(parts[7])
    feh = 0.0
    gc_radius_pc = 0.0
    sigma_h_msun_pc2 = 0.0
    imbh_mass_msun = 0.0

    if formation_row is not None:
        # The evolution code only knows about the compact GCini columns. The
        # merged public table restores birth-time GC properties from allcat.
        feh = float(formation_row[8])
        if len(formation_row) > 10:
            gc_radius_pc = float(formation_row[10])
        if len(formation_row) > 11:
            sigma_h_msun_pc2 = float(formation_row[11])
        if len(formation_row) > 12:
            imbh_mass_msun = float(formation_row[12])

    return (
        f"{hid:d} {gc_index_halo:d} {status:d} "
        f"{m_final_msun:.10e} {log10_m_final_msun:.10e} {m_init_msun:.10e} "
        f"{lookback_time_final_gyr:.10e} {lookback_time_init_gyr:.10e} "
        f"{r_final_kpc:.10e} {r_init_kpc:.10e} "
        f"{gc_radius_pc:.10e} {sigma_h_msun_pc2:.10e} {feh:.10e} {imbh_mass_msun:.10e}"
    )


def _combine_per_halo_outputs(
    per_halo_dir: Path,
    ns_output_dir: Path,
    ns_value: float,
    halo_ids: Sequence[int],
    all_rows: np.ndarray,
) -> None:
    """Merge temporary per-halo outputs for one N_s into the published files."""

    ns_tag = _ns_tag(ns_value)
    halo_ids_sorted = sorted({int(hid) for hid in halo_ids})
    hid_all = np.asarray(all_rows[:, 0], dtype=int)
    formation_rows_by_halo = {
        int(hid): np.asarray(all_rows[hid_all == int(hid)], dtype=float)
        for hid in halo_ids_sorted
    }

    gcfin_out = ns_output_dir / _final_gcs_ns_name(ns_value)
    depos_out = ns_output_dir / _depos_ns_name(ns_value)

    with gcfin_out.open("w", encoding="utf-8") as f_gcfin:
        f_gcfin.write("# " + COMBINED_FINAL_GC_HEADER.replace("\n", "\n# ") + "\n")
        for hid in halo_ids_sorted:
            src = _tmp_final_gcs_halo_path(per_halo_dir, hid, ns_tag)
            if not src.exists():
                raise FileNotFoundError(f"Missing per-halo GCfin file: {src}")
            halo_rows = formation_rows_by_halo.get(int(hid))
            if halo_rows is None:
                raise ValueError(f"Missing formation rows for halo {hid}.")
            for row in _iter_numeric_text_lines(src):
                parts = row.split()
                if len(parts) < 1:
                    raise ValueError(f"Malformed per-halo GCfin row in {src}: {row}")
                gc_index_halo = int(float(parts[0]))
                if gc_index_halo < 1 or gc_index_halo > len(halo_rows):
                    raise ValueError(
                        f"GC index {gc_index_halo} is out of bounds for halo {hid} "
                        f"with {len(halo_rows)} formation rows."
                    )
                # GC indices inside each temporary halo file are 1-based and
                # follow that halo's local allcat ordering.
                formation_row = halo_rows[gc_index_halo - 1]
                f_gcfin.write(_format_combined_gcfin_row(hid, row, formation_row=formation_row) + "\n")

    with depos_out.open("w", encoding="utf-8") as f_depos:
        f_depos.write("# " + COMBINED_DEPOS_HEADER.replace("\n", "\n# ") + "\n")
        for hid in halo_ids_sorted:
            src = _tmp_depos_halo_path(per_halo_dir, hid, ns_tag)
            if not src.exists():
                raise FileNotFoundError(f"Missing per-halo Depos file: {src}")
            for row in _iter_numeric_text_lines(src):
                f_depos.write(f"{hid:d} {row}\n")


def _combine_all_ns_outputs(output_dir: Path, ns_values: Sequence[float]) -> None:
    """Merge per-N_s combined GCfin/Depos files into one top-level file each."""

    gcfin_out = output_dir / "finalGCs_all.dat"
    depos_out = output_dir / "depos_all.dat"

    with gcfin_out.open("w", encoding="utf-8") as f_gcfin:
        f_gcfin.write("# " + GLOBAL_FINAL_GC_HEADER.replace("\n", "\n# ") + "\n")
        for ns in ns_values:
            ns_tag = _ns_tag(ns)
            src = output_dir / f"ns{ns_tag}" / _final_gcs_ns_name(ns)
            if not src.exists():
                raise FileNotFoundError(f"Missing per-N_s combined GCfin file: {src}")
            for row in _iter_numeric_text_lines(src):
                f_gcfin.write(f"{float(ns):.1f} {row}\n")

    with depos_out.open("w", encoding="utf-8") as f_depos:
        f_depos.write("# " + GLOBAL_DEPOS_HEADER.replace("\n", "\n# ") + "\n")
        for ns in ns_values:
            ns_tag = _ns_tag(ns)
            src = output_dir / f"ns{ns_tag}" / _depos_ns_name(ns)
            if not src.exists():
                raise FileNotFoundError(f"Missing per-N_s combined Depos file: {src}")
            for row in _iter_numeric_text_lines(src):
                f_depos.write(f"{float(ns):.1f} {row}\n")


def _build_halo_summary_table(
    all_rows: np.ndarray,
    status: np.ndarray,
    m_final: np.ndarray,
) -> pd.DataFrame:
    """Build one halo-level summary table, including the SMBH estimate."""

    hid = np.asarray(all_rows[:, 0], dtype=int)
    logmh_z0 = np.asarray(all_rows[:, 1], dtype=float)
    m_init = np.power(10.0, np.asarray(all_rows[:, 6], dtype=float))
    imbh_mass = np.asarray(all_rows[:, 12], dtype=float) if all_rows.shape[1] > 12 else np.zeros(len(all_rows))
    status = np.asarray(status, dtype=int)
    m_final = np.asarray(m_final, dtype=float)

    rows: List[Dict[str, float | int]] = []
    for hid0 in np.unique(hid):
        idx = hid == int(hid0)
        s = status[idx]
        imbh = imbh_mass[idx]
        n_sunk_gc = int(np.sum(s == -3))
        n_sunk_wanderer = int(np.sum(s == -5))
        m_smbh_gc_sunk = float(np.sum(imbh[s == -3]))
        m_smbh_wanderer_sunk = float(np.sum(imbh[s == -5]))
        rows.append(
            {
                "hid_z0": int(hid0),
                "logMh_z0": float(logmh_z0[idx][0]),
                "n_gc_total": int(np.sum(idx)),
                "n_alive": int(np.sum(s == 1)),
                "n_wanderer": int(np.sum(s == -4)),
                "n_exhausted": int(np.sum(s == -1)),
                "n_torn": int(np.sum(s == -2)),
                "n_sunk_gc": n_sunk_gc,
                "n_sunk_wanderer": n_sunk_wanderer,
                "n_sunk": n_sunk_gc + n_sunk_wanderer,
                "m_gc_init_total_msun": float(np.sum(m_init[idx])),
                "m_gc_final_total_msun": float(np.sum(m_final[idx])),
                "m_imbh_seed_total_msun": float(np.sum(imbh)),
                "m_smbh_gc_sunk_msun": m_smbh_gc_sunk,
                "m_smbh_wanderer_sunk_msun": m_smbh_wanderer_sunk,
                "m_smbh_est_msun": m_smbh_gc_sunk + m_smbh_wanderer_sunk,
            }
        )

    out = pd.DataFrame(rows, columns=HALO_SUMMARY_COLUMNS)
    if len(out) == 0:
        return out
    out.sort_values("hid_z0", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _interpolate_mpb_logmh_at_redshift(mpb_rows: np.ndarray, z_out: float) -> tuple[float, int]:
    """Interpolate MPB halo mass at the requested redshift in linear mass."""

    rows = np.asarray(mpb_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 6:
        return np.nan, 0

    redshift = rows[:, 5]
    logmh = rows[:, 0]
    valid = np.isfinite(redshift) & np.isfinite(logmh)
    if not np.any(valid):
        return np.nan, 0

    redshift = redshift[valid]
    logmh = logmh[valid]
    z_value = float(z_out)
    z_min = float(np.min(redshift))
    z_max = float(np.max(redshift))
    tol = 1.0e-10
    if z_value < z_min - tol or z_value > z_max + tol:
        return np.nan, 0

    cosmic_time = np.array([Redshift2CosmicTimeGyr(float(z)) for z in redshift], dtype=float)
    mass = np.power(10.0, logmh)
    valid = np.isfinite(cosmic_time) & np.isfinite(mass) & (mass > 0.0)
    if not np.any(valid):
        return np.nan, 0

    cosmic_time = cosmic_time[valid]
    mass = mass[valid]
    order = np.argsort(cosmic_time, kind="mergesort")
    cosmic_time = cosmic_time[order]
    mass = mass[order]
    keep_last_duplicate = np.r_[cosmic_time[1:] != cosmic_time[:-1], True]
    unique_time = cosmic_time[keep_last_duplicate]
    mass = mass[keep_last_duplicate]
    if len(unique_time) == 0:
        return np.nan, 0

    target_time = float(Redshift2CosmicTimeGyr(z_value))
    if target_time < float(unique_time[0]) - tol or target_time > float(unique_time[-1]) + tol:
        return np.nan, 0

    interp_mass = float(np.interp(target_time, unique_time, mass))
    if not np.isfinite(interp_mass) or interp_mass <= 0.0:
        return np.nan, 0
    return float(np.log10(interp_mass)), 1


def _build_halo_summary_by_z_table(
    all_rows: np.ndarray,
    status: np.ndarray,
    lookback_time_final_gyr: np.ndarray,
    extra_out_redshifts: Sequence[float],
    tree_dir: Path,
) -> pd.DataFrame:
    """Build one long-format halo summary table across requested output redshifts."""

    hid = np.asarray(all_rows[:, 0], dtype=int)
    imbh_mass = np.asarray(all_rows[:, 12], dtype=float) if all_rows.shape[1] > 12 else np.zeros(len(all_rows))
    status = np.asarray(status, dtype=int)
    lookback_time_final_gyr = np.asarray(lookback_time_final_gyr, dtype=float)
    t_z0 = float(Redshift2CosmicTimeGyr(0.0))
    output_redshifts = [0.0] + [float(z) for z in extra_out_redshifts]
    unique_hids = np.unique(hid)
    mpb_by_halo = {
        int(hid0): read_haloevo_mpb(_tree_file_for_halo(tree_dir, int(hid0)))
        for hid0 in unique_hids
    }

    rows: List[Dict[str, float | int]] = []
    for z_out in output_redshifts:
        lookback_to_z0_gyr = max(t_z0 - float(Redshift2CosmicTimeGyr(float(z_out))), 0.0)
        for hid0 in unique_hids:
            idx = hid == int(hid0)
            s = status[idx]
            imbh = imbh_mass[idx]
            lookback = lookback_time_final_gyr[idx]
            sunk_gc = (s == -3) & (lookback >= lookback_to_z0_gyr)
            sunk_wanderer = (s == -5) & (lookback >= lookback_to_z0_gyr)
            m_smbh_gc_sunk = float(np.sum(imbh[sunk_gc]))
            m_smbh_wanderer_sunk = float(np.sum(imbh[sunk_wanderer]))
            logmh_z, halo_mass_available = _interpolate_mpb_logmh_at_redshift(
                mpb_by_halo[int(hid0)],
                float(z_out),
            )
            rows.append(
                {
                    "hid_z0": int(hid0),
                    "z_out": float(z_out),
                    "lookback_to_z0_gyr": float(lookback_to_z0_gyr),
                    "halo_mass_available": int(halo_mass_available),
                    "logMh_z_msun": float(logmh_z),
                    "m_smbh_gc_sunk_msun": m_smbh_gc_sunk,
                    "m_smbh_wanderer_sunk_msun": m_smbh_wanderer_sunk,
                    "m_smbh_est_msun": m_smbh_gc_sunk + m_smbh_wanderer_sunk,
                }
            )

    out = pd.DataFrame(rows, columns=HALO_SUMMARY_BY_Z_COLUMNS)
    if len(out) == 0:
        return out
    out.sort_values(["hid_z0", "z_out"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _build_snap_map(snaps2redshifts_path: Path) -> np.ndarray:
    """Load the snapshot->redshift lookup used across the workflow."""

    z = np.loadtxt(snaps2redshifts_path, comments="#", ndmin=1)
    return np.asarray(z, dtype=float).reshape(-1)


def _nearest_snap(z_form: np.ndarray, z_snap: np.ndarray) -> np.ndarray:
    """Map formation redshifts onto the nearest discrete simulation snapshot."""

    out = np.empty(len(z_form), dtype=int)
    for i, z in enumerate(z_form):
        out[i] = int(np.argmin(np.abs(z_snap - z)))
    return out


def _legacy_tree_file_map(tree_dir: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for path in sorted(tree_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".txt", ".dat"):
            continue
        try:
            hid = int(path.stem)
        except ValueError:
            continue
        mapping[int(hid)] = str(path)
    return mapping


@lru_cache(maxsize=None)
def _tree_file_map(tree_dir_str: str) -> Dict[int, str]:
    tree_dir = Path(tree_dir_str)
    lookup_path = tree_dir / TREE_LOOKUP_BASENAME
    if lookup_path.is_file():
        mapping: Dict[int, str] = {}
        with lookup_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    hid = int(row["halo_id_z0"])
                    basename = row["fixed_tree_basename"].strip()
                except (KeyError, ValueError) as exc:
                    raise RuntimeError(f"Malformed tree lookup row in {lookup_path}: {row}") from exc
                path = tree_dir / basename
                if not path.is_file():
                    raise FileNotFoundError(f"Tree lookup references missing fixed tree: {path}")
                if hid in mapping:
                    raise RuntimeError(f"Duplicate halo_id_z0 {hid} in tree lookup {lookup_path}")
                mapping[hid] = str(path)
        if not mapping:
            raise RuntimeError(f"Tree lookup exists but contains no usable rows: {lookup_path}")
        return mapping

    mapping = _legacy_tree_file_map(tree_dir)
    if not mapping:
        raise RuntimeError(f"No usable fixed-tree files were found under {tree_dir}")
    return mapping


def _tree_file_for_halo(tree_dir: Path, halo_id: int) -> Path:
    hid = int(halo_id)
    mapping = _tree_file_map(str(tree_dir.resolve()))
    try:
        return Path(mapping[hid])
    except KeyError as exc:
        raise FileNotFoundError(f"Missing tree file for halo {hid} under {tree_dir}") from exc


def _build_ismpb_flags(all_rows: np.ndarray, tree_dir: Path) -> np.ndarray:
    """Map each formed GC to MPB/non-MPB using its formation subhalo ID."""

    hid = all_rows[:, 0].astype(int)
    form_id = all_rows[:, 2].astype(np.int64)
    flags = np.zeros(len(all_rows), dtype=int)

    for hz0 in np.unique(hid):
        try:
            tfile = _tree_file_for_halo(tree_dir, int(hz0))
        except FileNotFoundError:
            continue

        # Main branch ID in these trees corresponds to the "main leaf ID" column.
        mpbi = None
        mpb_ids = set()
        with tfile.open("r") as f:
            for line in f:
                s = line.strip()
                if (not s) or s.startswith("#") or s.lower().startswith("logmh"):
                    continue
                parts = s.split()
                if len(parts) < 4:
                    continue
                try:
                    subid = int(parts[2])
                    mpi = int(parts[3])
                except ValueError:
                    continue
                if mpbi is None:
                    mpbi = mpi
                if mpi == mpbi:
                    mpb_ids.add(subid)

        idx = np.where(hid == hz0)[0]
        if len(mpb_ids) == 0:
            # Fallback: keep rows active rather than dropping all in this halo.
            flags[idx] = 1
        else:
            flags[idx] = np.array([1 if int(form_id[j]) in mpb_ids else 0 for j in idx], dtype=int)

    return flags


def _stable_unique_halo_ids(halo_ids: np.ndarray) -> List[int]:
    ordered: List[int] = []
    seen: set[int] = set()
    for hid in np.asarray(halo_ids, dtype=int):
        hid_int = int(hid)
        if hid_int in seen:
            continue
        seen.add(hid_int)
        ordered.append(hid_int)
    return ordered


def _write_halo_tree_lookup(output_dir: Path, tree_dir: Path, halo_ids: np.ndarray) -> None:
    lookup_path = tree_dir / TREE_LOOKUP_BASENAME
    if not lookup_path.is_file():
        return

    rows_by_halo: Dict[int, Dict[str, str]] = {}
    with lookup_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                hid = int(row["halo_id_z0"])
            except (KeyError, ValueError) as exc:
                raise RuntimeError(f"Malformed tree lookup row in {lookup_path}: {row}") from exc
            if hid in rows_by_halo:
                raise RuntimeError(f"Duplicate halo_id_z0 {hid} in tree lookup {lookup_path}")
            rows_by_halo[hid] = dict(row)

    ordered_halo_ids = _stable_unique_halo_ids(halo_ids)
    out_rows = []
    for hid in ordered_halo_ids:
        try:
            row = rows_by_halo[int(hid)]
        except KeyError as exc:
            raise FileNotFoundError(
                f"Tree lookup {lookup_path} is missing halo_id_z0={hid}, which entered the simulation output."
            ) from exc
        out_rows.append(
            {
                "hid_z0": int(hid),
                "simulation_key": row["simulation_key"].strip(),
                "simulation": row["simulation"].strip(),
                "subhalo_id_z0": int(row["subhalo_id_z0"]),
                "fixed_tree_basename": row["fixed_tree_basename"].strip(),
                "file_index": int(row["file_index"]),
            }
        )

    pd.DataFrame(out_rows, columns=[
        "hid_z0",
        "simulation_key",
        "simulation",
        "subhalo_id_z0",
        "fixed_tree_basename",
        "file_index",
    ]).to_csv(output_dir / HALO_TREE_LOOKUP_NAME, index=False)


def _build_mpb_csv_from_trees(tree_dir: Path, halo_ids: np.ndarray, z_snap: np.ndarray, out_csv: Path) -> None:
    """Flatten the fixed trees into the compact MPB table used by plotting.

    The plotting script only needs the host id, snapshot number, halo mass, and
    spin vector, so this CSV is much smaller than re-reading the full trees
    every time figures are generated.
    """

    rows: List[Dict[str, float]] = []
    for hid in np.unique(halo_ids.astype(int)):
        try:
            tfile = _tree_file_for_halo(tree_dir, int(hid))
        except FileNotFoundError:
            continue
        with tfile.open("r") as f:
            for line in f:
                s = line.strip()
                if (not s) or s.startswith("#") or s.lower().startswith("logmh"):
                    continue
                parts = s.split()
                if len(parts) < 9:
                    continue
                try:
                    vals = [float(v) for v in parts[:9]]
                except ValueError:
                    continue
                z = vals[5]
                snap = int(np.argmin(np.abs(z_snap - z)))
                rows.append(
                    {
                        "subhalo_id_z0": int(hid),
                        "SnapNum": int(snap),
                        "Redshift": float(z),
                        "logMh_msun_h": float(vals[0]),
                        "SubhaloSpin_x": float(vals[6]),
                        "SubhaloSpin_y": float(vals[7]),
                        "SubhaloSpin_z": float(vals[8]),
                    }
                )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError(f"No MPB rows were built from tree directory: {tree_dir}")
    df.sort_values(["subhalo_id_z0", "SnapNum"], ascending=[True, False], inplace=True)
    df.to_csv(out_csv, index=False)


def _read_main_spatial_all(path: Path) -> np.ndarray:
    """Read ``all_<Ns>.txt`` generated by ``main_spatial.py``.

    The maintained modern schema is exactly 13 columns:
    the legacy 10-column formation catalog plus fixed formation-time GC radius,
    surface density, and IMBH mass.
    """

    if len(_iter_numeric_text_lines(path)) == 0:
        raise ValueError(
            f"{path} contains no GC rows. main_spatial.py likely selected no halos in the requested "
            "descendant z=0 mass window, found no usable tree entries in the configured tree directory, "
            "or formed no GCs in the selected run."
        )
    arr = np.loadtxt(path, comments="#", ndmin=2)
    n_expected = 13
    if arr.ndim != 2 or arr.shape[1] != n_expected:
        raise ValueError(f"{path} must have exactly {n_expected} columns; got shape={arr.shape}")
    return arr.astype(float, copy=False)


def _stable_row_order(all_rows: np.ndarray) -> np.ndarray:
    """Build a deterministic sort order that is independent of filesystem order."""

    n = len(all_rows)
    df = pd.DataFrame(
        {
            "row": np.arange(n, dtype=int),
            "hid_z0": all_rows[:, 0].astype(int),
            "subfind_form": all_rows[:, 2].astype(np.int64),
            "logMh_form": np.round(all_rows[:, 3], 8),
            "logMstar_form": np.round(all_rows[:, 4], 8),
            "logMgas_form": np.round(all_rows[:, 5], 8),
            "logM_form": np.round(all_rows[:, 6], 8),
            "zform": np.round(all_rows[:, 7], 8),
            "feh": np.round(all_rows[:, 8], 8),
        }
    )
    sort_cols = [
        "hid_z0",
        "subfind_form",
        "zform",
        "logMh_form",
        "logMstar_form",
        "logMgas_form",
        "logM_form",
        "feh",
        "row",
    ]
    # mergesort keeps ordering stable for any exact key ties.
    return df.sort_values(sort_cols, kind="mergesort")["row"].to_numpy(dtype=int)


def _run_main_spatial_for_ns(
    stage_dir: Path,
    data_dir: Path,
    tree_dir: Path,
    ns_value: float,
    *,
    p2: float,
    p3: float,
    mpb_only: int,
    lg_cut_off_mass: float,
    metal: str,
    accreted_baryon: str,
    run_all: int,
    log_mh_min: float,
    log_mh_max: float,
    n_halos: int,
    imbh: int,
    quiet: bool,
) -> Path:
    ns_str = f"{float(ns_value):.1f}"
    log_path = stage_dir / f"main_spatial_ns{_ns_tag(ns_value)}.log"
    cmd = [
        sys.executable,
        str(MAIN_SPATIAL_PATH),
        ns_str,
        "--data-dir",
        str(data_dir),
        "--tree-dir",
        str(tree_dir),
        "--output-dir",
        str(stage_dir),
        "--p2",
        f"{float(p2):g}",
        "--p3",
        f"{float(p3):g}",
        "--mpb-only",
        str(int(mpb_only)),
        "--lg_cut-off_mass",
        f"{float(lg_cut_off_mass):g}",
        "--metal",
        str(metal),
        "--accreted_baryon",
        str(accreted_baryon),
        "--run-all",
        str(int(run_all)),
        "--log-mh-min",
        f"{float(log_mh_min):g}",
        "--log-mh-max",
        f"{float(log_mh_max):g}",
        "--n-halos",
        str(int(n_halos)),
    ]
    cmd.extend([
        "--IMBH",
        str(int(imbh)),
    ])
    with log_path.open("w") as logf:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, stdout=logf, stderr=subprocess.STDOUT)
    if not quiet:
        print(f"main_spatial finished for N_s={ns_str}. log={log_path}")
    all_path = stage_dir / f"all_{ns_str}.txt"
    if not all_path.exists():
        raise FileNotFoundError(f"Expected formation catalog not found: {all_path}")
    return all_path


def _default_plot_ns_value(ns_values: Sequence[float]) -> float:
    if any(abs(float(ns) - 2.0) < 1.0e-8 for ns in ns_values):
        return 2.0
    return float(ns_values[0])


def _run_plot_gao2023(
    *,
    output_dir: Path,
    ns_values: Sequence[float],
    p2: float,
    p3: float,
    quiet: bool,
) -> Path:
    """Run the Gao+2023 plot suite against the freshly written model outputs."""

    del p2, p3
    plot_output_dir = output_dir / "_plots_Gao+2023"
    ns_values_arg = ",".join(f"{float(ns):.1f}" for ns in ns_values)
    cmd = [
        sys.executable,
        str(PLOT_GAO2023_PATH),
        "--out_dir",
        str(output_dir),
        "--ns-values",
        ns_values_arg,
        "--plot_dir",
        str(plot_output_dir),
    ]
    if not quiet:
        print(f"plot_Gao+2023.py starting. output={plot_output_dir}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if not quiet:
        print(f"plot_Gao+2023.py finished. output={plot_output_dir}")
    return plot_output_dir


def _run_plot_choksi2018(
    *,
    output_dir: Path,
    ns_value: float,
    quiet: bool,
) -> Path:
    plot_output_dir = output_dir / "_plots_Choksi+2018"
    cmd = [
        sys.executable,
        str(PLOT_CHOKSI2018_PATH),
        "--out_dir",
        str(output_dir),
        "--plot_dir",
        str(plot_output_dir),
        "--ns-value",
        f"{float(ns_value):.1f}",
    ]
    if not quiet:
        print(f"plot_Choksi+2018.py starting. output={plot_output_dir}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if not quiet:
        print(f"plot_Choksi+2018.py finished. output={plot_output_dir}")
    return plot_output_dir


def _run_plot_neumayer2020(
    *,
    output_dir: Path,
    ns_value: float,
    quiet: bool,
) -> Path:
    plot_output_dir = output_dir / "_plots_Neumayer+2020"
    cmd = [
        sys.executable,
        str(PLOT_NEUMAYER2020_PATH),
        "--out_dir",
        str(output_dir),
        "--plot_dir",
        str(plot_output_dir),
        "--ns-value",
        f"{float(ns_value):.1f}",
    ]
    if not quiet:
        print(f"plot_Neumayer+2020.py starting. output={plot_output_dir}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if not quiet:
        print(f"plot_Neumayer+2020.py finished. output={plot_output_dir}")
    return plot_output_dir


def _run_plot_kong2026(
    *,
    output_dir: Path,
    ns_value: float,
    quiet: bool,
) -> Path:
    plot_output_dir = output_dir / "_plots_Kong+2026"
    cmd = [
        sys.executable,
        str(PLOT_KONG2026_PATH),
        "--out_dir",
        str(output_dir),
        "--plot_dir",
        str(plot_output_dir),
        "--ns-value",
        f"{float(ns_value):.1f}",
    ]
    if not quiet:
        print(f"plot_Kong+2026.py starting. output={plot_output_dir}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if not quiet:
        print(f"plot_Kong+2026.py finished. output={plot_output_dir}")
    return plot_output_dir


def _build_allcat_table(
    all_rows: np.ndarray,
    *,
    tree_dir: Path,
    z_snap: np.ndarray,
) -> np.ndarray:
    """Assemble the plotting-facing allcat schema from main_spatial output."""

    hid_z0 = all_rows[:, 0].astype(int)
    logmh_z0 = all_rows[:, 1].astype(float)
    subfind_form = all_rows[:, 2].astype(np.int64)
    logmh_form = all_rows[:, 3].astype(float)
    logmstar_form = all_rows[:, 4].astype(float)
    logm_form = all_rows[:, 6].astype(float)
    z_form = all_rows[:, 7].astype(float)
    feh = all_rows[:, 8].astype(float)
    r_init = all_rows[:, 9].astype(float)
    gc_radius_pc = all_rows[:, 10].astype(float)
    sigma_h_msun_pc2 = all_rows[:, 11].astype(float)
    imbh_mass_msun = all_rows[:, 12].astype(float)

    logmstar_z0 = np.log10([smhm.SMHM(10.0 ** m, 0.0, scatter=False) for m in logmh_z0])
    snap_form = _nearest_snap(z_form, z_snap)
    is_mpb = _build_ismpb_flags(all_rows, tree_dir)

    return np.column_stack([
        hid_z0.astype(float),
        logmh_z0,
        logmstar_z0,
        logmh_form,
        logmstar_form,
        logm_form,
        z_form,
        feh,
        is_mpb.astype(float),
        subfind_form.astype(float),
        snap_form.astype(float),
        r_init,
        gc_radius_pc,
        sigma_h_msun_pc2,
        imbh_mass_msun,])


def _evolve_one_halo_task(
    *,
    hz0: int,
    halo_rows: np.ndarray,
    ns: float,
    ns_tag: str,
    tmp_work_dir: str,
    tree_halo: str,
    ts_m: float,
    ts_r: float,
    df: int,
    tidal_stripping: str,
    verbose: bool,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Worker for one halo evolution.

    The per-halo GC evolution is embarrassingly parallel once the formation
    catalog has already been built. Each worker writes its own temporary GCini
    file and returns only the columns needed to assemble the final vectors.
    """

    tmp_work_dir_p = Path(tmp_work_dir)
    tree_halo_p = Path(tree_halo)

    gcini_halo = tmp_work_dir_p / f"gcini_halo{hz0}_ns{ns_tag}.txt"
    # The fast evolution code now reads the modern per-GC formation rows,
    # including the fixed IMBH seed mass used by the wanderer branch.
    np.savetxt(gcini_halo, halo_rows, fmt="%.10e", header=FINAL_GC_HEADER)

    depos_halo = _tmp_depos_halo_path(tmp_work_dir_p, hz0, ns_tag)
    gcfin_halo = _tmp_final_gcs_halo_path(tmp_work_dir_p, hz0, ns_tag)
    gcfin_arr, _ = evolve_single_halo(
        ts_m=ts_m,
        ts_r=ts_r,
        df=df,
        gcini_path=gcini_halo,
        depos_path=depos_halo,
        gcfin_path=gcfin_halo,
        haloevo_path=tree_halo_p,
        verbose=verbose,
        sersic_n=float(ns),
        final_redshift=0.0,
        tidal_stripping=tidal_stripping,
    )

    return (
        int(hz0),
        gcfin_arr[:, 1].astype(int),
        np.asarray(gcfin_arr[:, 2], dtype=float),
        np.asarray(gcfin_arr[:, 4], dtype=float),
        np.asarray(gcfin_arr[:, 6], dtype=float),)


def _run_single_ns_pipeline(
    *,
    ns: float,
    data_dir: Path,
    tree_dir: Path,
    output_dir: Path,
    stage_root: Path,
    tmp_gcini_root: Path,
    z_snap: np.ndarray,
    p2: float,
    p3: float,
    mpb_only: int,
    lg_cut_off_mass: float,
    metal: str,
    accreted_baryon: str,
    run_all: int,
    log_mh_min: float,
    log_mh_max: float,
    n_halos: int,
    imbh: int,
    ts_m: float,
    ts_r: float,
    df: int,
    tidal_stripping: str,
    extra_out_redshifts: Sequence[float],
    jobs: int,
    quiet: bool,
) -> tuple[float, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run formation + evolution for one Sersic index value.

    Each ``N_s`` gets its own isolated temporary output directory for
    ``main_spatial.py`` and its own temporary GCini directory. That avoids file
    collisions when multiple ``N_s`` values are processed concurrently, while
    still reading the raw data directly from the bundled project ``data/``.
    """

    ns_tag = _ns_tag(ns)
    p2_tag = _fmt_param_tag(p2)
    p3_tag = _fmt_param_tag(p3)
    ns_output_dir = _ns_output_dir(output_dir, ns)

    stage_dir = stage_root / f"ns{ns_tag}"
    tmp_gcini_dir = tmp_gcini_root / f"ns{ns_tag}"
    _clear_dir_contents(stage_dir)
    _clear_dir_contents(tmp_gcini_dir)

    all_path = _run_main_spatial_for_ns(
        stage_dir,
        data_dir,
        tree_dir,
        ns,
        p2=p2,
        p3=p3,
        mpb_only=mpb_only,
        lg_cut_off_mass=lg_cut_off_mass,
        metal=metal,
        accreted_baryon=accreted_baryon,
        run_all=run_all,
        log_mh_min=log_mh_min,
        log_mh_max=log_mh_max,
        n_halos=n_halos,
        imbh=imbh,
        quiet=quiet,
    )
    all_rows_raw = _read_main_spatial_all(all_path)
    row_order = _stable_row_order(all_rows_raw)
    # The raw all_<Ns>.txt order depends on legacy tree traversal and can vary
    # with filesystem order. Sorting once here makes later ns-to-ns comparisons
    # and merged output tables deterministic.
    all_rows = np.array(all_rows_raw[row_order], dtype=float, copy=True)
    invalid_initial_r = (~np.isfinite(all_rows[:, 9])) | (all_rows[:, 9] <= 0.0)
    if np.any(invalid_initial_r):
        raise ValueError(
            f"{all_path} contains {int(np.sum(invalid_initial_r))} invalid initial GC radii "
            "after formation-time validation."
        )

    hid_z0 = all_rows[:, 0].astype(int)
    m_final = np.zeros(len(all_rows), dtype=float)
    lookback_time_final = np.zeros(len(all_rows), dtype=float)
    r_final = -1.0 * np.ones(len(all_rows), dtype=float)
    status = np.zeros(len(all_rows), dtype=int)
    unique_halos = np.unique(hid_z0)
    halo_index_map = {int(hz0): np.where(hid_z0 == hz0)[0] for hz0 in unique_halos}
    jobs = max(1, int(jobs))

    if jobs == 1:
        for hz0 in unique_halos:
            idx = halo_index_map[int(hz0)]
            tree_halo = _tree_file_for_halo(tree_dir, int(hz0))
            if not quiet:
                print(f"N_s={ns_tag}: evolving halo {hz0} ({len(idx)} GCs)")
            hz0_ret, status_h, m_final_h, lookback_time_final_h, r_final_h = _evolve_one_halo_task(
                hz0=int(hz0),
                halo_rows=np.array(all_rows[idx, :], dtype=float, copy=True),
                ns=float(ns),
                ns_tag=ns_tag,
                tmp_work_dir=str(tmp_gcini_dir),
                tree_halo=str(tree_halo),
                ts_m=ts_m,
                ts_r=ts_r,
                df=df,
                tidal_stripping=tidal_stripping,
                verbose=not quiet,
            )
            status[idx] = status_h
            m_final[idx] = m_final_h
            lookback_time_final[idx] = lookback_time_final_h
            r_final[idx] = r_final_h
    else:
        max_workers = min(jobs, len(unique_halos))
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for hz0 in unique_halos:
                idx = halo_index_map[int(hz0)]
                tree_halo = _tree_file_for_halo(tree_dir, int(hz0))
                fut = ex.submit(
                    _evolve_one_halo_task,
                    hz0=int(hz0),
                    halo_rows=np.array(all_rows[idx, :], dtype=float, copy=True),
                    ns=float(ns),
                    ns_tag=ns_tag,
                    tmp_work_dir=str(tmp_gcini_dir),
                    tree_halo=str(tree_halo),
                    ts_m=ts_m,
                    ts_r=ts_r,
                    df=df,
                    tidal_stripping=tidal_stripping,
                    verbose=False,
                )
                futures[fut] = int(hz0)

            completed = 0
            for fut in as_completed(futures):
                hz0_ret, status_h, m_final_h, lookback_time_final_h, r_final_h = fut.result()
                idx = halo_index_map[hz0_ret]
                status[idx] = status_h
                m_final[idx] = m_final_h
                lookback_time_final[idx] = lookback_time_final_h
                r_final[idx] = r_final_h
                completed += 1
                if (not quiet) and (completed == 1 or completed % 10 == 0 or completed == len(unique_halos)):
                    print(f"N_s={ns_tag}: completed {completed}/{len(unique_halos)} halos")

    allcat = _build_allcat_table(
        all_rows,
        tree_dir=tree_dir,
        z_snap=z_snap,
    )
    allcat_ns_path = ns_output_dir / f"allcat_ns{ns_tag}_s-0_p2-{p2_tag}_p3-{p3_tag}.txt"
    np.savetxt(allcat_ns_path, allcat, fmt="%.6e", header=ALLCAT_HEADER)

    _combine_per_halo_outputs(
        per_halo_dir=tmp_gcini_dir,
        ns_output_dir=ns_output_dir,
        ns_value=ns,
        halo_ids=unique_halos,
        all_rows=all_rows,
    )

    summary_df = pd.DataFrame(
        {
            "ns": np.full(len(all_rows), float(ns)),
            "hid_z0": hid_z0.astype(int),
            "status": status.astype(int),
            "m_final_msun": m_final,
            "r_final_kpc": r_final,
        }
    )
    halo_summary_df = _build_halo_summary_table(all_rows=all_rows, status=status, m_final=m_final)
    halo_summary_df.to_csv(ns_output_dir / f"haloSummary_ns{ns_tag}.csv", index=False)
    halo_summary_by_z_df = _build_halo_summary_by_z_table(
        all_rows=all_rows,
        status=status,
        lookback_time_final_gyr=lookback_time_final,
        extra_out_redshifts=extra_out_redshifts,
        tree_dir=tree_dir,
    )
    halo_summary_by_z_df.to_csv(ns_output_dir / _halo_summary_by_z_ns_name(ns), index=False)
    return float(ns), allcat[:, 0].astype(int), summary_df, halo_summary_df, halo_summary_by_z_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Run the High-z SMBHs Python GC pipeline using the bundled repository src/ and data/ layout, with an optional fixed-tree directory override."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--output", type=Path, default=Path("/lingshan/disk3/subonan/_outputs/Gao+2023"), help="Output directory.")
    parser.add_argument("--tree-dir", type=Path, default=None, help="Optional fixed-tree input directory. Defaults to the bundled data/fixed_trees_large_spin in this repository.")
    parser.add_argument("--clear-output", action="store_true", help="Clear output directory before writing.")
    parser.add_argument(
        "--ns-values",
        type=str,
        default=",".join(str(v) for v in NS_VALUES_DEFAULT),
        help="Comma-separated N_s values to run.",
    )

    # Physics/evolution controls used by the active Python GCevo rewrite.
    parser.add_argument("--ts-m", type=float, default=0.5, help="adaptive mass-loss timestep factor for evo")
    parser.add_argument("--ts-r", type=float, default=0.5, help="adaptive orbital-decay timestep factor for evo")
    parser.add_argument("--DF", dest="df", type=int, choices=[0, 1], default=1, help="if 1, enable dynamical-friction orbital decay; if 0, disable DF radial inspiral")
    parser.add_argument("--tidal_stripping", choices=TIDAL_STRIPPING_CHOICES, default=TIDAL_STRIPPING_FRAGIONE2019, help="continuous tidal-stripping prescription; default keeps the current Fragione+2019 local-orbit model")
    parser.add_argument(
        "--extra_out_z_list",
        type=str,
        default=EXTRA_OUT_Z_LIST_DEFAULT,
        help=("comma-separated extra redshifts for halo-level sunk-BH summaries only; "
              "the simulation itself always runs to z=0 and z=0 is always included automatically"),
    )

    # Formation-model parameters passed directly to main_spatial.py.
    parser.add_argument("--p2", type=float, default=6.75, help="GC formation-efficiency normalization in M_GC = 3e-5 * p2 * M_gas / f_b")
    parser.add_argument("--p3", type=float, default=0.5, help="threshold in ((Delta M_h / M_h) / Delta t) above which a GC formation event is triggered")
    parser.add_argument("--mpb-only", type=int, default=0, help="if 1, only use the main progenitor branch; if 0, include all retained branches")
    parser.add_argument("--lg_cut-off_mass", dest="lg_cut_off_mass", type=float, default=12.0, help="log10 Schechter cutoff mass Mc in Msun for the GC initial mass function")
    parser.add_argument("--metal", choices=METAL_CHOICES, default=METAL_CHOKSI2018, help="stellar mass-metallicity relation used at GC formation")
    parser.add_argument("--accreted_baryon", choices=ACCRETED_BARYON_CHOICES, default=ACCRETED_BARYON_MURATOV_GNEDIN2010, help="accreted-baryon fraction limiter used when computing cold gas mass")
    parser.add_argument("--run-all", type=int, default=1, help="if 1, process all halos in the tree set; if 0, apply the mass window and halo count below")
    parser.add_argument("--log-mh-min", type=float, default=11.5, help="minimum descendant z=0 host-halo log mass when --run-all=0")
    parser.add_argument("--log-mh-max", type=float, default=12.5, help="maximum descendant z=0 host-halo log mass when --run-all=0")
    parser.add_argument("--n-halos", type=int, default=10, help="maximum number of halos to run when --run-all=0")
    parser.add_argument("--IMBH", type=int, choices=[0, 1], default=1, help="enable the IMBH seeding module in main_spatial if 1, otherwise write zero IMBH-related columns")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel halo-evolution workers per N_s run.")
    parser.add_argument("--ns-jobs", type=int, default=1, help="Concurrent N_s pipelines.")
    parser.add_argument(
        "--plot_Gao+2023",
        dest="plot_gao2023",
        action="store_true",
        help="Run my/plot_Gao+2023.py automatically after the simulation and write figures to <output>/_plots_Gao+2023.",
    )
    parser.add_argument(
        "--plot_Choksi+2018",
        dest="plot_choksi2018",
        action="store_true",
        help="Run my/plot_Choksi+2018.py automatically after the simulation and write figures to <output>/_plots_Choksi+2018.",
    )
    parser.add_argument(
        "--plot_Neumayer+2020",
        dest="plot_neumayer2020",
        action="store_true",
        help="Run my/plot_Neumayer+2020.py automatically after the simulation and write figures to <output>/_plots_Neumayer+2020.",
    )
    parser.add_argument(
        "--plot_Kong+2026",
        dest="plot_kong2026",
        action="store_true",
        help="Run my/plot_Kong+2026.py automatically after the simulation and write figures to <output>/_plots_Kong+2026.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging.")
    args = parser.parse_args()

    data_dir, tree_dir = _check_project_layout(
        plot_gao2023_requested=bool(args.plot_gao2023),
        plot_choksi2018_requested=bool(args.plot_choksi2018),
        plot_neumayer2020_requested=bool(args.plot_neumayer2020),
        plot_kong2026_requested=bool(args.plot_kong2026),
        tree_dir=args.tree_dir,
    )

    output_dir = args.output.resolve()
    if args.clear_output:
        _confirm_clear_output(output_dir)
        _clear_dir_contents(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    ns_values = _parse_ns_values(args.ns_values)
    extra_out_redshifts = _parse_extra_out_z_list(args.extra_out_z_list)
    z_snap = _build_snap_map(SNAPS_PATH)

    # These directories are only transient working areas for the formation
    # and per-halo evolution stages. They no longer contain any copied or
    # linked copies of the raw Gao+2023 input data, and are always removed.
    stage_root_keeper = tempfile.TemporaryDirectory(prefix="gao2023_main_spatial_")
    tmp_gcini_keeper = tempfile.TemporaryDirectory(prefix="gao2023_gcini_")
    stage_root = Path(stage_root_keeper.name)
    tmp_gcini_root = Path(tmp_gcini_keeper.name)

    p2_tag = _fmt_param_tag(args.p2)
    p3_tag = _fmt_param_tag(args.p3)

    try:
        t0 = time.time()
        summary_parts: List[pd.DataFrame] = []
        halo_summary_parts: List[pd.DataFrame] = []
        halo_summary_by_z_parts: List[pd.DataFrame] = []
        template_halo_ids: np.ndarray | None = None
        ns_jobs = max(1, int(args.ns_jobs))
        ns_results: Dict[float, tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

        if ns_jobs == 1 or len(ns_values) == 1:
            for ns in ns_values:
                ns_ret, halo_ids_ret, summary_df_ret, halo_summary_df_ret, halo_summary_by_z_df_ret = _run_single_ns_pipeline(
                    ns=ns,
                    data_dir=data_dir,
                    tree_dir=tree_dir,
                    output_dir=output_dir,
                    stage_root=stage_root,
                    tmp_gcini_root=tmp_gcini_root,
                    z_snap=z_snap,
                    p2=args.p2,
                    p3=args.p3,
                    mpb_only=args.mpb_only,
                    lg_cut_off_mass=args.lg_cut_off_mass,
                    metal=args.metal,
                    accreted_baryon=args.accreted_baryon,
                    run_all=args.run_all,
                    log_mh_min=args.log_mh_min,
                    log_mh_max=args.log_mh_max,
                    n_halos=args.n_halos,
                    imbh=args.IMBH,
                    ts_m=args.ts_m,
                    ts_r=args.ts_r,
                    df=args.df,
                    tidal_stripping=args.tidal_stripping,
                    extra_out_redshifts=extra_out_redshifts,
                    jobs=args.jobs,
                    quiet=args.quiet,
                )
                ns_results[ns_ret] = (halo_ids_ret, summary_df_ret, halo_summary_df_ret, halo_summary_by_z_df_ret)
        else:
            max_ns_workers = min(ns_jobs, len(ns_values))
            if (not args.quiet) and args.jobs > 1:
                print(
                    "Running concurrent N_s pipelines with nested halo workers: "
                    f"ns_jobs={max_ns_workers}, halo_jobs={max(1, int(args.jobs))}, "
                    f"max_processes~{max_ns_workers * max(1, int(args.jobs))}"
                )
            futures = {}
            with ThreadPoolExecutor(max_workers=max_ns_workers) as ex:
                for ns in ns_values:
                    fut = ex.submit(
                        _run_single_ns_pipeline,
                        ns=ns,
                        data_dir=data_dir,
                        tree_dir=tree_dir,
                        output_dir=output_dir,
                        stage_root=stage_root,
                        tmp_gcini_root=tmp_gcini_root,
                        z_snap=z_snap,
                        p2=args.p2,
                        p3=args.p3,
                        mpb_only=args.mpb_only,
                        lg_cut_off_mass=args.lg_cut_off_mass,
                        metal=args.metal,
                        accreted_baryon=args.accreted_baryon,
                        run_all=args.run_all,
                        log_mh_min=args.log_mh_min,
                        log_mh_max=args.log_mh_max,
                        n_halos=args.n_halos,
                        imbh=args.IMBH,
                        ts_m=args.ts_m,
                        ts_r=args.ts_r,
                        df=args.df,
                        tidal_stripping=args.tidal_stripping,
                        extra_out_redshifts=extra_out_redshifts,
                        jobs=args.jobs,
                        quiet=args.quiet,
                    )
                    futures[fut] = float(ns)

                completed = 0
                for fut in as_completed(futures):
                    ns_ret, halo_ids_ret, summary_df_ret, halo_summary_df_ret, halo_summary_by_z_df_ret = fut.result()
                    ns_results[ns_ret] = (halo_ids_ret, summary_df_ret, halo_summary_df_ret, halo_summary_by_z_df_ret)
                    completed += 1
                    if not args.quiet:
                        print(f"N_s batch: completed {completed}/{len(ns_values)} N_s runs")

        for ns in ns_values:
            halo_ids_ret, summary_df_ret, halo_summary_df_ret, halo_summary_by_z_df_ret = ns_results[float(ns)]
            summary_parts.append(summary_df_ret)
            halo_summary_parts.append(halo_summary_df_ret.assign(ns=float(ns)))
            halo_summary_by_z_parts.append(halo_summary_by_z_df_ret.assign(ns=float(ns)))
            if template_halo_ids is None:
                template_halo_ids = halo_ids_ret
                template_ns_tag = _ns_tag(ns)
                template_ns_path = (
                    output_dir
                    / f"ns{template_ns_tag}"
                    / f"allcat_ns{template_ns_tag}_s-0_p2-{p2_tag}_p3-{p3_tag}.txt"
                )
                template_allcat = np.loadtxt(template_ns_path, ndmin=2)
                # Keep one top-level allcat template for downstream tools that
                # accept the historical single-file entry point and then infer
                # the per-N_s directories from it.
                template_path = output_dir / f"allcat_s-0_p2-{p2_tag}_p3-{p3_tag}.txt"
                np.savetxt(template_path, template_allcat, fmt="%.6e", header=ALLCAT_HEADER)

        if template_halo_ids is None:
            raise RuntimeError("No catalogs were produced; check input trees and model parameters.")

        _build_mpb_csv_from_trees(
            tree_dir=tree_dir,
            halo_ids=template_halo_ids,
            z_snap=z_snap,
            out_csv=output_dir / "mpb_from_fixed_trees.csv",
        )
        _write_halo_tree_lookup(
            output_dir=output_dir,
            tree_dir=tree_dir,
            halo_ids=template_halo_ids,
        )

        _combine_all_ns_outputs(output_dir=output_dir, ns_values=ns_values)

        summary = pd.concat(summary_parts, ignore_index=True)
        summary.to_csv(output_dir / "python_evo_summary.csv", index=False)
        halo_summary = pd.concat(halo_summary_parts, ignore_index=True)
        halo_summary.to_csv(output_dir / "haloSummary_all.csv", index=False)
        halo_summary_by_z = pd.concat(halo_summary_by_z_parts, ignore_index=True)
        halo_summary_by_z.to_csv(output_dir / "haloSummaryByZ_all.csv", index=False)
        with (output_dir / RUN_METADATA_NAME).open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "tree_dir": str(tree_dir.resolve()),
                    "final_redshift": 0.0,
                    "extra_out_z_list": [float(z) for z in extra_out_redshifts],
                    "output_redshifts": [0.0] + [float(z) for z in extra_out_redshifts],
                    "ts_m": float(args.ts_m),
                    "ts_r": float(args.ts_r),
                    "DF": int(args.df),
                    "tidal_stripping": args.tidal_stripping,
                    "p2": float(args.p2),
                    "p3": float(args.p3),
                    "lg_cut_off_mass": float(args.lg_cut_off_mass),
                    "metal": args.metal,
                    "accreted_baryon": args.accreted_baryon,
                    "IMBH": int(args.IMBH),
                    "mpb_only": int(args.mpb_only),
                    "run_all": int(args.run_all),
                    "log_mh_min": float(args.log_mh_min),
                    "log_mh_max": float(args.log_mh_max),
                    "n_halos": int(args.n_halos),
                    "ns_values": [float(v) for v in ns_values],
                },
                f,
                indent=2,
                sort_keys=True,
            )

        plot_outputs: List[Path] = []
        plot_ns_value = _default_plot_ns_value(ns_values)
        if args.plot_gao2023:
            plot_outputs.append(_run_plot_gao2023(
                output_dir=output_dir,
                ns_values=ns_values,
                p2=args.p2,
                p3=args.p3,
                quiet=args.quiet,
            ))
        if args.plot_choksi2018:
            plot_outputs.append(_run_plot_choksi2018(
                output_dir=output_dir,
                ns_value=plot_ns_value,
                quiet=args.quiet,
            ))
        if args.plot_neumayer2020:
            plot_outputs.append(_run_plot_neumayer2020(
                output_dir=output_dir,
                ns_value=plot_ns_value,
                quiet=args.quiet,
            ))
        if args.plot_kong2026:
            plot_outputs.append(_run_plot_kong2026(
                output_dir=output_dir,
                ns_value=plot_ns_value,
                quiet=args.quiet,
            ))

        elapsed = time.time() - t0
        print(
            "DONE "
            f"ns={len(ns_values)} "
            f"halos={len(np.unique(template_halo_ids))} "
            f"rows_per_ns={len(template_halo_ids)} "
            f"elapsed_s={elapsed:.2f}"
        )
        print(f"OUTPUT {output_dir}")
        for plot_output_dir in plot_outputs:
            print(f"PLOTS {plot_output_dir}")
    finally:
        if tmp_gcini_keeper is not None:
            tmp_gcini_keeper.cleanup()
        if stage_root_keeper is not None:
            stage_root_keeper.cleanup()


if __name__ == "__main__":
    main()
