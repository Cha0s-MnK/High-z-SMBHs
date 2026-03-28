#!/usr/bin/env python3

"""
Batch driver for the Python rewrite of Gao+2023 GC evolution.

This workflow uses raw files from ``Gao+2023/data``:

- ``fixed_trees_large_spin`` (halo trees)
- ``mass_loss.txt`` (stellar-evolution mass-loss table)
- ``snaps2redshifts.txt`` (snapshot-redshift table)

The script performs three major steps:
1. Run ``src/main_spatial.py`` per Sersic index ``N_s`` to build fresh GC formation catalogs from raw trees.
2. Evolve each catalog halo-by-halo with ``src/evoGC_fast.py`` physics.
3. Write plotting-ready outputs consumed by ``my/plot.py``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
GAO_ROOT = THIS_FILE.parents[1]
SRC_DIR = GAO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evoGC_fast import evolve_single_halo  # noqa: E402
import smhm  # noqa: E402


NS_VALUES_DEFAULT = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

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

RUN_METADATA_NAME = "run_metadata.json"


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


def _parse_halo_id_list(text: str) -> List[int]:
    """Parse one comma-separated list of z=0 halo IDs."""

    if text is None:
        return []
    values = set()
    for raw_token in str(text).split(","):
        token = raw_token.strip()
        if not token:
            continue
        try:
            halo_id = int(token)
        except ValueError as exc:
            raise ValueError(
                f"invalid halo ID '{token}' in --exclude_halo; expected comma-separated integers"
            ) from exc
        if halo_id < 0:
            raise ValueError(
                f"invalid halo ID '{token}' in --exclude_halo; expected non-negative integers"
            )
        values.add(halo_id)
    return sorted(values)


def _clear_dir_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


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


def _tree_file_for_halo(tree_dir: Path, halo_id: int) -> Path:
    hid = int(halo_id)
    for suffix in (".txt", ".dat"):
        candidate = tree_dir / f"{hid}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing tree file for halo {hid} under {tree_dir}")


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
    gao_root: Path,
    stage_dir: Path,
    data_dir: Path,
    tree_dir: Path,
    ns_value: float,
    *,
    p2: float,
    p3: float,
    mpb_only: int,
    mc: float,
    run_all: int,
    log_mh_min: float,
    log_mh_max: float,
    n_halos: int,
    exclude_halo: Sequence[int],
    imbh: int,
    final_redshift: float,
    quiet: bool,
) -> Path:
    ns_str = f"{float(ns_value):.1f}"
    log_path = stage_dir / f"main_spatial_ns{_ns_tag(ns_value)}.log"
    cmd = [
        sys.executable,
        str(gao_root / "src" / "main_spatial.py"),
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
        "--mc",
        f"{float(mc):g}",
        "--run-all",
        str(int(run_all)),
        "--log-mh-min",
        f"{float(log_mh_min):g}",
        "--log-mh-max",
        f"{float(log_mh_max):g}",
        "--n-halos",
        str(int(n_halos)),
    ]
    if exclude_halo:
        cmd.extend([
            "--exclude_halo",
            ",".join(str(int(hid)) for hid in exclude_halo),
        ])
    cmd.extend([
        "--IMBH",
        str(int(imbh)),
        "--final-z",
        f"{float(final_redshift):g}",
    ])
    with log_path.open("w") as logf:
        subprocess.run(cmd, cwd=gao_root, check=True, stdout=logf, stderr=subprocess.STDOUT)
    if not quiet:
        print(f"main_spatial finished for N_s={ns_str}. log={log_path}")
    all_path = stage_dir / f"all_{ns_str}.txt"
    if not all_path.exists():
        raise FileNotFoundError(f"Expected formation catalog not found: {all_path}")
    return all_path


def _run_plot_suite(
    *,
    gao_root: Path,
    output_dir: Path,
    ns_values: Sequence[float],
    p2: float,
    p3: float,
    final_redshift: float,
    quiet: bool,
) -> Path:
    """Run ``my/plot.py`` against the freshly written model outputs."""

    p2_tag = _fmt_param_tag(p2)
    p3_tag = _fmt_param_tag(p3)
    plot_output_dir = output_dir / "_plots"
    allcat_path = output_dir / f"allcat_s-0_p2-{p2_tag}_p3-{p3_tag}.txt"
    mpb_path = output_dir / "mpb_from_fixed_trees.csv"
    ns_values_arg = ",".join(f"{float(ns):.1f}" for ns in ns_values)
    cmd = [
        sys.executable,
        str(gao_root / "my" / "plot.py"),
        "--allcat",
        str(allcat_path),
        "--mpb",
        str(mpb_path),
        "--ns-values",
        ns_values_arg,
        "--output",
        str(plot_output_dir),
        "--final-z",
        f"{float(final_redshift):g}",
    ]
    if not quiet:
        print(f"plot.py starting. output={plot_output_dir}")
    subprocess.run(cmd, cwd=gao_root, check=True)
    if not quiet:
        print(f"plot.py finished. output={plot_output_dir}")
    return plot_output_dir


def _build_allcat_table(
    all_rows: np.ndarray,
    *,
    tree_dir: Path,
    z_snap: np.ndarray,
    final_redshift: float,
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

    logmstar_z0 = np.log10([smhm.SMHM(10.0 ** m, final_redshift, scatter=False) for m in logmh_z0])
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
    bgsw: int,
    ts_m: float,
    ts_r: float,
    analy: int,
    final_redshift: float,
    fortran_bug: bool,
    verbose: bool,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
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
        bgsw=bgsw,
        ts_m=ts_m,
        ts_r=ts_r,
        analy=analy,
        gcini_path=gcini_halo,
        depos_path=depos_halo,
        gcfin_path=gcfin_halo,
        haloevo_path=tree_halo_p,
        verbose=verbose,
        sersic_n=float(ns),
        final_redshift=final_redshift,
        fortran_bug=fortran_bug,)

    return (
        int(hz0),
        gcfin_arr[:, 1].astype(int),
        np.asarray(gcfin_arr[:, 2], dtype=float),
        np.asarray(gcfin_arr[:, 6], dtype=float),)


def _run_single_ns_pipeline(
    *,
    ns: float,
    gao_root: Path,
    data_dir: Path,
    tree_dir: Path,
    output_dir: Path,
    stage_root: Path,
    tmp_gcini_root: Path,
    z_snap: np.ndarray,
    p2: float,
    p3: float,
    mpb_only: int,
    mc: float,
    run_all: int,
    log_mh_min: float,
    log_mh_max: float,
    n_halos: int,
    exclude_halo: Sequence[int],
    imbh: int,
    bgsw: int,
    ts_m: float,
    ts_r: float,
    analy: int,
    final_redshift: float,
    fortran_bug: bool,
    jobs: int,
    quiet: bool,
) -> tuple[float, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Run formation + evolution for one Sersic index value.

    Each ``N_s`` gets its own isolated temporary output directory for
    ``main_spatial.py`` and its own temporary GCini directory. That avoids file
    collisions when multiple ``N_s`` values are processed concurrently, while
    still reading the raw data directly from ``Gao+2023/data``.
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
        gao_root,
        stage_dir,
        data_dir,
        tree_dir,
        ns,
        p2=p2,
        p3=p3,
        mpb_only=mpb_only,
        mc=mc,
        run_all=run_all,
        log_mh_min=log_mh_min,
        log_mh_max=log_mh_max,
        n_halos=n_halos,
        exclude_halo=exclude_halo,
        imbh=imbh,
        final_redshift=final_redshift,
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
            hz0_ret, status_h, m_final_h, r_final_h = _evolve_one_halo_task(
                hz0=int(hz0),
                halo_rows=np.array(all_rows[idx, :], dtype=float, copy=True),
                ns=float(ns),
                ns_tag=ns_tag,
                tmp_work_dir=str(tmp_gcini_dir),
                tree_halo=str(tree_halo),
                bgsw=bgsw,
                ts_m=ts_m,
                ts_r=ts_r,
                analy=analy,
                final_redshift=final_redshift,
                fortran_bug=fortran_bug,
                verbose=not quiet,
            )
            status[idx] = status_h
            m_final[idx] = m_final_h
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
                    bgsw=bgsw,
                    ts_m=ts_m,
                    ts_r=ts_r,
                    analy=analy,
                    final_redshift=final_redshift,
                    fortran_bug=fortran_bug,
                    verbose=False,
                )
                futures[fut] = int(hz0)

            completed = 0
            for fut in as_completed(futures):
                hz0_ret, status_h, m_final_h, r_final_h = fut.result()
                idx = halo_index_map[hz0_ret]
                status[idx] = status_h
                m_final[idx] = m_final_h
                r_final[idx] = r_final_h
                completed += 1
                if (not quiet) and (completed == 1 or completed % 10 == 0 or completed == len(unique_halos)):
                    print(f"N_s={ns_tag}: completed {completed}/{len(unique_halos)} halos")

    allcat = _build_allcat_table(
        all_rows,
        tree_dir=tree_dir,
        z_snap=z_snap,
        final_redshift=final_redshift,
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
    return float(ns), allcat[:, 0].astype(int), summary_df, halo_summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Run Gao+2023 Python GC evolution using Gao data files plus a configurable fixed-tree directory."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--gao-root", type=Path, default=GAO_ROOT, help="Path to Gao+2023 root.")
    parser.add_argument("--output", type=Path, default=Path("/lingshan/disk3/subonan/_outputs/Gao+2023"), help="Output directory.")
    parser.add_argument("--tree-dir", type=Path, default=None, help="Optional fixed-tree input directory. Defaults to <gao-root>/data/fixed_trees_large_spin.")
    parser.add_argument("--clear-output", action="store_true", help="Clear output directory before writing.")
    parser.add_argument(
        "--ns-values",
        type=str,
        default=",".join(str(v) for v in NS_VALUES_DEFAULT),
        help="Comma-separated N_s values to run.",
    )

    # Physics/evolution switches used by the Python GCevo rewrite.
    parser.add_argument("--bgsw", type=int, default=1, help="background model: 1 evolving host, 0 fixed Sersic MW-like, -1 fixed bulge+disk+DM MW-like")
    parser.add_argument("--ts-m", type=float, default=0.5, help="adaptive mass-loss timestep factor for evoGC_fast")
    parser.add_argument("--ts-r", type=float, default=0.5, help="adaptive orbital-decay timestep factor for evoGC_fast")
    parser.add_argument("--analy", type=int, choices=[0, 1], default=0, help="if 1, use analytic background-density evaluation in evoGC_fast instead of the lookup-table mode")
    parser.add_argument("--final-z", "--final-redshift", dest="final_z", type=float, default=0.0, help="final redshift where the run stops; 0 means the present day")
    parser.add_argument("--Fortran", dest="fortran_mode", action="store_true", help="make evoGC_fast emulate the archived Fortran HaloBG first-character reader bug")

    # Formation-model parameters passed directly to main_spatial.py.
    parser.add_argument("--p2", type=float, default=6.75, help="GC formation-efficiency normalization in M_GC = 3e-5 * p2 * M_gas / f_b")
    parser.add_argument("--p3", type=float, default=0.5, help="threshold in ((Delta M_h / M_h) / Delta t) above which a GC formation event is triggered")
    parser.add_argument("--mpb-only", type=int, default=0, help="if 1, only use the main progenitor branch; if 0, include all retained branches")
    parser.add_argument("--mc", type=float, default=12.0, help="log10 Schechter cutoff mass Mc in Msun for the GC initial mass function")
    parser.add_argument("--run-all", type=int, default=1, help="if 1, process all halos in the tree set; if 0, apply the mass window and halo count below")
    parser.add_argument("--log-mh-min", type=float, default=11.5, help="minimum retained host-halo log mass at the chosen final redshift")
    parser.add_argument("--log-mh-max", type=float, default=12.5, help="maximum retained host-halo log mass at the chosen final redshift")
    parser.add_argument("--n-halos", type=int, default=10, help="maximum number of halos to run when --run-all=0")
    parser.add_argument("--exclude_halo", type=str, default="", help="comma-separated z=0 halo IDs to exclude before halo selection and counting")
    parser.add_argument("--IMBH", type=int, choices=[0, 1], default=1, help="enable the IMBH seeding module in main_spatial if 1, otherwise write zero IMBH-related columns")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel halo-evolution workers per N_s run.")
    parser.add_argument("--ns-jobs", type=int, default=1, help="Concurrent N_s pipelines.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Run my/plot.py automatically after the simulation and write figures to <output>/_plots.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging.")
    args = parser.parse_args()

    gao_root = args.gao_root.resolve()
    data_dir = gao_root / "data"
    tree_dir = args.tree_dir.resolve() if args.tree_dir is not None else data_dir / "fixed_trees_large_spin"
    if not tree_dir.is_dir():
        raise FileNotFoundError(f"Missing tree directory: {tree_dir}")

    output_dir = args.output.resolve()
    if args.clear_output:
        _confirm_clear_output(output_dir)
        _clear_dir_contents(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    ns_values = _parse_ns_values(args.ns_values)
    exclude_halo = _parse_halo_id_list(args.exclude_halo)
    z_snap = _build_snap_map(data_dir / "snaps2redshifts.txt")

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
        template_halo_ids: np.ndarray | None = None
        ns_jobs = max(1, int(args.ns_jobs))
        ns_results: Dict[float, tuple[np.ndarray, pd.DataFrame, pd.DataFrame]] = {}

        if ns_jobs == 1 or len(ns_values) == 1:
            for ns in ns_values:
                ns_ret, halo_ids_ret, summary_df_ret, halo_summary_df_ret = _run_single_ns_pipeline(
                    ns=ns,
                    gao_root=gao_root,
                    data_dir=data_dir,
                    tree_dir=tree_dir,
                    output_dir=output_dir,
                    stage_root=stage_root,
                    tmp_gcini_root=tmp_gcini_root,
                    z_snap=z_snap,
                    p2=args.p2,
                    p3=args.p3,
                    mpb_only=args.mpb_only,
                    mc=args.mc,
                    run_all=args.run_all,
                    log_mh_min=args.log_mh_min,
                    log_mh_max=args.log_mh_max,
                    n_halos=args.n_halos,
                    exclude_halo=exclude_halo,
                    imbh=args.IMBH,
                    bgsw=args.bgsw,
                    ts_m=args.ts_m,
                    ts_r=args.ts_r,
                    analy=args.analy,
                    final_redshift=args.final_z,
                    fortran_bug=args.fortran_mode,
                    jobs=args.jobs,
                    quiet=args.quiet,
                )
                ns_results[ns_ret] = (halo_ids_ret, summary_df_ret, halo_summary_df_ret)
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
                        gao_root=gao_root,
                        data_dir=data_dir,
                        tree_dir=tree_dir,
                        output_dir=output_dir,
                        stage_root=stage_root,
                        tmp_gcini_root=tmp_gcini_root,
                        z_snap=z_snap,
                        p2=args.p2,
                        p3=args.p3,
                        mpb_only=args.mpb_only,
                        mc=args.mc,
                        run_all=args.run_all,
                        log_mh_min=args.log_mh_min,
                        log_mh_max=args.log_mh_max,
                        n_halos=args.n_halos,
                        exclude_halo=exclude_halo,
                        imbh=args.IMBH,
                        bgsw=args.bgsw,
                        ts_m=args.ts_m,
                        ts_r=args.ts_r,
                        analy=args.analy,
                        final_redshift=args.final_z,
                        fortran_bug=args.fortran_mode,
                        jobs=args.jobs,
                        quiet=args.quiet,
                    )
                    futures[fut] = float(ns)

                completed = 0
                for fut in as_completed(futures):
                    ns_ret, halo_ids_ret, summary_df_ret, halo_summary_df_ret = fut.result()
                    ns_results[ns_ret] = (halo_ids_ret, summary_df_ret, halo_summary_df_ret)
                    completed += 1
                    if not args.quiet:
                        print(f"N_s batch: completed {completed}/{len(ns_values)} N_s runs")

        for ns in ns_values:
            halo_ids_ret, summary_df_ret, halo_summary_df_ret = ns_results[float(ns)]
            summary_parts.append(summary_df_ret)
            halo_summary_parts.append(halo_summary_df_ret.assign(ns=float(ns)))
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

        _combine_all_ns_outputs(output_dir=output_dir, ns_values=ns_values)

        summary = pd.concat(summary_parts, ignore_index=True)
        summary.to_csv(output_dir / "python_evo_summary.csv", index=False)
        halo_summary = pd.concat(halo_summary_parts, ignore_index=True)
        halo_summary.to_csv(output_dir / "haloSummary_all.csv", index=False)
        with (output_dir / RUN_METADATA_NAME).open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "final_redshift": float(args.final_z),
                    "bgsw": int(args.bgsw),
                    "ts_m": float(args.ts_m),
                    "ts_r": float(args.ts_r),
                    "analy": int(args.analy),
                    "fortran_mode": bool(args.fortran_mode),
                    "p2": float(args.p2),
                    "p3": float(args.p3),
                    "mc": float(args.mc),
                    "IMBH": int(args.IMBH),
                    "mpb_only": int(args.mpb_only),
                    "run_all": int(args.run_all),
                    "log_mh_min": float(args.log_mh_min),
                    "log_mh_max": float(args.log_mh_max),
                    "n_halos": int(args.n_halos),
                    "exclude_halo": [int(v) for v in exclude_halo],
                    "ns_values": [float(v) for v in ns_values],
                },
                f,
                indent=2,
                sort_keys=True,
            )

        plot_output_dir: Path | None = None
        if args.plot:
            plot_output_dir = _run_plot_suite(
                gao_root=gao_root,
                output_dir=output_dir,
                ns_values=ns_values,
                p2=args.p2,
                p3=args.p3,
                final_redshift=args.final_z,
                quiet=args.quiet,
            )

        elapsed = time.time() - t0
        print(
            "DONE "
            f"ns={len(ns_values)} "
            f"halos={len(np.unique(template_halo_ids))} "
            f"rows_per_ns={len(template_halo_ids)} "
            f"elapsed_s={elapsed:.2f}"
        )
        print(f"OUTPUT {output_dir}")
        if plot_output_dir is not None:
            print(f"PLOTS {plot_output_dir}")
    finally:
        if tmp_gcini_keeper is not None:
            tmp_gcini_keeper.cleanup()
        if stage_root_keeper is not None:
            stage_root_keeper.cleanup()


if __name__ == "__main__":
    main()
