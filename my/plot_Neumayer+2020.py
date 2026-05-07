#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""
Reproduce Neumayer et al. (2020) Figure 12 from local High-z SMBHs outputs.

This script is intentionally standalone. It reads one local model output directory
and cached observational source tables under
``/home/subonan/High-z SMBHs/data/Neumayer+2020``. It does not modify the
existing run or plotting pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Tuple
import warnings

import matplotlib as mpl
from matplotlib import font_manager

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

mpl.use("Agg")

from astropy.table import Table, join, vstack
from astropy.units import UnitsWarning
from astropy.utils.metadata import MergeConflictWarning
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UnitsWarning)
warnings.filterwarnings("ignore", category=MergeConflictWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = Path("/lingshan/disk3/subonan/_outputs/High-z_SMBHs_Orig_R0.5_z0")
DEFAULT_OBS_CACHE_DIR = PROJECT_ROOT / "data" / "Neumayer+2020"

STD_DPI = 512
NS_VALUE_DEFAULT = 2.0
NSC_PROXY_RADIUS_PC = 10.0
FIGURE_03_FILENAME = "Fig.03_galaxy_demographics.png"
FIGURE_12_FILENAME = "Fig.12_nsc_scaling.png"
RAW_SUBDIR = "raw"
ORIGINAL_REVIEW_SUBDIR = "original_nsc_review"
COMPILED_FIG03_CSV = "neumayer2020_fig03_demographics.csv"
COMPILED_FIG03_META_JSON = "neumayer2020_fig03_demographics_meta.json"
COMPILED_OBS_CSV = "neumayer2020_fig12_compilation.csv"
COMPILED_OBS_META_JSON = "neumayer2020_fig12_compilation_meta.json"
RUN_METADATA_NAME = "run_metadata.json"
HALO_TREE_LOOKUP_NAME = "halo_tree_lookup.csv"
FULL_PHYSICS_COUNTERPARTS_NAME = "full_physics_counterparts_z0.csv"
NEUMAYER_DIVIDER_NAME = "neumayer2020_fig3_divider.json"
HOST_TYPE_MODES = ("auto", "split", "none")
OBS_HOST_TYPE_COLOURS = {"late": "tab:blue", "early": "tab:red"}
MODEL_HOST_TYPE_COLOURS = {"late": "dodgerblue", "early": "firebrick"}

PAPER_FULL_FIT = (0.48, 6.51)
PAPER_GOOD_FIT = (0.92, 6.13)
TIMES_COMPATIBLE_FONT_PATHS = (
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Bold.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-Italic.otf"),
    Path("/usr/share/fonts/urw-base35/NimbusRoman-BoldItalic.otf"),
)

REQUIRED_RAW_FILES = [
    "README.md",
    "sanchez-janssen19_tab4.tex",
    "sanchez-janssen19_tab5.tex",
    "ordenes-briceno18_tab1.tex",
    "eigenthaler18_tab1.fits",
    "georgiev16.fits",
    "georgiev14_tab1plus.fits",
    "georgiev14_tab2.fits",
    "spengler17_tab8.dat",
    "erwin12_tab2.tex",
    "additional_goodmass.dat",
    "lauer05_alltab.fits",
]


@dataclass
class DepositProfile:
    halo_ids: np.ndarray
    r_outer_kpc: List[np.ndarray]
    cumulative_mass_msun: List[np.ndarray]


@dataclass
class ObsCatalog:
    table: pd.DataFrame
    cache_dir: Path
    metadata: Dict[str, object]


@dataclass
class Fig03Catalog:
    table: pd.DataFrame
    cache_dir: Path
    metadata: Dict[str, object]


@dataclass
class ModelSummary:
    table: pd.DataFrame
    ns_value: float
    nsc_radius_pc: float
    fit_slope: float
    fit_intercept: float
    divider: Dict[str, object] | None = None
    mixed_suite: bool = False
    host_type_mode: str = "auto"


def _apply_plot_style() -> None:
    plt.style.use("default")
    serif_fonts = _available_times_serif_fonts()
    use_tex = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif_fonts,
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


def _safe_log10(arr: np.ndarray, floor: float = 1.0e-30) -> np.ndarray:
    return np.log10(np.clip(np.asarray(arr, dtype=float), floor, None))


def _ns_tag(ns_value: float) -> str:
    return f"{float(ns_value):.1f}".replace(".", "p")


def _regular_log_bin_edges(values: Iterable[float], step_dex: float = 0.5) -> np.ndarray:
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


def _binned_percentiles(x_log: pd.Series | np.ndarray, y: pd.Series | np.ndarray, bins: np.ndarray, min_count: int = 5) -> pd.DataFrame:
    x_arr = np.asarray(x_log, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    rows = []
    for idx, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        include_right = idx == len(bins) - 2
        mask = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr >= left)
        mask &= (x_arr <= right) if include_right else (x_arr < right)
        if int(mask.sum()) < int(min_count):
            continue
        y_sel = y_arr[mask]
        rows.append(
            {
                "x": 0.5 * (left + right),
                "count": int(mask.sum()),
                "q25": float(np.percentile(y_sel, 25.0)),
                "median": float(np.percentile(y_sel, 50.0)),
                "q75": float(np.percentile(y_sel, 75.0)),
            }
        )
    return pd.DataFrame(rows)


def _occupation_fraction_summary(x_log: pd.Series | np.ndarray, has_nsc: pd.Series | np.ndarray, bins: np.ndarray, min_count: int = 1) -> pd.DataFrame:
    x_arr = np.asarray(x_log, dtype=float)
    nsc_arr = np.asarray(has_nsc, dtype=bool)
    rows = []
    for idx, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        include_right = idx == len(bins) - 2
        mask = np.isfinite(x_arr) & (x_arr >= left)
        mask &= (x_arr <= right) if include_right else (x_arr < right)
        total = int(mask.sum())
        if total < int(min_count):
            continue
        n_nsc = int(nsc_arr[mask].sum())
        fraction = float(n_nsc / total)
        err = float(math.sqrt(fraction * (1.0 - fraction) / total)) if total > 0 else np.nan
        rows.append(
            {
                "x": 0.5 * (left + right),
                "count": total,
                "n_nsc": n_nsc,
                "fraction": fraction,
                "err": err,
            }
        )
    return pd.DataFrame(rows)


def _read_comment_columns(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                text = line[1:].strip()
                if text:
                    return text.split()
    raise ValueError(f"Cannot find header columns in {path}")


def _model_output_root_from_allcat_path(allcat_path: Path) -> Path:
    parent = allcat_path.parent
    if parent.name.startswith("ns"):
        return parent.parent
    return parent


def _resolve_model_inputs_from_out_dir(out_dir: Path) -> Path:
    model_root = out_dir.resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"Model output directory does not exist: {model_root}")
    if not model_root.is_dir():
        raise NotADirectoryError(f"Model output path is not a directory: {model_root}")
    allcat_candidates = sorted(model_root.glob("allcat_s-*.txt"))
    if len(allcat_candidates) == 0:
        raise FileNotFoundError(f"Missing root allcat file in {model_root}. Expected one file matching allcat_s-*.txt.")
    if len(allcat_candidates) > 1:
        names = ", ".join(path.name for path in allcat_candidates)
        raise RuntimeError(f"Found multiple root allcat files in {model_root}; expected exactly one: {names}")
    return allcat_candidates[0].resolve()


def _build_ns_allcat_path(allcat_template_path: Path, ns_value: float) -> Path:
    model_output_root = _model_output_root_from_allcat_path(allcat_template_path)
    name = allcat_template_path.name
    if "_s-" not in name or not name.endswith(".txt"):
        raise ValueError(f"Cannot infer per-N_s allcat path from template name: {name}")
    prefix, suffix = name.split("_s-", 1)
    if "_ns" in prefix:
        prefix = prefix.rsplit("_ns", 1)[0]
    ns_tag = _ns_tag(ns_value)
    return model_output_root / f"ns{ns_tag}" / f"{prefix}_ns{ns_tag}_s-{suffix}"


def _find_final_gcs_file(allcat_ns_path: Path) -> Path:
    stem = allcat_ns_path.stem
    if "_ns" not in stem:
        raise ValueError(f"Could not infer N_s tag from {allcat_ns_path.name}")
    ns_tag = stem.split("_ns", 1)[1].split("_", 1)[0]
    path = allcat_ns_path.parent / f"finalGCs_ns{ns_tag}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing finalGCs file for {allcat_ns_path.name}: {path}")
    return path


def load_allcat(allcat_path: Path) -> pd.DataFrame:
    columns = _read_comment_columns(allcat_path)
    raw = pd.read_csv(allcat_path, sep=r"\s+", comment="#", header=None, engine="python")
    raw = raw.iloc[:, : len(columns)].copy()
    raw.columns = columns[: raw.shape[1]]
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    gc = raw.dropna(subset=["hid_z0", "logMh_z0", "logMstar_z0"]).copy()
    gc["hid_z0"] = gc["hid_z0"].astype(int)
    return gc.reset_index(drop=True)


def _load_final_gcs_table(path: Path, expected_len: int, expected_halo_ids: np.ndarray) -> pd.DataFrame:
    columns = _read_comment_columns(path)
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")
    df = df.iloc[:, : len(columns)].copy()
    df.columns = columns[: df.shape[1]]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if len(df) != expected_len:
        raise ValueError(f"{path} has {len(df)} rows, expected {expected_len}")
    halo_ids = df["halo_id_z0"].to_numpy(dtype=int)
    if not np.array_equal(halo_ids, np.asarray(expected_halo_ids, dtype=int)):
        raise ValueError(f"Row-order mismatch between {path} and the matching allcat_ns file")
    df["status"] = df["status"].astype(int)
    return df.reset_index(drop=True)


def _find_deposit_file(allcat_ns_path: Path) -> Path | None:
    stem = allcat_ns_path.stem
    if "_ns" not in stem:
        return None
    ns_tag = stem.split("_ns", 1)[1].split("_", 1)[0]
    path = allcat_ns_path.parent / f"depos_ns{ns_tag}.dat"
    return path if path.exists() else None


def _load_deposit_profile(allcat_ns_path: Path) -> DepositProfile | None:
    path = _find_deposit_file(allcat_ns_path)
    if path is None:
        return None
    arr = np.asarray(np.loadtxt(path, ndmin=2), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Unexpected deposit-file shape in {path}: {arr.shape}")

    halo_ids: List[int] = []
    r_outer_rows: List[np.ndarray] = []
    cumulative_rows: List[np.ndarray] = []
    for hid in pd.unique(arr[:, 0].astype(int)):
        block = arr[arr[:, 0].astype(int) == int(hid)]
        if len(block) == 0:
            continue
        last_time = float(block[-1, 1])
        block = block[np.isclose(block[:, 1], last_time)]
        block = block[np.argsort(block[:, 2])]
        halo_ids.append(int(hid))
        r_outer_rows.append(np.asarray(block[:, 4], dtype=float))
        cumulative_rows.append(np.cumsum(np.asarray(block[:, 5], dtype=float)))

    if len(halo_ids) == 0:
        return None
    return DepositProfile(
        halo_ids=np.asarray(halo_ids, dtype=int),
        r_outer_kpc=r_outer_rows,
        cumulative_mass_msun=cumulative_rows,
    )


def _deposit_mass_within_radius(profile: DepositProfile, radius_kpc: float, halo_ids: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
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
        rout = np.asarray(profile.r_outer_kpc[ii], dtype=float)
        cum = np.asarray(profile.cumulative_mass_msun[ii], dtype=float)
        vals[jj] = float(np.interp(float(radius_kpc), rout, cum, left=0.0, right=cum[-1]))
    return halo_use, vals


def _required_obs_raw_paths(cache_dir: Path) -> List[Path]:
    raw_dir = cache_dir / RAW_SUBDIR
    return [raw_dir / name for name in REQUIRED_RAW_FILES]


def _fig03_data_path(cache_dir: Path, filename: str) -> Path:
    candidates = [
        cache_dir / RAW_SUBDIR / filename,
        cache_dir / ORIGINAL_REVIEW_SUBDIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing Neumayer+2020 Fig.03 source file {filename!r}. "
        f"Expected one of: {', '.join(str(path) for path in candidates)}"
    )


def _fig03_frame(source: str, logmstar: np.ndarray, g_minus_i: np.ndarray, nucflag: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "source": source,
            "logMstar_gal": np.asarray(logmstar, dtype=float),
            "g_minus_i": np.asarray(g_minus_i, dtype=float),
            "has_nsc": np.asarray(nucflag, dtype=int) == 1,
        }
    )
    return df


def _build_fig03_compilation_from_cache(cache_dir: Path) -> Tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    compiled_csv = cache_dir / COMPILED_FIG03_CSV
    compiled_meta = cache_dir / COMPILED_FIG03_META_JSON
    if compiled_csv.exists() and compiled_meta.exists():
        return compiled_csv, compiled_meta

    source_files: Dict[str, str] = {}
    def path_for(filename: str) -> Path:
        path = _fig03_data_path(cache_dir, filename)
        source_files[filename] = str(path)
        return path

    frames: List[pd.DataFrame] = []

    ei = Table.read(path_for("eigenthaler18_tab1.fits"))
    ei_gi = np.asarray(ei["__g_-i__0"], dtype=float)
    ei_sloani = np.asarray(ei["g_mag_lc"], dtype=float) - ei_gi
    ei_mli = np.power(10.0, 0.979 * ei_gi - 0.831)
    ei_logmstar = np.log10(ei_mli * np.power(10.0, -0.4 * (ei_sloani - 4.53)))
    ei_nucflag = np.zeros(len(ei), dtype=int)
    ei_nucflag[np.asarray(ei["Type"], dtype=str) == "o"] = 1
    frames.append(_fig03_frame("NGFS", ei_logmstar, ei_gi, ei_nucflag))

    sjgal = Table.read(path_for("sanchez-janssen19_tab4.tex"))
    sj_gi = np.asarray(sjgal["gmag"], dtype=float) - np.asarray(sjgal["imag"], dtype=float)
    frames.append(_fig03_frame("NGVS", np.asarray(sjgal["logmstar"], dtype=float), sj_gi, np.asarray(sjgal["nucflag"], dtype=int)))

    g09 = Table.read(path_for("georgiev09_galaxies.dat"), format="ascii")
    g09_gi = 1.481 * np.asarray(g09["vi0"], dtype=float) - 0.536
    g09_sloani = (np.asarray(g09["mv"], dtype=float) - np.asarray(g09["vi0"], dtype=float)) - (-0.370282 * np.asarray(g09["vi0"], dtype=float) - 0.161448)
    g09_mli = np.power(10.0, 0.979 * g09_gi - 0.831)
    g09_logmstar = np.log10(g09_mli * np.power(10.0, -0.4 * (g09_sloani - 4.53)))
    frames.append(_fig03_frame("G09", g09_logmstar, g09_gi, np.asarray(g09["nucflag"], dtype=int)))

    g14_gal = Table.read(path_for("georgiev14_tab1plus.fits"))
    g14_nonuc = Table.read(path_for("georgiev14_tab2.fits"))
    g14_gal["nucflag"] = 1
    g14_nonuc["nucflag"] = 0
    g14 = vstack([g14_gal, g14_nonuc], join_type="outer")
    g14_gi = 0.7451 * (np.asarray(g14["Bmag"], dtype=float) - np.asarray(g14["Imag"], dtype=float)) - 0.4388
    g14_sloani = np.asarray(g14["Imag"], dtype=float) + 0.3887 + 0.0831 * (np.asarray(g14["Bmag"], dtype=float) - np.asarray(g14["Imag"], dtype=float))
    g14_mli = np.power(10.0, 0.979 * g14_gi - 0.831)
    g14_logmstar = np.log10(g14_mli * np.power(10.0, -0.4 * (g14_sloani - np.asarray(g14["MOD"], dtype=float) - 4.53)))
    g14_finite = np.isfinite(np.asarray(g14["Imag"], dtype=float))
    frames.append(_fig03_frame("G14", g14_logmstar[g14_finite], g14_gi[g14_finite], np.asarray(g14["nucflag"], dtype=int)[g14_finite]))

    f06_tab12 = Table.read(path_for("ferrarese06_tab12.fits"))
    f06_tab34 = Table.read(path_for("ferrarese06_tab34.fits"))
    f06 = join(f06_tab12, f06_tab34, keys="ACSVCS", join_type="left")
    f06_gi = np.asarray(f06["g-z"], dtype=float) * 0.7851 - 0.006 - (np.asarray(f06["E_B-V_"], dtype=float) * 3.1 * (1.20585 - 0.49246))
    f06_mlz = np.power(10.0, 0.886 * f06_gi - 0.848)
    f06_logmstar = np.log10(f06_mlz * np.power(10.0, -0.4 * (np.asarray(f06["zmag_g"], dtype=float) - np.asarray(f06["E_B-V_"], dtype=float) * 3.1 * 0.49246 - 31.087 - 4.50)))
    f06_nucflag = np.zeros(len(f06), dtype=int)
    f06_n = np.asarray(f06["N"], dtype=str)
    f06_nucflag[(f06_n == "Ia") | (f06_n == "Ib")] = 1
    f06_keep = np.asarray(f06["VCC"], dtype=int) != 881
    frames.append(_fig03_frame("ACSVCS", f06_logmstar[f06_keep], f06_gi[f06_keep], f06_nucflag[f06_keep]))

    l05 = Table.read(path_for("lauer05_alltab.fits"))
    l05_remove = ["NGC4382", "NGC4473", "NGC4486B", "NGC4621", "NGC4649", "NGC4660", "NGC4552", "NGC4472", "NGC4458", "NGC4365", "NGC4478"]
    l05_names = np.asarray(l05["galaxy"], dtype=str)
    l05_keep = ~np.isin(l05_names, np.asarray(l05_remove, dtype=str))
    l05_keep &= np.asarray(l05["vigal"], dtype=float) < 4.0
    frames.append(_fig03_frame("L05", np.asarray(l05["logmstar"], dtype=float)[l05_keep], np.asarray(l05["g-i"], dtype=float)[l05_keep], np.asarray(l05["nucflag"], dtype=int)[l05_keep]))

    df = pd.concat(frames, ignore_index=True)
    df["logMstar_gal"] = pd.to_numeric(df["logMstar_gal"], errors="coerce")
    df["g_minus_i"] = pd.to_numeric(df["g_minus_i"], errors="coerce")
    good = (
        np.isfinite(df["logMstar_gal"].to_numpy(dtype=float))
        & np.isfinite(df["g_minus_i"].to_numpy(dtype=float))
        & (df["logMstar_gal"].to_numpy(dtype=float) > 0.0)
        & (df["logMstar_gal"].to_numpy(dtype=float) < 50.0)
        & (df["g_minus_i"].to_numpy(dtype=float) > -10.0)
        & (df["g_minus_i"].to_numpy(dtype=float) < 10.0)
    )
    df = df.loc[good].copy()
    df["name"] = [f"fig03_{idx:04d}" for idx in range(len(df))]

    divider_slope = 0.12
    divider_intercept = -0.32
    divider_value = divider_slope * df["logMstar_gal"].to_numpy(dtype=float) + divider_intercept
    df["colour_class_fig3"] = np.where(df["g_minus_i"].to_numpy(dtype=float) > divider_value, "red_sequence", "blue_cloud")
    df["host_type_fig3"] = np.where(df["colour_class_fig3"] == "red_sequence", "early", "late")
    df = df[["name", "source", "logMstar_gal", "g_minus_i", "has_nsc", "colour_class_fig3", "host_type_fig3"]].copy()
    df.to_csv(compiled_csv, index=False)

    bins = np.arange(5.5, 12.5, 0.7, dtype=float)
    metadata = {
        "n_total_rows": int(len(df)),
        "n_nucleated": int(df["has_nsc"].sum()),
        "n_non_nucleated": int((~df["has_nsc"]).sum()),
        "n_blue_cloud": int((df["colour_class_fig3"] == "blue_cloud").sum()),
        "n_red_sequence": int((df["colour_class_fig3"] == "red_sequence").sum()),
        "divider_slope": divider_slope,
        "divider_intercept": divider_intercept,
        "occupation_bins": bins.tolist(),
        "source_files": source_files,
    }
    compiled_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return compiled_csv, compiled_meta


def load_fig03_observations(cache_dir: Path = DEFAULT_OBS_CACHE_DIR) -> Fig03Catalog:
    compiled_csv, compiled_meta = _build_fig03_compilation_from_cache(cache_dir)
    table = pd.read_csv(compiled_csv)
    required = ["name", "source", "logMstar_gal", "g_minus_i", "has_nsc", "colour_class_fig3", "host_type_fig3"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{compiled_csv} is missing required columns: {missing}")
    if table["has_nsc"].dtype != bool:
        table["has_nsc"] = table["has_nsc"].map(lambda value: str(value).strip().lower() in {"1", "true", "t", "yes"})
    metadata = json.loads(compiled_meta.read_text(encoding="utf-8"))
    return Fig03Catalog(table=table, cache_dir=cache_dir, metadata=metadata)


def _build_obs_compilation_from_cache(cache_dir: Path) -> Tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    compiled_csv = cache_dir / COMPILED_OBS_CSV
    compiled_meta = cache_dir / COMPILED_OBS_META_JSON
    if compiled_csv.exists() and compiled_meta.exists():
        return compiled_csv, compiled_meta

    missing = [path for path in _required_obs_raw_paths(cache_dir) if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing cached Neumayer+2020 observation files. "
            "Populate the cache first under /home/subonan/High-z SMBHs/data/Neumayer+2020/raw:\n"
            f"{missing_text}"
        )

    raw_dir = cache_dir / RAW_SUBDIR
    sjgal = Table.read(raw_dir / "sanchez-janssen19_tab4.tex")
    sjnuc = Table.read(raw_dir / "sanchez-janssen19_tab5.tex")
    sjgal["g-i"] = sjgal["gmag"] - sjgal["imag"]
    sjgal["source"] = "NGVS"
    sj = join(sjnuc, sjgal, keys="id")
    sj["logmnuc"] = sj["logmstar_1"]
    sj["logmstargal"] = sj["logmstar_2"]
    sj["T"] = -1

    ob = Table.read(raw_dir / "ordenes-briceno18_tab1.tex")
    ob["NGFS"] = [text[4:-1] for text in ob["nucleus"]]
    ei = Table.read(raw_dir / "eigenthaler18_tab1.fits")
    ei["g-i"] = ei["__g_-i__0"]
    ei["sloani"] = ei["g_mag_lc"] - ei["g-i"]
    ei["mli"] = 10.0 ** (0.979 * ei["g-i"] - 0.831)
    ei["logmstargal"] = np.log10(ei["mli"] * 10.0 ** (-0.4 * (ei["sloani"] - 4.53)))
    ei["nucflag"] = np.zeros(len(ei), dtype=int)
    ei["nucflag"][(ei["Type"] == "o")] = 1
    ei["source"] = "NGFS"
    ngfs = join(ob, ei, keys="NGFS")
    ngfs["T"] = -1
    ngfs["logmnuc"][2] = "0.00       "
    ngfs["logmnuc"] = np.array([text[:5] for text in ngfs["logmnuc"]], dtype=float)

    g16 = Table.read(raw_dir / "georgiev16.fits")
    g16["Name"] = g16["Gal"]
    g14_gal = Table.read(raw_dir / "georgiev14_tab1plus.fits")
    g14_nonuc = Table.read(raw_dir / "georgiev14_tab2.fits")
    g14_gal["nucflag"] = 1
    g14_nonuc["nucflag"] = 0
    g14 = vstack([g14_gal, g14_nonuc], join_type="outer")
    g14["source"] = "G14"
    g14["g-i"] = 0.7451 * (g14["Bmag"] - g14["Imag"]) - 0.4388
    g14["sloani"] = g14["Imag"] + 0.3887 + 0.0831 * (g14["Bmag"] - g14["Imag"])
    g14["mli"] = 10.0 ** (0.979 * g14["g-i"] - 0.831)
    g14["logmstargal"] = np.log10(g14["mli"] * 10.0 ** (-0.4 * (g14["sloani"] - g14["MOD"] - 4.53)))
    g14use = g14[np.isfinite(g14["Imag"])]
    ge = join(g14use, g16, keys="Name")
    ge["logmnuc"] = np.log10(ge["MNSC"] * 1.0e4)
    ge["T"] = ge["t"]

    s17 = Table.read(raw_dir / "spengler17_tab8.dat", format="ascii")
    s17["source"] = "S17"
    s17["logmnuc"] = s17["logmstarnuc"]
    s17["T"] = -1

    oe12 = Table.read(raw_dir / "erwin12_tab2.tex")
    oe12["source"] = "E12"
    ag = Table.read(raw_dir / "additional_goodmass.dat", format="ascii")
    ag["source"] = "N18"
    e12 = vstack([oe12, ag], join_type="inner")

    l05 = Table.read(raw_dir / "lauer05_alltab.fits")
    for galaxy in ["NGC4382", "NGC4473", "NGC4486B", "NGC4621", "NGC4649", "NGC4660", "NGC4552", "NGC4472", "NGC4458", "NGC4365", "NGC4478"]:
        remove_index = np.where(l05["galaxy"] == galaxy)[0]
        if len(remove_index) > 0:
            l05.remove_row(int(remove_index[0]))
    l05 = l05[l05["nucflag"] == 1]

    allnuc = vstack([sj, ngfs, ge, s17, e12, l05], join_type="inner")
    df = allnuc.to_pandas()
    df["name"] = [f"obs_{i:04d}" for i in range(len(df))]
    for col in ["logmstargal", "logmnuc", "T"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["logmstargal", "logmnuc"]:
        bad = (~np.isfinite(df[col])) | (df[col] <= 0.0) | (df[col] > 50.0)
        df.loc[bad, col] = np.nan
    df["host_type"] = np.where(df["T"] > 1.0, "late", "early")
    df["is_high_quality"] = df["source"].isin(["E12", "N18"])
    df["log_fraction"] = df["logmnuc"] - df["logmstargal"]

    keep = df[["name", "source", "host_type", "T", "is_high_quality", "logmstargal", "logmnuc", "log_fraction"]].copy()
    keep = keep.rename(columns={"logmstargal": "logMstar_gal", "logmnuc": "logM_nsc"})
    keep.to_csv(compiled_csv, index=False)

    fit_df = keep.dropna(subset=["logMstar_gal", "logM_nsc"]).copy()
    full_fit = np.polyfit(fit_df["logMstar_gal"].to_numpy() - 9.0, fit_df["logM_nsc"].to_numpy(), 1)
    good_df = fit_df.loc[fit_df["is_high_quality"]].copy()
    good_fit = np.polyfit(good_df["logMstar_gal"].to_numpy() - 9.0, good_df["logM_nsc"].to_numpy(), 1)
    metadata = {
        "n_total_rows": int(len(keep)),
        "n_fit_rows": int(len(fit_df)),
        "n_high_quality_rows": int(len(good_df)),
        "fit_full_slope": float(full_fit[0]),
        "fit_full_intercept": float(full_fit[1]),
        "fit_high_quality_slope": float(good_fit[0]),
        "fit_high_quality_intercept": float(good_fit[1]),
        "paper_full_fit_slope": PAPER_FULL_FIT[0],
        "paper_full_fit_intercept": PAPER_FULL_FIT[1],
        "paper_high_quality_fit_slope": PAPER_GOOD_FIT[0],
        "paper_high_quality_fit_intercept": PAPER_GOOD_FIT[1],
    }
    compiled_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return compiled_csv, compiled_meta


def load_observations(cache_dir: Path = DEFAULT_OBS_CACHE_DIR) -> ObsCatalog:
    compiled_csv, compiled_meta = _build_obs_compilation_from_cache(cache_dir)
    table = pd.read_csv(compiled_csv)
    metadata = json.loads(compiled_meta.read_text(encoding="utf-8"))
    return ObsCatalog(table=table, cache_dir=cache_dir, metadata=metadata)


def _is_missing_counterpart_products_error(exc: FileNotFoundError) -> bool:
    text = str(exc)
    return (
        "cached full-physics counterpart products" in text
        or FULL_PHYSICS_COUNTERPARTS_NAME in text
        or NEUMAYER_DIVIDER_NAME in text
    )


def _load_mixed_suite_inputs(out_dir: Path, require_counterparts: bool) -> tuple[bool, pd.DataFrame | None, pd.DataFrame | None, Dict[str, object] | None]:
    if not require_counterparts:
        return False, None, None, None

    metadata_path = out_dir / RUN_METADATA_NAME
    halo_lookup_path = out_dir / HALO_TREE_LOOKUP_NAME
    if not metadata_path.exists():
        if halo_lookup_path.exists():
            raise FileNotFoundError(
                f"Mixed-suite output {out_dir} is missing {RUN_METADATA_NAME}. "
                "Re-run my/run.py after the provenance update."
            )
        return False, None, None, None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tree_dir_raw = metadata.get("tree_dir")
    if not tree_dir_raw:
        if halo_lookup_path.exists():
            raise FileNotFoundError(
                f"Mixed-suite output {out_dir} is missing the resolved tree_dir in {RUN_METADATA_NAME}. "
                "Re-run my/run.py after the provenance update."
            )
        return False, None, None, None

    tree_dir = Path(str(tree_dir_raw)).resolve()
    tree_lookup_in_tree_dir = tree_dir / "id_lookup_large_dark.csv"
    mixed_suite = tree_lookup_in_tree_dir.is_file() or halo_lookup_path.is_file()
    if not mixed_suite:
        return False, None, None, None

    if not halo_lookup_path.exists():
        raise FileNotFoundError(
            f"Mixed-suite output {out_dir} is missing {HALO_TREE_LOOKUP_NAME}. "
            "Re-run my/run.py after the provenance update."
        )

    data_root = tree_dir.parent
    counterparts_path = data_root / FULL_PHYSICS_COUNTERPARTS_NAME
    divider_path = data_root / NEUMAYER_DIVIDER_NAME
    missing = [str(path) for path in [counterparts_path, divider_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Mixed-suite output requires the cached full-physics counterpart products, "
            f"but these files are missing: {', '.join(missing)}. "
            "Run scripts/5_build_full_physics_counterparts.py first."
        )

    halo_lookup = pd.read_csv(halo_lookup_path)
    counterparts = pd.read_csv(counterparts_path)
    divider = json.loads(divider_path.read_text(encoding="utf-8"))
    return True, halo_lookup, counterparts, divider


def build_model_summary(out_dir: Path, ns_value: float, nsc_radius_pc: float = NSC_PROXY_RADIUS_PC, host_type_mode: str = "auto") -> ModelSummary:
    if host_type_mode not in HOST_TYPE_MODES:
        raise ValueError(f"host_type_mode must be one of {HOST_TYPE_MODES}; got {host_type_mode!r}")

    allcat_template_path = _resolve_model_inputs_from_out_dir(out_dir)
    allcat_path = _build_ns_allcat_path(allcat_template_path, ns_value)
    final_gcs_path = _find_final_gcs_file(allcat_path)
    formed = load_allcat(allcat_path)
    final_gcs = _load_final_gcs_table(final_gcs_path, len(formed), formed["hid_z0"].to_numpy(dtype=int))
    deposit_profile = _load_deposit_profile(allcat_path)

    halo = (
        formed.groupby("hid_z0", sort=True)
        .agg(logMh_z0=("logMh_z0", "first"), logMstar_z0=("logMstar_z0", "first"))
        .reset_index()
    )
    halo["M_halo_z0"] = np.power(10.0, halo["logMh_z0"].to_numpy(dtype=float))
    halo["M_star_z0"] = np.power(10.0, halo["logMstar_z0"].to_numpy(dtype=float))
    halo["M_nsc_proxy"] = 0.0

    radius_kpc = float(nsc_radius_pc) / 1000.0
    if deposit_profile is not None:
        dep_halo_ids, dep_mass = _deposit_mass_within_radius(deposit_profile, radius_kpc, halo["hid_z0"].to_numpy(dtype=int))
        dep_map = {int(hid): float(mass) for hid, mass in zip(dep_halo_ids, dep_mass)}
        halo["M_nsc_proxy"] = halo["hid_z0"].map(dep_map).fillna(0.0)
    else:
        fallback = formed.join(final_gcs[[col for col in ["status", "m_final_msun", "r_final_kpc"] if col in final_gcs.columns]])
        fallback = fallback.loc[(fallback["status"] == 1) & (fallback["r_final_kpc"] <= radius_kpc)].copy()
        fallback_mass = fallback.groupby("hid_z0")["m_final_msun"].sum()
        halo["M_nsc_proxy"] = halo["hid_z0"].map(fallback_mass).fillna(0.0)

    resolved_host_type_mode = host_type_mode
    if host_type_mode == "none":
        mixed_suite, halo_lookup, counterparts, divider = _load_mixed_suite_inputs(out_dir, require_counterparts=False)
    elif host_type_mode == "split":
        mixed_suite, halo_lookup, counterparts, divider = _load_mixed_suite_inputs(out_dir, require_counterparts=True)
        resolved_host_type_mode = "split" if mixed_suite else "none"
    else:
        try:
            mixed_suite, halo_lookup, counterparts, divider = _load_mixed_suite_inputs(out_dir, require_counterparts=True)
            resolved_host_type_mode = "split" if mixed_suite else "none"
        except FileNotFoundError as exc:
            if not _is_missing_counterpart_products_error(exc):
                raise
            print(f"Warning: {exc} Falling back to --host-type-mode none.")
            mixed_suite, halo_lookup, counterparts, divider = _load_mixed_suite_inputs(out_dir, require_counterparts=False)
            resolved_host_type_mode = "none"

    if mixed_suite:
        assert halo_lookup is not None
        assert counterparts is not None
        halo_lookup = halo_lookup.copy()
        counterparts = counterparts.copy()
        for col in ["hid_z0", "subhalo_id_z0", "file_index"]:
            halo_lookup[col] = pd.to_numeric(halo_lookup[col], errors="raise")
        for col in [
            "halo_id_z0",
            "subhalo_id_z0_dark",
            "fp_subhalo_id_z0",
            "matched",
            "ambiguous_match",
            "n_fp_matches",
            "stellar_mass_fp_msun",
            "logMstar_fp_msun",
            "g_mag_fp",
            "i_mag_fp",
            "g_minus_i",
        ]:
            if col in counterparts.columns:
                counterparts[col] = pd.to_numeric(counterparts[col], errors="coerce")

        halo = halo.merge(
            halo_lookup[["hid_z0", "simulation_key", "simulation", "subhalo_id_z0", "fixed_tree_basename", "file_index"]],
            on="hid_z0",
            how="left",
            validate="one_to_one",
        )
        if halo["simulation_key"].isna().any():
            missing = halo.loc[halo["simulation_key"].isna(), "hid_z0"].astype(int).tolist()
            raise ValueError(
                "halo_tree_lookup.csv is missing one or more halos present in the model output: "
                + ", ".join(str(item) for item in missing[:12])
            )

        halo = halo.merge(
            counterparts[
                [
                    "simulation_key_dark",
                    "subhalo_id_z0_dark",
                    "fp_subhalo_id_z0",
                    "logMstar_fp_msun",
                    "g_minus_i",
                    "host_type_fig3",
                    "colour_class_fig3",
                    "matched",
                    "ambiguous_match",
                    "n_fp_matches",
                    "stellar_mass_fp_msun",
                    "g_mag_fp",
                    "i_mag_fp",
                ]
            ],
            left_on=["simulation_key", "subhalo_id_z0"],
            right_on=["simulation_key_dark", "subhalo_id_z0_dark"],
            how="left",
            validate="one_to_one",
        )
        if halo["matched"].isna().any():
            missing = halo.loc[halo["matched"].isna(), ["hid_z0", "simulation_key", "subhalo_id_z0"]]
            preview = ", ".join(
                f"({int(row.hid_z0)}, {row.simulation_key}, {int(row.subhalo_id_z0)})"
                for row in missing.head(12).itertuples()
            )
            raise ValueError(
                "The cached full-physics counterpart table is missing one or more selected halos: "
                + preview
            )
    else:
        halo["simulation_key"] = pd.Series(pd.NA, index=halo.index, dtype="object")
        halo["simulation"] = pd.Series(pd.NA, index=halo.index, dtype="object")
        halo["subhalo_id_z0"] = np.nan
        halo["fixed_tree_basename"] = pd.Series(pd.NA, index=halo.index, dtype="object")
        halo["file_index"] = np.nan
        halo["simulation_key_dark"] = pd.Series(pd.NA, index=halo.index, dtype="object")
        halo["subhalo_id_z0_dark"] = np.nan
        halo["fp_subhalo_id_z0"] = np.nan
        halo["logMstar_fp_msun"] = np.nan
        halo["g_minus_i"] = np.nan
        halo["host_type_fig3"] = "unmatched"
        halo["colour_class_fig3"] = "unmatched"
        halo["matched"] = 0
        halo["ambiguous_match"] = 0
        halo["n_fp_matches"] = 0
        halo["stellar_mass_fp_msun"] = np.nan
        halo["g_mag_fp"] = np.nan
        halo["i_mag_fp"] = np.nan

    halo["logMstar_plot"] = halo["logMstar_z0"].to_numpy(dtype=float)
    use_fp = np.isfinite(pd.to_numeric(halo["logMstar_fp_msun"], errors="coerce").to_numpy(dtype=float))
    halo.loc[use_fp, "logMstar_plot"] = halo.loc[use_fp, "logMstar_fp_msun"].to_numpy(dtype=float)
    halo["M_star_plot"] = np.power(10.0, halo["logMstar_plot"].to_numpy(dtype=float))
    halo["f_nsc_proxy"] = halo["M_nsc_proxy"] / halo["M_star_z0"]
    halo["f_nsc_proxy_plot"] = halo["M_nsc_proxy"] / halo["M_star_plot"]
    halo["logM_nsc_proxy"] = _safe_log10(halo["M_nsc_proxy"].to_numpy(dtype=float))
    fit_mask = np.isfinite(halo["logMstar_plot"]) & np.isfinite(halo["logM_nsc_proxy"]) & (halo["M_nsc_proxy"] > 0.0)
    if int(fit_mask.sum()) < 2:
        raise ValueError("Need at least two model halos with finite host stellar mass and non-zero NSC proxy mass to fit Fig.12.")
    fit = np.polyfit(halo.loc[fit_mask, "logMstar_plot"].to_numpy(dtype=float) - 9.0, halo.loc[fit_mask, "logM_nsc_proxy"].to_numpy(dtype=float), 1)
    return ModelSummary(
        table=halo,
        ns_value=float(ns_value),
        nsc_radius_pc=float(nsc_radius_pc),
        fit_slope=float(fit[0]),
        fit_intercept=float(fit[1]),
        divider=divider,
        mixed_suite=bool(mixed_suite),
        host_type_mode=resolved_host_type_mode,
    )


def build_figure_03(model: ModelSummary, fig03_obs: Fig03Catalog) -> Tuple[plt.Figure, Dict[str, object]]:
    _apply_plot_style()

    obs_table = fig03_obs.table.dropna(subset=["logMstar_gal", "g_minus_i"]).copy()
    obs_table["has_nsc"] = obs_table["has_nsc"].astype(bool)
    bins = np.asarray(fig03_obs.metadata.get("occupation_bins", np.arange(5.5, 12.5, 0.7)), dtype=float)
    slope = float(fig03_obs.metadata.get("divider_slope", 0.12))
    intercept = float(fig03_obs.metadata.get("divider_intercept", -0.32))
    split_host_types = model.mixed_suite and model.host_type_mode == "split"

    fig, axes = plt.subplots(1, 2, constrained_layout=True, dpi=STD_DPI, figsize=(10.8, 4.8))
    ax_left, ax_right = axes

    x_line = np.linspace(5.5, 12.0, 400, dtype=float)
    y_line = slope * x_line + intercept
    ax_left.fill_between(x_line, y_line, np.zeros_like(x_line) + 1.55, color="tab:red", alpha=0.14, linewidth=0.0)
    ax_left.fill_between(x_line, np.zeros_like(x_line) - 1.4, y_line, color="tab:blue", alpha=0.14, linewidth=0.0)
    non_nucleated = obs_table.loc[~obs_table["has_nsc"]].copy()
    nucleated = obs_table.loc[obs_table["has_nsc"]].copy()
    ax_left.scatter(non_nucleated["logMstar_gal"].to_numpy(dtype=float), non_nucleated["g_minus_i"].to_numpy(dtype=float), facecolors="white", edgecolors="black", s=22, alpha=0.75, linewidths=0.5, label="No NSC detected")
    ax_left.scatter(nucleated["logMstar_gal"].to_numpy(dtype=float), nucleated["g_minus_i"].to_numpy(dtype=float), facecolors="black", edgecolors="black", s=20, alpha=0.55, linewidths=0.0, label="With NSC")
    ax_left.plot(x_line, y_line, c="black", lw=1.8, label="Blue/red divider")
    ax_left.text(6.0, 1.28, "red sequence", color="tab:red", fontsize=9)
    ax_left.text(6.0, -1.08, "blue cloud", color="tab:blue", fontsize=9)

    for host_type, colour, label in [("early", OBS_HOST_TYPE_COLOURS["early"], "Obs early-type"), ("late", OBS_HOST_TYPE_COLOURS["late"], "Obs late-type")]:
        subset = obs_table.loc[obs_table["host_type_fig3"] == host_type].copy()
        summary = _occupation_fraction_summary(subset["logMstar_gal"], subset["has_nsc"], bins, min_count=1)
        if summary.empty:
            continue
        x = summary["x"].to_numpy(dtype=float)
        fraction = summary["fraction"].to_numpy(dtype=float)
        err = summary["err"].to_numpy(dtype=float)
        ax_right.fill_between(x, np.clip(fraction - err, 0.0, 1.0), np.clip(fraction + err, 0.0, 1.0), color=colour, alpha=0.18, linewidth=0.0)
        ax_right.plot(x, fraction, c=colour, lw=2.2, marker="o", markersize=4, label=label)

    model_table = model.table.copy()
    model_mask = np.isfinite(model_table["logMstar_plot"].to_numpy(dtype=float))
    model_rows = model_table.loc[model_mask].copy()
    model_rows["has_nsc_model"] = model_rows["M_nsc_proxy"].to_numpy(dtype=float) > 0.0

    if split_host_types:
        for host_type, colour, label in [("early", MODEL_HOST_TYPE_COLOURS["early"], "Model early-type"), ("late", MODEL_HOST_TYPE_COLOURS["late"], "Model late-type")]:
            subset = model_rows.loc[model_rows["host_type_fig3"] == host_type].copy()
            summary = _occupation_fraction_summary(subset["logMstar_plot"], subset["has_nsc_model"], bins, min_count=1)
            if summary.empty:
                continue
            ax_right.plot(summary["x"].to_numpy(dtype=float), summary["fraction"].to_numpy(dtype=float), c=colour, lw=2.0, ls="--", marker="s", markersize=3, label=label)
        unmatched = model_rows.loc[~model_rows["host_type_fig3"].isin(["early", "late"])].copy()
        unmatched_summary = _occupation_fraction_summary(unmatched["logMstar_plot"], unmatched["has_nsc_model"], bins, min_count=1)
        if not unmatched_summary.empty:
            ax_right.plot(unmatched_summary["x"].to_numpy(dtype=float), unmatched_summary["fraction"].to_numpy(dtype=float), c="0.45", lw=1.6, ls=":", label="Model unmatched")
    else:
        model_summary = _occupation_fraction_summary(model_rows["logMstar_plot"], model_rows["has_nsc_model"], bins, min_count=1)
        if not model_summary.empty:
            ax_right.plot(model_summary["x"].to_numpy(dtype=float), model_summary["fraction"].to_numpy(dtype=float), c="black", lw=2.0, ls="--", marker="s", markersize=3, label="Model")

    ax_left.set_xlim(5.5, 12.0)
    ax_left.set_ylim(-1.4, 1.55)
    ax_left.set_xlabel(r"$\log_{10}(M_{\star}/M_{\odot})$")
    ax_left.set_ylabel(r"$(g-i)_0$")
    ax_left.grid(True, alpha=0.3, linestyle=":", which="both")
    ax_left.legend(frameon=False, loc="best", ncol=1, fontsize=8)

    ax_right.set_xlim(5.5, 12.0)
    ax_right.set_ylim(0.0, 1.0)
    ax_right.set_xlabel(r"$\log_{10}(M_{\star}/M_{\odot})$")
    ax_right.set_ylabel("Fraction of galaxies with NSC")
    ax_right.grid(True, alpha=0.3, linestyle=":", which="both")
    ax_right.legend(frameon=False, loc="best", ncol=1, fontsize=8)

    summary = {
        "n_obs_total": int(len(obs_table)),
        "n_obs_nucleated": int(obs_table["has_nsc"].sum()),
        "n_obs_non_nucleated": int((~obs_table["has_nsc"]).sum()),
        "n_obs_late": int((obs_table["host_type_fig3"] == "late").sum()),
        "n_obs_early": int((obs_table["host_type_fig3"] == "early").sum()),
        "n_model_total": int(len(model_rows)),
        "n_model_nucleated": int(model_rows["has_nsc_model"].sum()),
        "host_type_mode": model.host_type_mode,
        "split_host_types": int(split_host_types),
        "n_model_late": int((model_rows["host_type_fig3"] == "late").sum()),
        "n_model_early": int((model_rows["host_type_fig3"] == "early").sum()),
        "n_model_unmatched": int((~model_rows["host_type_fig3"].isin(["late", "early"])).sum()),
        "divider_slope": slope,
        "divider_intercept": intercept,
    }
    return fig, summary


def build_figure_12(model: ModelSummary, obs: ObsCatalog) -> Tuple[plt.Figure, Dict[str, object]]:
    _apply_plot_style()

    obs_table = obs.table.dropna(subset=["logMstar_gal", "logM_nsc"]).copy()
    obs_bins = _regular_log_bin_edges(obs_table["logMstar_gal"], 0.5)

    model_table = model.table.copy()
    model_fit_mask = np.isfinite(model_table["logMstar_plot"]) & np.isfinite(model_table["logM_nsc_proxy"]) & (model_table["M_nsc_proxy"] > 0.0)
    model_points = model_table.loc[model_fit_mask].copy()
    model_bins = _regular_log_bin_edges(model_points["logMstar_plot"], 0.5)
    model_ratio_summary = _binned_percentiles(model_points["logMstar_plot"], model_points["f_nsc_proxy_plot"], model_bins, min_count=5)
    split_host_types = model.mixed_suite and model.host_type_mode == "split"

    fig, axes = plt.subplots(1, 2, constrained_layout=True, dpi=STD_DPI, figsize=(10.8, 4.8))
    ax_left, ax_right = axes

    late_obs = obs_table.loc[obs_table["host_type"] == "late"].copy()
    early_obs = obs_table.loc[obs_table["host_type"] == "early"].copy()
    late_good = late_obs.loc[late_obs["is_high_quality"]].copy()
    early_good = early_obs.loc[early_obs["is_high_quality"]].copy()
    ax_left.scatter(np.power(10.0, late_obs["logMstar_gal"].to_numpy(dtype=float)), np.power(10.0, late_obs["logM_nsc"].to_numpy(dtype=float)), c=OBS_HOST_TYPE_COLOURS["late"], s=18, alpha=0.35, linewidths=0.0)
    ax_left.scatter(np.power(10.0, early_obs["logMstar_gal"].to_numpy(dtype=float)), np.power(10.0, early_obs["logM_nsc"].to_numpy(dtype=float)), c=OBS_HOST_TYPE_COLOURS["early"], s=18, alpha=0.35, linewidths=0.0)
    ax_left.scatter(np.power(10.0, late_good["logMstar_gal"].to_numpy(dtype=float)), np.power(10.0, late_good["logM_nsc"].to_numpy(dtype=float)), c=OBS_HOST_TYPE_COLOURS["late"], s=130, alpha=0.95, marker="*", edgecolors="black", linewidths=0.35)
    ax_left.scatter(np.power(10.0, early_good["logMstar_gal"].to_numpy(dtype=float)), np.power(10.0, early_good["logM_nsc"].to_numpy(dtype=float)), c=OBS_HOST_TYPE_COLOURS["early"], s=130, alpha=0.95, marker="*", edgecolors="black", linewidths=0.35)

    if split_host_types:
        late_model = model_points.loc[model_points["host_type_fig3"] == "late"].copy()
        early_model = model_points.loc[model_points["host_type_fig3"] == "early"].copy()
        unmatched_model = model_points.loc[~model_points["host_type_fig3"].isin(["late", "early"])].copy()
        if len(late_model) > 0:
            ax_left.scatter(late_model["M_star_plot"].to_numpy(dtype=float), late_model["M_nsc_proxy"].to_numpy(dtype=float), c=MODEL_HOST_TYPE_COLOURS["late"], s=20, alpha=0.45, linewidths=0.0)
        if len(early_model) > 0:
            ax_left.scatter(early_model["M_star_plot"].to_numpy(dtype=float), early_model["M_nsc_proxy"].to_numpy(dtype=float), c=MODEL_HOST_TYPE_COLOURS["early"], s=20, alpha=0.45, linewidths=0.0)
        if len(unmatched_model) > 0:
            ax_left.scatter(unmatched_model["M_star_plot"].to_numpy(dtype=float), unmatched_model["M_nsc_proxy"].to_numpy(dtype=float), c="0.65", s=18, alpha=0.25, linewidths=0.0)
    else:
        ax_left.scatter(model_points["M_star_plot"].to_numpy(dtype=float), model_points["M_nsc_proxy"].to_numpy(dtype=float), c="0.25", s=18, alpha=0.45, linewidths=0.0)

    x_line = np.logspace(5.5, 11.2, 300)
    y_paper = np.power(10.0, PAPER_FULL_FIT[0] * (np.log10(x_line) - 9.0) + PAPER_FULL_FIT[1])
    y_good = np.power(10.0, PAPER_GOOD_FIT[0] * (np.log10(x_line) - 9.0) + PAPER_GOOD_FIT[1])
    y_model = np.power(10.0, model.fit_slope * (np.log10(x_line) - 9.0) + model.fit_intercept)
    ax_left.plot(x_line, y_paper, c="black", ls="--", lw=1.8)
    ax_left.plot(x_line, y_good, c="black", ls="-.", lw=1.8)
    ax_left.plot(x_line, y_model, c="black", ls="-", lw=2.0)

    for host_type, colour in [("late", OBS_HOST_TYPE_COLOURS["late"]), ("early", OBS_HOST_TYPE_COLOURS["early"])]:
        subset = obs_table.loc[obs_table["host_type"] == host_type].copy()
        summary = _binned_percentiles(subset["logMstar_gal"], np.power(10.0, subset["log_fraction"]), obs_bins, min_count=5)
        if summary.empty:
            continue
        ax_right.fill_between(np.power(10.0, summary["x"]), summary["q25"], summary["q75"], facecolor=colour, edgecolor="none", linewidth=0.0, alpha=0.2)
        ax_right.plot(np.power(10.0, summary["x"]), summary["median"], c=colour, lw=2.0)

    if split_host_types:
        for host_type, colour in [("late", MODEL_HOST_TYPE_COLOURS["late"]), ("early", MODEL_HOST_TYPE_COLOURS["early"])]:
            subset = model_points.loc[model_points["host_type_fig3"] == host_type].copy()
            summary = _binned_percentiles(subset["logMstar_plot"], subset["f_nsc_proxy_plot"], model_bins, min_count=5)
            if summary.empty:
                continue
            ax_right.fill_between(np.power(10.0, summary["x"]), summary["q25"], summary["q75"], facecolor=colour, edgecolor="none", linewidth=0.0, alpha=0.15)
            ax_right.plot(np.power(10.0, summary["x"]), summary["median"], c=colour, lw=2.0)
    else:
        if not model_ratio_summary.empty:
            ax_right.fill_between(np.power(10.0, model_ratio_summary["x"]), model_ratio_summary["q25"], model_ratio_summary["q75"], facecolor="0.55", edgecolor="none", linewidth=0.0, alpha=0.25)
            ax_right.plot(np.power(10.0, model_ratio_summary["x"]), model_ratio_summary["median"], c="black", lw=2.0)
    ax_right.plot(x_line, y_paper / x_line, c="black", ls="--", lw=1.8)

    left_handles = [
        mpl.lines.Line2D([], [], marker="o", ls="", color=OBS_HOST_TYPE_COLOURS["late"], markersize=6, alpha=0.7, label="Obs late-type"),
        mpl.lines.Line2D([], [], marker="o", ls="", color=OBS_HOST_TYPE_COLOURS["early"], markersize=6, alpha=0.7, label="Obs early-type"),
        mpl.lines.Line2D([], [], marker="*", ls="", color="black", markersize=10, label="Obs dyn/spec subset"),
    ]
    if split_host_types:
        left_handles.extend(
            [
                mpl.lines.Line2D([], [], marker="o", ls="", color=MODEL_HOST_TYPE_COLOURS["late"], markersize=6, alpha=0.7, label="Model late (Fig.3 colour cut)"),
                mpl.lines.Line2D([], [], marker="o", ls="", color=MODEL_HOST_TYPE_COLOURS["early"], markersize=6, alpha=0.7, label="Model early (Fig.3 colour cut)"),
                mpl.lines.Line2D([], [], marker="o", ls="", color="0.65", markersize=6, alpha=0.45, label="Model unmatched"),
            ]
        )
    else:
        left_handles.append(mpl.lines.Line2D([], [], marker="o", ls="", color="0.25", markersize=6, alpha=0.7, label="Model"))
    left_handles.extend(
        [
            mpl.lines.Line2D([], [], c="black", ls="--", lw=1.8, label="Paper fit"),
            mpl.lines.Line2D([], [], c="black", ls="-.", lw=1.8, label="Paper dyn/spec fit"),
            mpl.lines.Line2D([], [], c="black", ls="-", lw=2.0, label=f"Model fit ({int(model.nsc_radius_pc)} pc proxy)"),
        ]
    )
    ax_left.legend(handles=left_handles, frameon=False, loc="upper left", ncol=1, fontsize=8)

    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    ax_left.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    ax_left.set_xlabel(r"$M_{\star,\mathrm{gal}}\,[M_{\odot}]$")
    ax_left.set_ylabel(r"$M_{\mathrm{NSC}}\,[M_{\odot}]$")
    ax_left.set_xlim(10.0 ** 5.5, 10.0 ** 11.2)
    ax_left.set_ylim(10.0 ** 4.5, 10.0 ** 9.1)
    ax_left.grid(True, alpha=0.3, linestyle=":", which="both")

    ax_right.set_xscale("log")
    ax_right.set_yscale("log")
    ax_right.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    ax_right.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    ax_right.set_xlabel(r"$M_{\star,\mathrm{gal}}\,[M_{\odot}]$")
    ax_right.set_ylabel(r"$M_{\mathrm{NSC}}/M_{\star,\mathrm{gal}}$")
    ax_right.set_xlim(10.0 ** 5.5, 10.0 ** 11.2)
    ax_right.set_ylim(1.0e-4, 1.0)
    ax_right.grid(True, alpha=0.3, linestyle=":", which="both")
    right_handles = [
        mpl.lines.Line2D([], [], c=OBS_HOST_TYPE_COLOURS["late"], lw=2.0, label="Obs late-type median"),
        mpl.lines.Line2D([], [], c=OBS_HOST_TYPE_COLOURS["early"], lw=2.0, label="Obs early-type median"),
    ]
    if split_host_types:
        right_handles.extend(
            [
                mpl.patches.Patch(facecolor=MODEL_HOST_TYPE_COLOURS["late"], edgecolor="none", alpha=0.15, label="Model late IQR"),
                mpl.lines.Line2D([], [], c=MODEL_HOST_TYPE_COLOURS["late"], lw=2.0, label="Model late median"),
                mpl.patches.Patch(facecolor=MODEL_HOST_TYPE_COLOURS["early"], edgecolor="none", alpha=0.15, label="Model early IQR"),
                mpl.lines.Line2D([], [], c=MODEL_HOST_TYPE_COLOURS["early"], lw=2.0, label="Model early median"),
            ]
        )
    else:
        right_handles.extend(
            [
                mpl.patches.Patch(facecolor="0.55", edgecolor="none", alpha=0.25, label="Model IQR"),
                mpl.lines.Line2D([], [], c="black", lw=2.0, label="Model median"),
            ]
        )
    right_handles.append(mpl.lines.Line2D([], [], c="black", ls="--", lw=1.8, label=r"Paper fit / $M_{\star}$"))
    ax_right.legend(handles=right_handles, frameon=False, loc="lower left", ncol=1, fontsize=8)

    summary = {
        "obs_full_fit_slope_from_cache": float(obs.metadata["fit_full_slope"]),
        "obs_full_fit_intercept_from_cache": float(obs.metadata["fit_full_intercept"]),
        "obs_high_quality_fit_slope_from_cache": float(obs.metadata["fit_high_quality_slope"]),
        "obs_high_quality_fit_intercept_from_cache": float(obs.metadata["fit_high_quality_intercept"]),
        "model_fit_slope": float(model.fit_slope),
        "model_fit_intercept": float(model.fit_intercept),
        "n_obs_fit_rows": int(obs.metadata["n_fit_rows"]),
        "n_model_halos": int(len(model_points)),
        "nsc_proxy_radius_pc": float(model.nsc_radius_pc),
        "mixed_suite": int(model.mixed_suite),
        "host_type_mode": model.host_type_mode,
        "split_host_types": int(split_host_types),
        "n_model_late": int((model_points["host_type_fig3"] == "late").sum()),
        "n_model_early": int((model_points["host_type_fig3"] == "early").sum()),
        "n_model_unmatched": int((~model_points["host_type_fig3"].isin(["late", "early"])).sum()),
    }
    return fig, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Neumayer+2020 Figures 3 and 12 from one local High-z SMBHs output directory.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Model output directory containing allcat/ns*/deposit products.")
    parser.add_argument("--plot_dir", type=Path, default=None, help="Plot output directory. Default: <out_dir>/_plots_Neumayer+2020")
    parser.add_argument("--ns-value", type=float, default=NS_VALUE_DEFAULT, help="Sersic N_s value to load from the ns* subdirectory.")
    parser.add_argument("--host-type-mode", choices=HOST_TYPE_MODES, default="auto", help="Use full-physics early/late split ('split'), automatically fall back when unavailable ('auto'), or plot all hosts together ('none').")
    parser.add_argument("--no-host-type-split", action="store_true", help="Shortcut for --host-type-mode none.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    plot_dir = args.plot_dir.resolve() if args.plot_dir is not None else out_dir / "_plots_Neumayer+2020"
    plot_dir.mkdir(parents=True, exist_ok=True)

    obs = load_observations()
    fig03_obs = load_fig03_observations()
    host_type_mode = "none" if args.no_host_type_split else args.host_type_mode
    model = build_model_summary(out_dir, args.ns_value, NSC_PROXY_RADIUS_PC, host_type_mode=host_type_mode)

    fig03, summary03 = build_figure_03(model, fig03_obs)
    figure03_path = plot_dir / FIGURE_03_FILENAME
    fig03.savefig(figure03_path, bbox_inches="tight")
    plt.close(fig03)
    print(
        f"Saved {figure03_path} | "
        f"n_obs={summary03['n_obs_total']} | "
        f"n_obs_nsc={summary03['n_obs_nucleated']} | "
        f"host_type_mode={summary03['host_type_mode']}"
    )

    fig12, summary12 = build_figure_12(model, obs)
    figure12_path = plot_dir / FIGURE_12_FILENAME
    fig12.savefig(figure12_path, bbox_inches="tight")
    plt.close(fig12)
    print(
        f"Saved {figure12_path} | "
        f"model_fit=({summary12['model_fit_slope']:.3f}, {summary12['model_fit_intercept']:.3f}) | "
        f"obs_fit=({summary12['obs_full_fit_slope_from_cache']:.3f}, {summary12['obs_full_fit_intercept_from_cache']:.3f}) | "
        f"host_type_mode={summary12['host_type_mode']}"
    )


if __name__ == "__main__":
    main()
