#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""
Reproduce the Choksi, Gnedin & Li (2018) GC-system figures from local outputs.

This script is intentionally standalone. It reads the local High-z SMBHs model
products from one model output directory in ``/lingshan/disk3/subonan/_outputs``
and uses persistent, machine-readable observational tables cached under
``/home/subonan/High-z SMBHs/data/Choksi+2018``. It does not modify the
existing run or plotting pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import matplotlib as mpl
import tarfile

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Sequence, Tuple
import zipfile

from scipy.ndimage import gaussian_filter
from scipy.stats import ks_2samp
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import smhm  # noqa: E402
from evo import Redshift2CosmicTimeGyr  # noqa: E402

np.random.seed(1)

STD_DPI = 512
NS_VALUE_DEFAULT = 2.0
RUN_METADATA_NAME = "run_metadata.json"

DEFAULT_OUT_DIR = Path("/lingshan/disk3/subonan/_outputs/High-z_SMBHs_Max64_z0")
DEFAULT_OBS_CACHE_DIR = PROJECT_ROOT / "data" / "Choksi+2018"
CHOKSI_SUPPLEMENT_DIR = DEFAULT_OBS_CACHE_DIR / "choksi_supplement"
CHOKSI_MODEL_PATH = CHOKSI_SUPPLEMENT_DIR / "model.txt"
T_UNIVERSE_GYR = float(Redshift2CosmicTimeGyr(0.0))

FIGURE_STEMS = {
    1: "mean_feh",
    2: "sigma_feh",
    3: "mgc_vs_mhalo",
    4: "mdf_examples",
    5: "red_blue_peaks",
    6: "formation_histories",
    7: "blue_tilt",
    8: "host_masses_at_formation",
    9: "age_metallicity",
    10: "gc_fraction_vs_feh",
}

FEH_MIN = -2.3
FEH_MAX = 0.3
GLOBAL_SPLIT_DEFAULT = -0.88
MIN_GMM_COUNT = 20
MIN_SYSTEM_GC_COUNT = 5

H100 = 0.704
FB = 0.167
MMR_SLOPE = 0.35
MMR_TURNOVER = 10.5
MAX_FEH = 0.3
TDEP = 0.3
MMR_EVOLUTION = 0.9

HARRIS_2015_RATIO = 3.4e-5
VIRGO_DISTANCE_MODULUS = 31.09
ACS_SOLAR_MAG_Z = 4.51
GC_ML_Z = 1.45

CHOKSI_SUPP_WAYBACK_URL = (
    "http://web.archive.org/web/20220426175612if_/"
    "http://ugastro.berkeley.edu/~nchoksi/cgl18_supplemental.zip"
)
ACSVCS_HOSTS_URL = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=J/ApJS/164/334/acsvcs"
ACSVCS_GC_URL = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=J/ApJS/180/54/table4"
VANDENBERG_2013_ARXIV_URL = "https://arxiv.org/e-print/1308.2257"
LEAMAN_2013_ARXIV_URL = "https://arxiv.org/e-print/1309.0822"
WAGNER_KAISER_2017_ARXIV_URL = "https://arxiv.org/e-print/1707.01571"
LAMERS_2017_ARXIV_URL = "https://arxiv.org/e-print/1706.00939"

ALLCAT_REQUIRED_COLUMNS = [
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


@dataclass
class ModelCatalog:
    formed: pd.DataFrame
    catalog: pd.DataFrame
    survivors: pd.DataFrame
    halo_summary: pd.DataFrame
    mpb: pd.DataFrame
    allcat_path: Path
    final_gcs_path: Path
    run_metadata: Dict[str, object]
    split_threshold: float


@dataclass
class PaperModelCatalog:
    survivors: pd.DataFrame
    split_threshold: float


@dataclass
class ObsCatalog:
    systems: pd.DataFrame
    vcs_systems: pd.DataFrame
    acsvcs_hosts: pd.DataFrame
    acsvcs_gc: pd.DataFrame
    mw_age_metallicity: pd.DataFrame
    lmc_age_metallicity: pd.DataFrame
    lamers_summary: pd.DataFrame
    obs_cache_dir: Path


def _apply_plot_style() -> None:
    """Use the local plotting style, enabling TeX only when available."""

    use_tex = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
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


def _ns_tag(ns_value: float) -> str:
    return f"{float(ns_value):.1f}".replace(".", "p")


def _model_output_root_from_allcat_path(allcat_path: Path) -> Path:
    parent = allcat_path.parent
    if re.fullmatch(r"ns[0-9]+p[0-9]+", parent.name):
        return parent.parent
    return parent


def _load_run_metadata(allcat_path: Path) -> Dict[str, object]:
    path = _model_output_root_from_allcat_path(allcat_path) / RUN_METADATA_NAME
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_comment_columns(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                continue
            text = line[1:].strip()
            if text:
                return text.split()
    raise ValueError(f"Cannot find header columns in {path}")


def _build_ns_allcat_path(allcat_template_path: Path, ns_value: float) -> Path:
    model_output_root = _model_output_root_from_allcat_path(allcat_template_path)
    name = allcat_template_path.name
    match = re.match(r"^(?P<prefix>.+?)(?P<suffix>_s-.*\.txt)$", name)
    if match is None:
        raise ValueError(
            "Cannot infer per-N_s allcat path from template name. "
            f"Expected '*_s-...txt', got {name}"
        )
    prefix = re.sub(r"_ns[0-9p]+$", "", match.group("prefix"))
    suffix = match.group("suffix")
    ns_tag = _ns_tag(ns_value)
    return model_output_root / f"ns{ns_tag}" / f"{prefix}_ns{ns_tag}{suffix}"


def _resolve_model_inputs_from_out_dir(out_dir: Path) -> Tuple[Path, Path]:
    model_root = out_dir.resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"Model output directory does not exist: {model_root}")
    if not model_root.is_dir():
        raise NotADirectoryError(f"Model output path is not a directory: {model_root}")

    allcat_candidates = sorted(model_root.glob("allcat_s-*.txt"))
    if len(allcat_candidates) == 0:
        raise FileNotFoundError(
            f"Missing root allcat file in {model_root}. Expected exactly one file matching allcat_s-*.txt."
        )
    if len(allcat_candidates) > 1:
        names = ", ".join(path.name for path in allcat_candidates)
        raise RuntimeError(
            f"Found multiple root allcat files in {model_root}; expected exactly one: {names}"
        )

    mpb_path = model_root / "mpb_from_fixed_trees.csv"
    if not mpb_path.exists():
        raise FileNotFoundError(f"Missing MPB catalog in {model_root}: {mpb_path}")
    return allcat_candidates[0].resolve(), mpb_path.resolve()


def _find_final_gcs_file(allcat_ns_path: Path) -> Path:
    match = re.search(r"_ns([0-9]+p[0-9]+)", allcat_ns_path.stem)
    if match is None:
        raise ValueError(f"Could not infer N_s tag from {allcat_ns_path.name}")
    path = allcat_ns_path.parent / f"finalGCs_ns{match.group(1)}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing finalGCs file for {allcat_ns_path.name}: {path}")
    return path


def load_allcat(allcat_path: Path) -> pd.DataFrame:
    """Load one allcat table and add derived mass columns."""

    columns = _read_comment_columns(allcat_path)
    raw = pd.read_csv(
        allcat_path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )
    if raw.shape[1] < len(ALLCAT_REQUIRED_COLUMNS):
        raise ValueError(
            f"Allcat file has {raw.shape[1]} columns; expected at least {len(ALLCAT_REQUIRED_COLUMNS)}"
        )

    n_keep = min(raw.shape[1], len(columns))
    raw = raw.iloc[:, :n_keep].copy()
    raw.columns = columns[:n_keep]
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    gc = raw.dropna(subset=ALLCAT_REQUIRED_COLUMNS).copy()
    gc["hid_z0"] = gc["hid_z0"].astype(int)
    gc["isMPB"] = gc["isMPB"].astype(int)
    gc["subfind_form"] = gc["subfind_form"].astype(int)
    gc["snap_form"] = gc["snap_form"].astype(int)
    gc["M_form"] = np.power(10.0, gc["logM_form"].to_numpy(dtype=float))
    gc["M_halo_z0"] = np.power(10.0, gc["logMh_z0"].to_numpy(dtype=float))
    gc["M_halo_form"] = np.power(10.0, gc["logMh_form"].to_numpy(dtype=float))
    gc["M_star_z0"] = np.power(10.0, gc["logMstar_z0"].to_numpy(dtype=float))
    gc["M_star_form"] = np.power(10.0, gc["logMstar_form"].to_numpy(dtype=float))
    return gc.reset_index(drop=True)


def load_mpb(mpb_path: Path) -> pd.DataFrame:
    """Load the MPB table and add convenience columns."""

    mpb = pd.read_csv(mpb_path)
    for col in ["subhalo_id_z0", "SnapNum"]:
        if col not in mpb.columns:
            raise ValueError(f"MPB table is missing required column '{col}': {mpb_path}")
    mpb["subhalo_id_z0"] = pd.to_numeric(mpb["subhalo_id_z0"], errors="coerce").astype(int)
    mpb["SnapNum"] = pd.to_numeric(mpb["SnapNum"], errors="coerce").astype(int)
    if {"SubhaloSpin_x", "SubhaloSpin_y", "SubhaloSpin_z"}.issubset(mpb.columns):
        mpb["spin_mag"] = np.sqrt(
            np.square(pd.to_numeric(mpb["SubhaloSpin_x"], errors="coerce"))
            + np.square(pd.to_numeric(mpb["SubhaloSpin_y"], errors="coerce"))
            + np.square(pd.to_numeric(mpb["SubhaloSpin_z"], errors="coerce"))
        )
        mpb["spin_mag"] = np.where(np.isfinite(mpb["spin_mag"]), mpb["spin_mag"], 500.0)
    else:
        mpb["spin_mag"] = 500.0
    if "logMh_msun_h" in mpb.columns:
        mpb["logMh_msun_h"] = pd.to_numeric(mpb["logMh_msun_h"], errors="coerce")
    if "Redshift" in mpb.columns:
        mpb["Redshift"] = pd.to_numeric(mpb["Redshift"], errors="coerce")
    return mpb


def _load_final_gcs_table(path: Path, expected_len: int, expected_halo_ids: np.ndarray) -> pd.DataFrame:
    """Load the merged finalGCs table and validate row ordering."""

    columns = _read_comment_columns(path)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )
    df = df.iloc[:, : len(columns)].copy()
    df.columns = columns[: df.shape[1]]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) != expected_len:
        raise ValueError(f"{path} has {len(df)} rows, expected {expected_len}")

    halo_ids = df["halo_id_z0"].to_numpy(dtype=int)
    if not np.array_equal(halo_ids, np.asarray(expected_halo_ids, dtype=int)):
        raise ValueError(f"Row-order mismatch between {path} and the matching allcat_ns file")

    expected_gc_index = np.empty(expected_len, dtype=int)
    halo_ids_arr = np.asarray(expected_halo_ids, dtype=int)
    for hid in np.unique(halo_ids_arr):
        idx = np.where(halo_ids_arr == int(hid))[0]
        expected_gc_index[idx] = np.arange(1, len(idx) + 1, dtype=int)
    gc_index = df["gc_index_halo"].to_numpy(dtype=int)
    if not np.array_equal(gc_index, expected_gc_index):
        raise ValueError(f"GC index ordering mismatch between {path} and the matching allcat_ns file")

    df["status"] = df["status"].astype(int)
    df["halo_id_z0"] = df["halo_id_z0"].astype(int)
    df["gc_index_halo"] = df["gc_index_halo"].astype(int)
    if "m_final_msun" in df.columns:
        df["m_final_msun"] = np.where(
            np.isfinite(df["m_final_msun"]) & (df["m_final_msun"] > 0.0),
            df["m_final_msun"],
            0.0,
        )
        m_final = df["m_final_msun"].to_numpy(dtype=float)
        log_m_final = np.full(len(m_final), np.nan, dtype=float)
        positive = m_final > 0.0
        log_m_final[positive] = np.log10(m_final[positive])
        df["log10_m_final_msun"] = log_m_final
    return df.reset_index(drop=True)


def _load_halo_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["hid_z0"] = df["hid_z0"].astype(int)
    return df.sort_values("hid_z0").reset_index(drop=True)


def cosmic_time_gyr(z: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(z, dtype=float)
    flat = np.array([Redshift2CosmicTimeGyr(float(value)) for value in arr.ravel()], dtype=float)
    out = flat.reshape(arr.shape)
    if np.isscalar(z):
        return float(out)
    return out


def stellar_mass_from_halo_mass(halo_mass: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
    mh = np.asarray(halo_mass, dtype=float)
    zz = np.asarray(z, dtype=float)
    zz = np.broadcast_to(zz, mh.shape)
    out = np.empty(mh.size, dtype=float)
    flat_mh = mh.ravel()
    flat_z = zz.ravel()
    for i, (mass, redshift) in enumerate(zip(flat_mh, flat_z)):
        out[i] = smhm.SMHM(float(mass), float(redshift), scatter=False) if mass > 0.0 else np.nan
    out = out.reshape(mh.shape)
    if np.isscalar(halo_mass):
        return float(out)
    return out


def present_day_stellar_mass_from_halo_mass(halo_mass: np.ndarray | float) -> np.ndarray | float:
    """Use the Kravtsov+2014 z=0 correction for present-day plotting only."""

    mh = np.asarray(halo_mass, dtype=float)
    out = np.empty(mh.size, dtype=float)
    flat_mh = mh.ravel()
    for i, mass in enumerate(flat_mh):
        out[i] = smhm.SMHM(float(mass), 0.0, k=True, scatter=False, mdef="mvir") if mass > 0.0 else np.nan
    out = out.reshape(mh.shape)
    if np.isscalar(halo_mass):
        return float(out)
    return out


_PRESENT_DAY_SMHM_INVERSE_CACHE: Tuple[np.ndarray, np.ndarray] | None = None


def present_day_halo_mass_from_observed_stellar_mass(stellar_mass: np.ndarray | float) -> np.ndarray | float:
    global _PRESENT_DAY_SMHM_INVERSE_CACHE
    if _PRESENT_DAY_SMHM_INVERSE_CACHE is None:
        log_mh_grid = np.linspace(8.0, 16.0, 4096)
        mh_grid = np.power(10.0, log_mh_grid)
        mstar_grid = np.array([present_day_stellar_mass_from_halo_mass(float(mh)) for mh in mh_grid], dtype=float)
        log_mstar_grid = np.log10(np.clip(mstar_grid, 1.0e-30, None))
        order = np.argsort(log_mstar_grid)
        _PRESENT_DAY_SMHM_INVERSE_CACHE = (log_mstar_grid[order], log_mh_grid[order])

    log_mstar_grid, log_mh_grid = _PRESENT_DAY_SMHM_INVERSE_CACHE
    sm = np.asarray(stellar_mass, dtype=float)
    valid = np.isfinite(sm) & (sm > 0.0)
    out = np.full(sm.shape, np.nan, dtype=float)
    out[valid] = np.power(
        10.0,
        np.interp(np.log10(sm[valid]), log_mstar_grid, log_mh_grid, left=np.nan, right=np.nan),
    )
    if np.isscalar(stellar_mass):
        return float(out)
    return out


def metallicity_mmr(log_mstar: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
    log_sm = np.asarray(log_mstar, dtype=float)
    zz = np.asarray(z, dtype=float)
    zz = np.broadcast_to(zz, log_sm.shape)
    feh = MMR_SLOPE * (log_sm - MMR_TURNOVER) - MMR_EVOLUTION * np.log10(1.0 + zz)
    feh = np.minimum(feh, MAX_FEH)
    if np.isscalar(log_mstar):
        return float(feh)
    return feh


def gas_mass_from_stellar_halo(stellar_mass: np.ndarray | float, halo_mass: np.ndarray | float, z: np.ndarray | float) -> np.ndarray | float:
    sm = np.asarray(stellar_mass, dtype=float)
    mh = np.asarray(halo_mass, dtype=float)
    zz = np.asarray(z, dtype=float)
    zz = np.broadcast_to(zz, sm.shape)
    mh = np.broadcast_to(mh, sm.shape)

    slope = np.where(sm < 1.0e9, 0.19, 0.33)
    log_ratio = 0.05 - 0.5 - slope * (np.log10(np.clip(sm, 1.0e-30, None)) - 9.0)

    mask_low = zz < 2.0
    mask_mid = (zz >= 2.0) & (zz < 3.0)
    mask_high = zz >= 3.0
    log_ratio = log_ratio.copy()
    log_ratio[mask_low] += (3.0 - TDEP) * np.log10((1.0 + zz[mask_low]) / 3.0) + (3.0 - TDEP) * np.log10(3.0)
    log_ratio[mask_mid] += (1.7 - TDEP) * np.log10((1.0 + zz[mask_mid]) / 3.0) + (3.0 - TDEP) * np.log10(3.0)
    log_ratio[mask_high] += (1.7 - TDEP) * np.log10(4.0 / 3.0) + (3.0 - TDEP) * np.log10(3.0)

    mg = sm * np.power(10.0, log_ratio)
    fstar = sm / np.clip(FB * mh, 1.0e-30, None)
    fgas = mg / np.clip(FB * mh, 1.0e-30, None)

    e_of_z = np.array([smhm.E(float(redshift)) for redshift in zz.ravel()], dtype=float).reshape(zz.shape)
    mc = 3.6e9 * np.exp(-0.6 * (1.0 + zz)) / H100
    mc_min = 1.5e10 * np.power(180.0, -0.5) / np.clip(e_of_z * H100, 1.0e-30, None)
    mc = np.maximum(mc, mc_min)
    fin = 1.0 / np.power(1.0 + mc / np.clip(mh, 1.0e-30, None), 3.0)
    overflow = fstar + fgas > fin
    fgas = np.where(overflow, np.maximum(fin - fstar, 0.0), fgas)
    mg = fgas * FB * mh

    if np.isscalar(stellar_mass):
        return float(mg)
    return mg


def _solve_gaussian_crossing(mu1: float, sig1: float, w1: float, mu2: float, sig2: float, w2: float) -> float:
    a = 0.5 / (sig2 * sig2) - 0.5 / (sig1 * sig1)
    b = mu1 / (sig1 * sig1) - mu2 / (sig2 * sig2)
    c = (
        0.5 * mu2 * mu2 / (sig2 * sig2)
        - 0.5 * mu1 * mu1 / (sig1 * sig1)
        + np.log(np.clip((w2 * sig1) / np.clip(w1 * sig2, 1.0e-30, None), 1.0e-30, None))
    )
    if abs(a) < 1.0e-12:
        if abs(b) < 1.0e-12:
            return 0.5 * (mu1 + mu2)
        return -c / b
    roots = np.roots([a, b, c])
    real_roots = [float(root.real) for root in roots if abs(root.imag) < 1.0e-8]
    interior = [root for root in real_roots if min(mu1, mu2) <= root <= max(mu1, mu2)]
    if interior:
        return interior[0]
    if real_roots:
        return min(real_roots, key=lambda value: abs(value - 0.5 * (mu1 + mu2)))
    return 0.5 * (mu1 + mu2)


def fit_metallicity_split(values: Iterable[float], min_count: int = MIN_GMM_COUNT) -> Tuple[float, float, float]:
    """Return split threshold, blue peak, and red peak for a metallicity sample."""

    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr >= FEH_MIN) & (arr <= FEH_MAX)]
    if len(arr) < min_count:
        return GLOBAL_SPLIT_DEFAULT, np.nan, np.nan

    gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    gm.fit(arr.reshape(-1, 1))
    means = gm.means_.ravel()
    variances = gm.covariances_.reshape(-1)
    weights = gm.weights_.ravel()
    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    weights = weights[order]
    sigmas = np.sqrt(np.clip(variances, 1.0e-8, None))

    split = _solve_gaussian_crossing(means[0], sigmas[0], weights[0], means[1], sigmas[1], weights[1])
    split = float(np.clip(split, FEH_MIN, FEH_MAX))
    blue_peak = float(means[0])
    red_peak = float(means[1]) if means[1] >= -1.0 else np.nan
    return split, blue_peak, red_peak


def _population_from_threshold(feh: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(np.where(feh.to_numpy(dtype=float) <= threshold, "blue", "red"), index=feh.index)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: subprocess.CalledProcessError | None = None
    for _ in range(3):
        try:
            subprocess.run(
                ["wget", "-c", "--tries=3", "--waitretry=2", "-O", str(destination), url],
                check=True,
            )
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


def _ensure_choksi_supplement(obs_cache_dir: Path, overwrite: bool = False) -> Dict[str, Path]:
    raw_zip = obs_cache_dir / "cgl18_supplemental_wayback.zip"
    extract_dir = obs_cache_dir / "cgl18_supplemental_wayback"
    normalized_dir = obs_cache_dir / "choksi_supplement"
    data_path = normalized_dir / "data.txt"
    model_zip_path = normalized_dir / "model.txt.zip"
    model_path = normalized_dir / "model.txt"

    if overwrite or not raw_zip.exists():
        _download_file(CHOKSI_SUPP_WAYBACK_URL, raw_zip)

    if overwrite or not (extract_dir / "cgl18_supplemental" / "data.txt").exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw_zip) as zf:
            zf.extractall(extract_dir)

    normalized_dir.mkdir(parents=True, exist_ok=True)
    extracted_data = extract_dir / "cgl18_supplemental" / "data.txt"
    extracted_model_zip = extract_dir / "cgl18_supplemental" / "model.txt.zip"
    if overwrite or not data_path.exists():
        shutil.copy2(extracted_data, data_path)
    if overwrite or not model_zip_path.exists():
        shutil.copy2(extracted_model_zip, model_zip_path)
    if overwrite or not model_path.exists():
        with zipfile.ZipFile(model_zip_path) as zf:
            member = next(name for name in zf.namelist() if name.endswith("model.txt"))
            with zf.open(member) as src, model_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    return {"data": data_path, "model": model_path, "model_zip": model_zip_path}


def _ensure_acsvcs_tables(obs_cache_dir: Path, overwrite: bool = False) -> Dict[str, Path]:
    acsvcs_dir = obs_cache_dir / "acsvcs"
    hosts_path = acsvcs_dir / "hosts_J_ApJS_164_334_acsvcs.tsv"
    gc_path = acsvcs_dir / "gc_catalog_J_ApJS_180_54_table4.tsv"
    if overwrite or not hosts_path.exists():
        _download_file(ACSVCS_HOSTS_URL, hosts_path)
    if overwrite or not gc_path.exists():
        _download_file(ACSVCS_GC_URL, gc_path)
    return {"hosts": hosts_path, "gc_catalog": gc_path}


def _clean_tex_text(value: str) -> str:
    cleaned = value.strip()
    replacements = {
        r"\phantom{0}": "",
        r"\phantom{1}": "",
        r"\pm": "±",
        r"\sim": "~",
        r"\,$": "",
        r"\,": "",
        r"\FeH": "[Fe/H]",
        r"\Rgcst": "R_gc_st",
        r"\msun": "M_sun",
        r"M$\,$": "M ",
        r"NGC$\,$": "NGC ",
        r"Ter$\,$": "Ter ",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("{", "")
    cleaned = cleaned.replace("}", "")
    cleaned = cleaned.replace("^", "")
    cleaned = cleaned.replace("_", "")
    cleaned = cleaned.replace("\\", "")
    cleaned = " ".join(cleaned.split())
    return cleaned


def _parse_pm_value(value: str) -> Tuple[float, float]:
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", _clean_tex_text(value))
    if len(numbers) < 2:
        raise ValueError(f"Cannot parse symmetric uncertainty from '{value}'")
    return float(numbers[0]), float(numbers[1])


def _parse_asymmetric_value(value: str) -> Tuple[float, float, float]:
    cleaned = _clean_tex_text(value)
    if "±" in cleaned:
        center, err = _parse_pm_value(value)
        return center, err, err
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if len(numbers) < 3:
        raise ValueError(f"Cannot parse asymmetric uncertainty from '{value}'")
    return float(numbers[0]), abs(float(numbers[2])), abs(float(numbers[1]))


def _download_and_extract_arxiv_source(url: str, raw_path: Path, extract_dir: Path, overwrite: bool = False) -> None:
    if overwrite or not raw_path.exists():
        _download_file(url, raw_path)
    marker = extract_dir / ".extracted"
    if overwrite or not marker.exists():
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(raw_path, "r:*") as tar:
            tar.extractall(extract_dir)
        marker.write_text(f"{url}\n", encoding="utf-8")


def _parse_deluxetable_rows(tex_text: str, caption_fragment: str) -> List[List[str]]:
    start = tex_text.index(caption_fragment)
    start = tex_text.index(r"\startdata", start)
    end = tex_text.index(r"\enddata", start)
    block = tex_text[start + len(r"\startdata") : end]
    rows: List[List[str]] = []
    current = ""
    for raw_line in block.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("%") or stripped.startswith(r"\noalign") or stripped.startswith(r"\multispan"):
            continue
        current += " " + stripped
        if stripped.endswith(r"\\"):
            row = current.rsplit(r"\\", 1)[0].strip()
            current = ""
            if "&" in row:
                rows.append([field.strip() for field in row.split("&")])
    return rows


def _parse_tabular_rows(tex_text: str, caption_fragment: str) -> List[List[str]]:
    start = tex_text.index(caption_fragment)
    start = tex_text.index(r"\begin{tabular}", start)
    end = tex_text.index(r"\end{tabular}", start)
    block = tex_text[start:end]
    rows: List[List[str]] = []
    for raw_line in block.splitlines():
        stripped = raw_line.strip()
        if (
            not stripped
            or stripped.startswith("%")
            or stripped.startswith("\\")
            or "&" not in stripped
            or not stripped.endswith(r"\\")
        ):
            continue
        rows.append([field.strip() for field in stripped[:-2].split("&")])
    return rows


def _read_text_flexible(path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _ensure_vandenberg_2013_tables(obs_cache_dir: Path, overwrite: bool = False) -> Dict[str, Path]:
    dataset_dir = obs_cache_dir / "vandenberg2013"
    raw_path = dataset_dir / "1308.2257.tar.gz"
    extract_dir = dataset_dir / "src"
    csv_path = dataset_dir / "table2_gc_ages.csv"
    tex_path = extract_dir / "ms.tex"

    _download_and_extract_arxiv_source(VANDENBERG_2013_ARXIV_URL, raw_path, extract_dir, overwrite)
    if overwrite or not csv_path.exists():
        tex_text = _read_text_flexible(tex_path)
        rows = _parse_deluxetable_rows(tex_text, r"\tablecaption{Ages and Other Properties of the Globular Cluster Sample")
        parsed_rows: List[dict] = []
        for row in rows:
            ngc = _clean_tex_text(row[0])
            name = _clean_tex_text(row[1])
            feh = float(re.findall(r"[-+]?\d+(?:\.\d+)?", _clean_tex_text(row[2]))[0])
            age_gyr, age_err_gyr = _parse_pm_value(row[3])
            display_name = name if name else (f"NGC {ngc}" if ngc else "unknown")
            parsed_rows.append(
                {
                    "ngc_id": ngc if ngc else "",
                    "cluster": display_name,
                    "feh": feh,
                    "age_gyr": age_gyr,
                    "age_err_gyr": age_err_gyr,
                    "source_note": "VandenBerg et al. 2013 Table 2",
                }
            )
        pd.DataFrame(parsed_rows).to_csv(csv_path, index=False)
    return {"raw": raw_path, "tex": tex_path, "table2_csv": csv_path}


def _ensure_leaman_2013_source(obs_cache_dir: Path, overwrite: bool = False) -> Dict[str, Path]:
    dataset_dir = obs_cache_dir / "leaman2013"
    raw_path = dataset_dir / "1309.0822.tar.gz"
    extract_dir = dataset_dir / "src"
    tex_path = extract_dir / "gcamraph.tex"
    pdf_path = extract_dir / "cmdisohalo.pdf"
    _download_and_extract_arxiv_source(LEAMAN_2013_ARXIV_URL, raw_path, extract_dir, overwrite)
    return {"raw": raw_path, "tex": tex_path, "outer_halo_pdf": pdf_path}


def _ensure_wagner_kaiser_2017_tables(obs_cache_dir: Path, overwrite: bool = False) -> Dict[str, Path]:
    dataset_dir = obs_cache_dir / "wagner_kaiser2017"
    raw_path = dataset_dir / "1707.01571.tar.gz"
    extract_dir = dataset_dir / "src"
    csv_path = dataset_dir / "lmc_gc_age_metallicity.csv"
    tex_path = extract_dir / "LMCI.tex"

    _download_and_extract_arxiv_source(WAGNER_KAISER_2017_ARXIV_URL, raw_path, extract_dir, overwrite)
    if overwrite or not csv_path.exists():
        tex_text = _read_text_flexible(tex_path)
        feh_rows = _parse_tabular_rows(tex_text, r"\caption{Assumed metallicities for our target LMC clusters}")
        age_rows = _parse_tabular_rows(tex_text, r"\caption{Relative ages for our LMC cluster sample}")

        feh_map: Dict[str, dict] = {}
        for row in feh_rows:
            if "Cluster" in row[0]:
                continue
            cluster = _clean_tex_text(row[0]).replace(" ", "")
            feh_center, feh_err = _parse_pm_value(row[4])
            feh_map[cluster] = {"feh": feh_center, "feh_err": feh_err}

        parsed_rows: List[dict] = []
        for row in age_rows:
            if "Cluster" in row[0]:
                continue
            cluster_display = _clean_tex_text(row[0])
            cluster_key = cluster_display.replace(" ", "")
            age_center, age_err_lo, age_err_hi = _parse_asymmetric_value(row[6])
            parsed_rows.append(
                {
                    "cluster": cluster_display,
                    "feh": feh_map[cluster_key]["feh"],
                    "feh_err": feh_map[cluster_key]["feh_err"],
                    "age_gyr": age_center,
                    "age_err_lo_gyr": age_err_lo,
                    "age_err_hi_gyr": age_err_hi,
                    "source_note": "Wagner-Kaiser et al. 2017 Tables 1 and 2",
                }
            )
        pd.DataFrame(parsed_rows).to_csv(csv_path, index=False)
    return {"raw": raw_path, "tex": tex_path, "lmc_csv": csv_path}


def _ensure_lamers_2017_tables(obs_cache_dir: Path, overwrite: bool = False) -> Dict[str, Path]:
    dataset_dir = obs_cache_dir / "lamers2017"
    raw_path = dataset_dir / "1706.00939.tar.gz"
    extract_dir = dataset_dir / "src"
    csv_path = dataset_dir / "table1_mdf_summary.csv"
    tex_path = extract_dir / "Lamers-AA_2017_31062.tex"

    _download_and_extract_arxiv_source(LAMERS_2017_ARXIV_URL, raw_path, extract_dir, overwrite)
    if overwrite or not csv_path.exists():
        tex_text = _read_text_flexible(tex_path)
        start = tex_text.index(r"\begin{table*} \label{tbl:summary}")
        end = tex_text.index(r"\end{tabular}", start)
        block = tex_text[start:end]
        parsed_rows: List[dict] = []
        current_galaxy = ""
        current_boundary = ""
        for raw_line in block.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("\\") or "&" not in stripped or not stripped.endswith(r"\\"):
                continue
            fields = [field.strip() for field in stripped[:-2].split("&")]
            if len(fields) != 7:
                continue
            galaxy = _clean_tex_text(fields[0]) or current_galaxy
            boundary = _clean_tex_text(fields[1]) or current_boundary
            current_galaxy = galaxy
            current_boundary = boundary
            object_type = _clean_tex_text(fields[2])
            if galaxy == "Galaxy" or object_type == "Objects":
                continue
            parsed_rows.append(
                {
                    "galaxy": galaxy,
                    "boundary": boundary,
                    "object_type": object_type,
                    "inner_feh_lt_minus1": _clean_tex_text(fields[3]),
                    "inner_feh_gt_minus1": _clean_tex_text(fields[4]),
                    "outer_feh_lt_minus1": _clean_tex_text(fields[5]),
                    "outer_feh_gt_minus1": _clean_tex_text(fields[6]),
                    "source_note": "Lamers et al. 2017 Table 1",
                }
            )
        pd.DataFrame(parsed_rows).to_csv(csv_path, index=False)
    return {"raw": raw_path, "tex": tex_path, "summary_csv": csv_path}


def ensure_obs_downloads(obs_cache_dir: Path = DEFAULT_OBS_CACHE_DIR) -> Dict[str, Path]:
    """Download only the observation files that are reproducibly accessible here."""

    obs_cache_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    paths.update({f"choksi_{key}": value for key, value in _ensure_choksi_supplement(obs_cache_dir, overwrite=False).items()})
    paths.update({f"acsvcs_{key}": value for key, value in _ensure_acsvcs_tables(obs_cache_dir, overwrite=False).items()})
    paths.update({f"vandenberg2013_{key}": value for key, value in _ensure_vandenberg_2013_tables(obs_cache_dir, overwrite=False).items()})
    paths.update({f"leaman2013_{key}": value for key, value in _ensure_leaman_2013_source(obs_cache_dir, overwrite=False).items()})
    paths.update({f"wagner2017_{key}": value for key, value in _ensure_wagner_kaiser_2017_tables(obs_cache_dir, overwrite=False).items()})
    paths.update({f"lamers2017_{key}": value for key, value in _ensure_lamers_2017_tables(obs_cache_dir, overwrite=False).items()})
    return paths


def load_choksi_system_table(path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            galaxy_id, log_sm, mean_feh, err_mean, sigma_feh, err_sigma, blue_peak, red_peak = line.split()[:8]
            rows.append(
                {
                    "galaxyID": galaxy_id,
                    "logSM": float(log_sm),
                    "mean_feh": float(mean_feh),
                    "err_mean": float(err_mean),
                    "sigma_feh": float(sigma_feh),
                    "err_sigma": float(err_sigma),
                    "blue_peak": float(blue_peak),
                    "red_peak": np.nan if float(red_peak) > 1000.0 else float(red_peak),
                }
            )
    systems = pd.DataFrame(rows)
    systems["dataset"] = np.select(
        [
            systems["galaxyID"].str.startswith("VCS"),
            systems["galaxyID"].str.startswith("HST_BCG"),
            systems["galaxyID"].isin(["MW", "M31"]),
        ],
        ["VCS", "HST_BCG", "LG"],
        default="other",
    )
    systems["VCC"] = np.nan
    vcs_mask = systems["dataset"] == "VCS"
    systems.loc[vcs_mask, "VCC"] = (
        systems.loc[vcs_mask, "galaxyID"]
        .str.replace("VCS", "", regex=False)
        .str.replace(".0", "", regex=False)
        .astype(int)
    )
    systems["M_star_msun"] = np.power(10.0, systems["logSM"].to_numpy(dtype=float))
    systems["M_halo_plot_msun"] = present_day_halo_mass_from_observed_stellar_mass(systems["M_star_msun"].to_numpy(dtype=float))
    systems["logMh_plot"] = np.log10(np.clip(systems["M_halo_plot_msun"].to_numpy(dtype=float), 1.0e-30, None))
    return systems


def _read_vizier_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#")


def load_acsvcs_hosts(path: Path) -> pd.DataFrame:
    hosts = _read_vizier_tsv(path)
    for col in ["VCC", "BTmag", "E(B-V)", "Vsys"]:
        if col in hosts.columns:
            hosts[col] = pd.to_numeric(hosts[col], errors="coerce")
    hosts = hosts.dropna(subset=["VCC"]).copy()
    hosts["VCC"] = hosts["VCC"].astype(int)
    return hosts


def _vcs_color_to_feh(g_minus_z: np.ndarray) -> np.ndarray:
    colour = np.asarray(g_minus_z, dtype=float)
    disc = 0.481 * 0.481 - 4.0 * 0.051 * (1.513 - colour)
    disc = np.clip(disc, 0.0, None)
    return (-0.481 + np.sqrt(disc)) / (2.0 * 0.051)


def _zmag_to_mass_proxy(zmag: np.ndarray) -> np.ndarray:
    zmag = np.asarray(zmag, dtype=float)
    abs_mag = zmag - VIRGO_DISTANCE_MODULUS
    lum = np.power(10.0, -0.4 * (abs_mag - ACS_SOLAR_MAG_Z))
    return GC_ML_Z * lum


def load_acsvcs_gc_catalog(path: Path) -> pd.DataFrame:
    gc = _read_vizier_tsv(path)
    numeric_cols = ["VCC", "RAJ2000", "DEJ2000", "GDist", "zmag", "zamag", "gmag", "gamag", "rhz", "rhg", "pGC", "E(B-V)"]
    for col in numeric_cols:
        if col in gc.columns:
            gc[col] = pd.to_numeric(gc[col], errors="coerce")
    gc = gc.dropna(subset=["VCC"]).copy()
    gc["VCC"] = gc["VCC"].astype(int)
    g_use = np.where(np.isfinite(gc["gamag"]), gc["gamag"], gc["gmag"])
    z_use = np.where(np.isfinite(gc["zamag"]), gc["zamag"], gc["zmag"])
    gc["g_minus_z"] = g_use - z_use
    gc["feh"] = _vcs_color_to_feh(gc["g_minus_z"].to_numpy(dtype=float))
    gc["feh"] = np.where((gc["feh"] >= FEH_MIN) & (gc["feh"] <= FEH_MAX), gc["feh"], np.nan)
    gc["m_gc_proxy_msun"] = _zmag_to_mass_proxy(z_use)
    return gc


def load_observations(obs_cache_dir: Path = DEFAULT_OBS_CACHE_DIR) -> ObsCatalog:
    paths = ensure_obs_downloads(obs_cache_dir)
    systems = load_choksi_system_table(paths["choksi_data"])
    hosts = load_acsvcs_hosts(paths["acsvcs_hosts"])
    gc = load_acsvcs_gc_catalog(paths["acsvcs_gc_catalog"])
    vcs_systems = systems.loc[systems["dataset"] == "VCS"].copy()
    vcs_systems["VCC"] = vcs_systems["VCC"].astype(int)
    mw_age_metallicity = pd.read_csv(paths["vandenberg2013_table2_csv"])
    lmc_age_metallicity = pd.read_csv(paths["wagner2017_lmc_csv"])
    lamers_summary = pd.read_csv(paths["lamers2017_summary_csv"])
    return ObsCatalog(
        systems=systems,
        vcs_systems=vcs_systems,
        acsvcs_hosts=hosts,
        acsvcs_gc=gc,
        mw_age_metallicity=mw_age_metallicity,
        lmc_age_metallicity=lmc_age_metallicity,
        lamers_summary=lamers_summary,
        obs_cache_dir=obs_cache_dir,
    )


def build_model_catalog(allcat_template_path: Path, mpb_path: Path, ns_value: float) -> ModelCatalog:
    allcat_path = _build_ns_allcat_path(allcat_template_path, ns_value)
    final_gcs_path = _find_final_gcs_file(allcat_path)
    formed = load_allcat(allcat_path)
    final_gcs = _load_final_gcs_table(final_gcs_path, len(formed), formed["hid_z0"].to_numpy(dtype=int))

    keep_cols = [col for col in ["status", "m_final_msun", "log10_m_final_msun", "m_init_msun", "r_final_kpc"] if col in final_gcs.columns]
    catalog = formed.join(final_gcs[keep_cols])
    if "m_final_msun" not in catalog.columns:
        catalog["m_final_msun"] = np.nan
    if "log10_m_final_msun" not in catalog.columns:
        m_final = catalog["m_final_msun"].to_numpy(dtype=float)
        log_m_final = np.full(len(m_final), np.nan, dtype=float)
        positive = m_final > 0.0
        log_m_final[positive] = np.log10(m_final[positive])
        catalog["log10_m_final_msun"] = log_m_final
    catalog["status"] = catalog["status"].fillna(0).astype(int)

    survivors = catalog.loc[catalog["status"] == 1].copy().reset_index(drop=True)
    split_threshold, _, _ = fit_metallicity_split(survivors["feh"].to_numpy(dtype=float))
    survivors["population"] = _population_from_threshold(survivors["feh"], split_threshold)
    survivors["logM_final"] = np.log10(np.clip(survivors["m_final_msun"].to_numpy(dtype=float), 1.0e-30, None))
    survivors["t_form_gyr"] = cosmic_time_gyr(survivors["zform"].to_numpy(dtype=float))
    survivors["M_gas_form"] = gas_mass_from_stellar_halo(
        np.power(10.0, survivors["logMstar_form"].to_numpy(dtype=float)),
        np.power(10.0, survivors["logMh_form"].to_numpy(dtype=float)),
        survivors["zform"].to_numpy(dtype=float),
    )
    survivors["logMgas_form"] = np.log10(np.clip(survivors["M_gas_form"].to_numpy(dtype=float), 1.0e-30, None))

    halo_summary_path = allcat_path.parent / f"haloSummary_ns{_ns_tag(ns_value)}.csv"
    halo_summary = _load_halo_summary(halo_summary_path)
    mpb = load_mpb(mpb_path)
    run_metadata = _load_run_metadata(allcat_template_path)
    return ModelCatalog(
        formed=formed,
        catalog=catalog,
        survivors=survivors,
        halo_summary=halo_summary,
        mpb=mpb,
        allcat_path=allcat_path,
        final_gcs_path=final_gcs_path,
        run_metadata=run_metadata,
        split_threshold=split_threshold,
    )


def load_choksi_paper_model(path: Path = CHOKSI_MODEL_PATH) -> PaperModelCatalog:
    if not path.exists():
        raise FileNotFoundError(f"Missing Choksi+2018 model catalog: {path}")
    columns = [
        "hid_z0",
        "logMh_z0",
        "logMstar_z0",
        "logMh_form",
        "logMstar_form",
        "logM_final",
        "logM_form",
        "zform",
        "cluster_age_gyr",
        "feh",
        "isMPB",
    ]
    survivors = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=columns, engine="python")
    for col in columns:
        survivors[col] = pd.to_numeric(survivors[col], errors="coerce")
    survivors = survivors.dropna(subset=columns).copy()
    survivors["hid_z0"] = survivors["hid_z0"].astype(int)
    survivors["isMPB"] = survivors["isMPB"].astype(int)
    survivors["m_final_msun"] = np.power(10.0, survivors["logM_final"].to_numpy(dtype=float))
    survivors["M_gas_form"] = gas_mass_from_stellar_halo(
        np.power(10.0, survivors["logMstar_form"].to_numpy(dtype=float)),
        np.power(10.0, survivors["logMh_form"].to_numpy(dtype=float)),
        survivors["zform"].to_numpy(dtype=float),
    )
    survivors["logMgas_form"] = np.log10(np.clip(survivors["M_gas_form"].to_numpy(dtype=float), 1.0e-30, None))
    survivors["t_form_gyr"] = T_UNIVERSE_GYR - survivors["cluster_age_gyr"].to_numpy(dtype=float)
    survivors["t_form_gyr"] = np.where(np.isfinite(survivors["t_form_gyr"]), survivors["t_form_gyr"], np.nan)
    split_threshold, _, _ = fit_metallicity_split(survivors["feh"].to_numpy(dtype=float))
    survivors["population"] = _population_from_threshold(survivors["feh"], split_threshold)
    return PaperModelCatalog(survivors=survivors.reset_index(drop=True), split_threshold=split_threshold)


def _build_halo_level_table_from_survivors(
    survivors: pd.DataFrame,
    halo_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    halo_table = (
        survivors.groupby("hid_z0", sort=True)
        .agg(
            logMh_z0=("logMh_z0", "first"),
            logMstar_z0=("logMstar_z0", "first"),
            n_gc=("feh", "size"),
            mean_feh=("feh", "mean"),
            sigma_feh=("feh", lambda values: float(np.std(values.to_numpy(dtype=float), ddof=0))),
            blue_fraction=("population", lambda values: float(np.mean(values == "blue"))),
            M_gc_final=("m_final_msun", "sum"),
        )
        .reset_index()
    )
    if halo_summary is not None and "m_gc_final_total_msun" in halo_summary.columns:
        halo_table = halo_table.merge(
            halo_summary[["hid_z0", "m_gc_final_total_msun"]],
            on="hid_z0",
            how="left",
        )
        halo_table["M_gc_final"] = halo_table["m_gc_final_total_msun"].fillna(halo_table["M_gc_final"])
        halo_table = halo_table.drop(columns=["m_gc_final_total_msun"])
    halo_table["logM_gc_final"] = np.log10(np.clip(halo_table["M_gc_final"].to_numpy(dtype=float), 1.0e-30, None))
    return halo_table


def build_halo_level_table(model: ModelCatalog) -> pd.DataFrame:
    return _build_halo_level_table_from_survivors(model.survivors, model.halo_summary)


def build_paper_halo_level_table(paper_model: PaperModelCatalog) -> pd.DataFrame:
    return _build_halo_level_table_from_survivors(paper_model.survivors)


def _unique_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([values.min() - 0.5, values.max() + 0.5], dtype=float)
    return edges


def _regular_logmass_bin_edges(values: np.ndarray, step_dex: float = 0.25) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
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


def _binned_quantiles(x: np.ndarray, y: np.ndarray, bins: np.ndarray, min_count: int = 1) -> pd.DataFrame:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    rows: List[dict] = []
    for left, right in zip(bins[:-1], bins[1:]):
        if right <= left:
            continue
        if right == bins[-1]:
            sel = (x >= left) & (x <= right)
        else:
            sel = (x >= left) & (x < right)
        if np.count_nonzero(sel) < min_count:
            continue
        ys = y[sel]
        q25, q50, q75 = np.quantile(ys, [0.25, 0.5, 0.75])
        rows.append(
            {
                "x": 0.5 * (left + right),
                "left": left,
                "right": right,
                "q25": float(q25),
                "median": float(q50),
                "q75": float(q75),
                "count": int(np.count_nonzero(sel)),
            }
        )
    return pd.DataFrame(rows, columns=["x", "left", "right", "q25", "median", "q75", "count"]).astype(
        {"x": float, "left": float, "right": float, "q25": float, "median": float, "q75": float, "count": int}
    )


def _binned_mean(x: np.ndarray, y: np.ndarray, bins: np.ndarray, min_count: int = 1) -> pd.DataFrame:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    rows: List[dict] = []
    for left, right in zip(bins[:-1], bins[1:]):
        if right <= left:
            continue
        if right == bins[-1]:
            sel = (x >= left) & (x <= right)
        else:
            sel = (x >= left) & (x < right)
        if np.count_nonzero(sel) < min_count:
            continue
        rows.append({"x": 0.5 * (left + right), "mean": float(np.mean(y[sel])), "count": int(np.count_nonzero(sel))})
    return pd.DataFrame(rows, columns=["x", "mean", "count"]).astype({"x": float, "mean": float, "count": int})


def _halo_system_quantiles(sample: pd.DataFrame, quantity: str) -> pd.DataFrame:
    rows: List[dict] = []
    for hid, grp in sample.groupby("hid_z0", sort=True):
        values = grp[quantity].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        rows.append(
            {
                "hid_z0": int(hid),
                "logMh_z0": float(grp["logMh_z0"].iloc[0]),
                "q25": float(q25),
                "median": float(q50),
                "q75": float(q75),
                "n_gc": int(len(values)),
            }
        )
    return pd.DataFrame(rows, columns=["hid_z0", "logMh_z0", "q25", "median", "q75", "n_gc"]).astype(
        {"hid_z0": int, "logMh_z0": float, "q25": float, "median": float, "q75": float, "n_gc": int}
    )


def _binned_median_halo_quantiles(halo_quantiles: pd.DataFrame, bins: np.ndarray, min_halos: int = 1) -> pd.DataFrame:
    rows: List[dict] = []
    if halo_quantiles.empty:
        return pd.DataFrame(rows, columns=["x", "q25", "median", "q75", "count"]).astype(
            {"x": float, "q25": float, "median": float, "q75": float, "count": int}
        )
    x = halo_quantiles["logMh_z0"].to_numpy(dtype=float)
    q25 = halo_quantiles["q25"].to_numpy(dtype=float)
    q50 = halo_quantiles["median"].to_numpy(dtype=float)
    q75 = halo_quantiles["q75"].to_numpy(dtype=float)
    for left, right in zip(bins[:-1], bins[1:]):
        if right <= left:
            continue
        if right == bins[-1]:
            select = (x >= left) & (x <= right)
        else:
            select = (x >= left) & (x < right)
        if int(np.count_nonzero(select)) < int(min_halos):
            continue
        rows.append(
            {
                "x": 0.5 * (left + right),
                "q25": float(np.median(q25[select])),
                "median": float(np.median(q50[select])),
                "q75": float(np.median(q75[select])),
                "count": int(np.count_nonzero(select)),
            }
        )
    return pd.DataFrame(rows, columns=["x", "q25", "median", "q75", "count"]).astype(
        {"x": float, "q25": float, "median": float, "q75": float, "count": int}
    )


def _mass_from_logmh(log_mh: np.ndarray | pd.Series | float) -> np.ndarray | float:
    out = np.power(10.0, np.asarray(log_mh, dtype=float))
    if np.isscalar(log_mh):
        return float(out)
    return out


def _choose_representative_vcs_systems(obs: ObsCatalog, model_halos: pd.DataFrame) -> pd.DataFrame:
    gc = obs.acsvcs_gc.loc[(obs.acsvcs_gc["pGC"] >= 0.5) & np.isfinite(obs.acsvcs_gc["feh"])].copy()
    counts = gc.groupby("VCC", sort=True).size().rename("n_gc").reset_index()
    systems = obs.vcs_systems.merge(counts, on="VCC", how="left").fillna({"n_gc": 0})
    logsm_lo = float(model_halos["logMstar_z0"].min()) - 0.15
    logsm_hi = float(model_halos["logMstar_z0"].max()) + 0.2
    candidates = systems.loc[(systems["logSM"] >= logsm_lo) & (systems["logSM"] <= logsm_hi) & (systems["n_gc"] >= 20)].copy()
    if len(candidates) < 4:
        candidates = systems.loc[systems["n_gc"] >= 20].copy()
    candidates = candidates.sort_values(["logSM", "n_gc"]).reset_index(drop=True)
    if len(candidates) < 4:
        raise RuntimeError("Not enough ACSVCS galaxies with >=20 probable GCs to build Figure 4")

    chosen_rows: List[pd.Series] = []
    taken: set[int] = set()
    quantiles = np.linspace(0.0, 1.0, 4)
    for quantile in quantiles:
        target = float(candidates["logSM"].quantile(quantile))
        order = np.argsort(np.abs(candidates["logSM"].to_numpy(dtype=float) - target))
        for idx in order:
            if int(idx) not in taken:
                taken.add(int(idx))
                chosen_rows.append(candidates.iloc[int(idx)])
                break
    return pd.DataFrame(chosen_rows).sort_values("logSM").reset_index(drop=True)


def _match_model_halos_to_observations(obs_examples: pd.DataFrame, halo_table: pd.DataFrame) -> List[int]:
    used: set[int] = set()
    halo_rows = halo_table.sort_values("logMstar_z0").reset_index(drop=True)
    matched: List[int] = []
    for logsm in obs_examples["logSM"].to_numpy(dtype=float):
        order = np.argsort(np.abs(halo_rows["logMstar_z0"].to_numpy(dtype=float) - logsm))
        pick = None
        for idx in order:
            hid = int(halo_rows.loc[int(idx), "hid_z0"])
            if hid not in used:
                pick = hid
                used.add(hid)
                break
        if pick is None:
            pick = int(halo_rows.loc[int(order[0]), "hid_z0"])
        matched.append(pick)
    return matched


def _build_obs_gc_population(obs: ObsCatalog) -> pd.DataFrame:
    gc = obs.acsvcs_gc.loc[(obs.acsvcs_gc["pGC"] >= 0.5) & np.isfinite(obs.acsvcs_gc["feh"])].copy()
    threshold, _, _ = fit_metallicity_split(gc["feh"].to_numpy(dtype=float))
    gc["population"] = _population_from_threshold(gc["feh"], threshold)
    gc["logM_gc_proxy"] = np.log10(np.clip(gc["m_gc_proxy_msun"].to_numpy(dtype=float), 1.0e-30, None))
    return gc


def build_figure_01(model: ModelCatalog, obs: ObsCatalog, paper_model: PaperModelCatalog | None = None) -> plt.Figure:
    halo_table = build_halo_level_table(model)
    bins = np.arange(11.0, 15.1, 0.2)
    summary = _binned_quantiles(halo_table["logMh_z0"], halo_table["mean_feh"], bins, min_count=1)
    formatter = mpl.ticker.LogFormatterMathtext(base=10.0)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 4.5))
    if not summary.empty:
        x_model = _mass_from_logmh(summary["x"])
        ax.fill_between(
            x_model,
            summary["q25"],
            summary["q75"],
            facecolor="tab:blue",
            edgecolor="none",
            linewidth=0.0,
            alpha=0.25,
            label="model IQR",
        )
        ax.plot(x_model, summary["median"], c="tab:blue", lw=1.8, label="model median")
    if paper_model is not None:
        paper_halo_table = build_paper_halo_level_table(paper_model)
        paper_summary = _binned_quantiles(paper_halo_table["logMh_z0"], paper_halo_table["mean_feh"], bins, min_count=1)
        if not paper_summary.empty:
            x_paper = _mass_from_logmh(paper_summary["x"])
            ax.fill_between(x_paper, paper_summary["q25"], paper_summary["q75"], facecolor="0.72", edgecolor="none", linewidth=0.0, alpha=0.18)
            ax.plot(x_paper, paper_summary["median"], c="0.35", ls="--", lw=1.4, label="Choksi+2018")
    ax.errorbar(
        obs.systems["M_halo_plot_msun"],
        obs.systems["mean_feh"],
        yerr=obs.systems["err_mean"],
        fmt="o",
        ms=3.0,
        c="black",
        ecolor="black",
        elinewidth=0.8,
        capsize=2.0,
        alpha=0.8,
        label="observations",
    )
    ax.set_xscale("log")
    ax.set_xticks([1.0e11, 1.0e12, 1.0e13, 1.0e14])
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel(r"$M_\mathrm{halo}~[M_\odot]$")
    ax.set_ylabel(r"Mean [Fe/H] of GCs")
    ax.set_xlim(1.0e11, 10.0**14.5)
    ax.set_ylim(-1.8, -0.3)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="best", ncol=1)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    return fig


def build_figure_02(model: ModelCatalog, obs: ObsCatalog, paper_model: PaperModelCatalog | None = None) -> plt.Figure:
    halo_table = build_halo_level_table(model)
    bins = np.arange(11.0, 15.1, 0.2)
    summary = _binned_quantiles(halo_table["logMh_z0"], halo_table["sigma_feh"], bins, min_count=1)
    formatter = mpl.ticker.LogFormatterMathtext(base=10.0)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 4.5))
    if not summary.empty:
        x_model = _mass_from_logmh(summary["x"])
        ax.fill_between(
            x_model,
            summary["q25"],
            summary["q75"],
            facecolor="tab:orange",
            edgecolor="none",
            linewidth=0.0,
            alpha=0.25,
            label="model IQR",
        )
        ax.plot(x_model, summary["median"], c="tab:orange", lw=1.8, label="model median")
    if paper_model is not None:
        paper_halo_table = build_paper_halo_level_table(paper_model)
        paper_summary = _binned_quantiles(paper_halo_table["logMh_z0"], paper_halo_table["sigma_feh"], bins, min_count=1)
        if not paper_summary.empty:
            x_paper = _mass_from_logmh(paper_summary["x"])
            ax.fill_between(x_paper, paper_summary["q25"], paper_summary["q75"], facecolor="0.72", edgecolor="none", linewidth=0.0, alpha=0.18)
            ax.plot(x_paper, paper_summary["median"], c="0.35", ls="--", lw=1.4, label="Choksi+2018")
    ax.errorbar(
        obs.systems["M_halo_plot_msun"],
        obs.systems["sigma_feh"],
        yerr=obs.systems["err_sigma"],
        fmt="o",
        ms=3.0,
        c="black",
        ecolor="black",
        elinewidth=0.8,
        capsize=2.0,
        alpha=0.8,
        label="observations",
    )
    ax.set_xscale("log")
    ax.set_xticks([1.0e11, 1.0e12, 1.0e13, 1.0e14])
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel(r"$M_\mathrm{halo}\,[M_{\odot}]$")
    ax.set_ylabel(r"$\sigma_{\rm [Fe/H]}$")
    ax.set_xlim(1.0e11, 10.0**14.5)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="best", ncol=1)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    return fig


def build_figure_03(model: ModelCatalog, obs: ObsCatalog, paper_model: PaperModelCatalog | None = None) -> plt.Figure:
    _ = obs
    halo_table = build_halo_level_table(model)
    #bins = np.arange(11.0, 15.1, 0.2)
    bins = np.arange(9.0, 15.1, 0.2)
    summary = _binned_quantiles(halo_table["logMh_z0"], halo_table["M_gc_final"], bins, min_count=1)
    #mh_line = np.logspace(11.0, 14.5, 200)
    mh_line = np.logspace(9.0, 14.5, 256)
    formatter = mpl.ticker.LogFormatterMathtext(base=10.0)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 4.5))
    if not summary.empty:
        x_model = _mass_from_logmh(summary["x"])
        ax.fill_between(
            x_model,
            summary["q25"],
            summary["q75"],
            facecolor="tab:green",
            edgecolor="none",
            linewidth=0.0,
            alpha=0.25,
            label="model IQR",
        )
        ax.plot(x_model, summary["median"], c="tab:green", lw=1.8, label="model median")
    if paper_model is not None:
        paper_halo_table = build_paper_halo_level_table(paper_model)
        paper_summary = _binned_quantiles(paper_halo_table["logMh_z0"], paper_halo_table["M_gc_final"], bins, min_count=1)
        if not paper_summary.empty:
            x_paper = _mass_from_logmh(paper_summary["x"])
            ax.fill_between(x_paper, paper_summary["q25"], paper_summary["q75"], facecolor="0.72", edgecolor="none", linewidth=0.0, alpha=0.18)
            ax.plot(x_paper, paper_summary["median"], c="0.35", ls="--", lw=1.4, label="Choksi+2018")
    ax.plot(mh_line, HARRIS_2015_RATIO * mh_line, c="black", ls="--", lw=1.2, label="Harris+2015")
    ax.set_xscale("log")
    ax.set_xticks([1.0e11, 1.0e12, 1.0e13, 1.0e14])
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel(r"$M_{h}\,[M_{\odot}]$")
    ax.set_ylabel(r"$M_{\rm GC}\,[M_{\odot}]$")
    #ax.set_xlim(1.0e11, 10.0**14.5)
    ax.set_xlim(1.0e9, 10.0**14.5)
    ax.set_yscale("log")
    #ax.set_ylim(3.0e5, 2.0e10)
    ax.set_ylim(3.0e3, 2.0e10)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="best", ncol=1)
    return fig


def build_figure_04(model: ModelCatalog, obs: ObsCatalog) -> plt.Figure:
    halo_table = build_halo_level_table(model)
    obs_examples = _choose_representative_vcs_systems(obs, halo_table)
    matched_halo_ids = _match_model_halos_to_observations(obs_examples, halo_table)
    gc = obs.acsvcs_gc.loc[(obs.acsvcs_gc["pGC"] >= 0.5) & np.isfinite(obs.acsvcs_gc["feh"])].copy()
    bins = np.linspace(FEH_MIN, FEH_MAX, 24)

    fig, axes = plt.subplots(2, 2, constrained_layout=True, dpi=STD_DPI, figsize=(9.2, 6.8), sharex=True, sharey=True)
    for ax, (_, obs_row), hid in zip(axes.flat, obs_examples.iterrows(), matched_halo_ids):
        obs_feh = gc.loc[gc["VCC"] == int(obs_row["VCC"]), "feh"].to_numpy(dtype=float)
        model_feh = model.survivors.loc[model.survivors["hid_z0"] == int(hid), "feh"].to_numpy(dtype=float)
        obs_feh = obs_feh[np.isfinite(obs_feh)]
        model_feh = model_feh[np.isfinite(model_feh)]
        ks_p = ks_2samp(model_feh, obs_feh).pvalue
        ax.hist(model_feh, bins=bins, density=True, histtype="stepfilled", color="tab:blue", alpha=0.35, label="model")
        ax.hist(obs_feh, bins=bins, density=True, histtype="step", color="black", lw=1.5, label="VCS")
        ax.text(
            0.03,
            0.95,
            f"VCC {int(obs_row['VCC'])}\n"
            + rf"$\log_{{10}}M_{{\ast}}={obs_row['logSM']:.2f}$"
            + "\n"
            + rf"$p_{{KS}}={ks_p:.2g}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        ax.grid(True, alpha=0.3, linestyle=":", which="both")
        ax.legend(frameon=False, loc="upper right", ncol=1)
    for ax in axes[-1]:
        ax.set_xlabel(r"[Fe/H]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized count")
    return fig


def build_figure_05(model: ModelCatalog, obs: ObsCatalog, paper_model: PaperModelCatalog | None = None) -> plt.Figure:
    peak_rows: List[dict] = []
    for hid, grp in model.survivors.groupby("hid_z0", sort=True):
        split, blue_peak, red_peak = fit_metallicity_split(grp["feh"].to_numpy(dtype=float))
        if not np.isfinite(blue_peak):
            continue
        peak_rows.append(
            {
                "hid_z0": int(hid),
                "logMh_z0": float(grp["logMh_z0"].iloc[0]),
                "blue_peak": blue_peak,
                "red_peak": red_peak,
                "split": split,
                "n_gc": int(len(grp)),
            }
        )
    peak_table = pd.DataFrame(
        peak_rows,
        columns=["hid_z0", "logMh_z0", "blue_peak", "red_peak", "split", "n_gc"],
    ).astype({"hid_z0": int, "logMh_z0": float, "blue_peak": float, "red_peak": float, "split": float, "n_gc": int})
    bins = np.arange(11.0, 15.1, 0.2)
    blue_summary = _binned_quantiles(peak_table["logMh_z0"], peak_table["blue_peak"], bins, min_count=1)
    red_summary = _binned_quantiles(peak_table["logMh_z0"], peak_table["red_peak"], bins, min_count=1)
    formatter = mpl.ticker.LogFormatterMathtext(base=10.0)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 4.5))
    if not blue_summary.empty:
        x_blue = _mass_from_logmh(blue_summary["x"])
        ax.fill_between(
            x_blue,
            blue_summary["q25"],
            blue_summary["q75"],
            facecolor="tab:blue",
            edgecolor="none",
            linewidth=0.0,
            alpha=0.25,
        )
        ax.plot(x_blue, blue_summary["median"], c="tab:blue", lw=1.8, label="model blue")
    if not red_summary.empty:
        x_red = _mass_from_logmh(red_summary["x"])
        ax.fill_between(
            x_red,
            red_summary["q25"],
            red_summary["q75"],
            facecolor="tab:red",
            edgecolor="none",
            linewidth=0.0,
            alpha=0.25,
        )
        ax.plot(x_red, red_summary["median"], c="tab:red", lw=1.8, label="model red")
    if paper_model is not None:
        paper_peak_rows: List[dict] = []
        for hid, grp in paper_model.survivors.groupby("hid_z0", sort=True):
            split, blue_peak, red_peak = fit_metallicity_split(grp["feh"].to_numpy(dtype=float))
            if not np.isfinite(blue_peak):
                continue
            paper_peak_rows.append(
                {
                    "hid_z0": int(hid),
                    "logMh_z0": float(grp["logMh_z0"].iloc[0]),
                    "blue_peak": blue_peak,
                    "red_peak": red_peak,
                    "split": split,
                    "n_gc": int(len(grp)),
                }
            )
        paper_peak_table = pd.DataFrame(
            paper_peak_rows,
            columns=["hid_z0", "logMh_z0", "blue_peak", "red_peak", "split", "n_gc"],
        ).astype({"hid_z0": int, "logMh_z0": float, "blue_peak": float, "red_peak": float, "split": float, "n_gc": int})
        if not paper_peak_table.empty:
            paper_blue_summary = _binned_quantiles(paper_peak_table["logMh_z0"], paper_peak_table["blue_peak"], bins, min_count=1)
            paper_red_summary = _binned_quantiles(paper_peak_table["logMh_z0"], paper_peak_table["red_peak"], bins, min_count=1)
            if not paper_blue_summary.empty:
                ax.plot(_mass_from_logmh(paper_blue_summary["x"]), paper_blue_summary["median"], c="tab:blue", ls="--", lw=1.4, label="Choksi+2018 blue")
            if not paper_red_summary.empty:
                ax.plot(_mass_from_logmh(paper_red_summary["x"]), paper_red_summary["median"], c="tab:red", ls="--", lw=1.4, label="Choksi+2018 red")

    obs_blue = obs.systems.loc[np.isfinite(obs.systems["blue_peak"])].copy()
    obs_red = obs.systems.loc[np.isfinite(obs.systems["red_peak"])].copy()
    ax.scatter(obs_blue["M_halo_plot_msun"], obs_blue["blue_peak"], s=18.0, c="tab:blue", alpha=0.75, label="obs blue")
    ax.scatter(obs_red["M_halo_plot_msun"], obs_red["red_peak"], s=18.0, c="tab:red", alpha=0.75, label="obs red")

    ax.set_xscale("log")
    ax.set_xticks([1.0e11, 1.0e12, 1.0e13, 1.0e14])
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel(r"$M_{h}\,[M_{\odot}]$")
    ax.set_ylabel(r"Peak [Fe/H]")
    ax.set_xlim(1.0e11, 10.0**14.5)
    ax.set_ylim(-2.3, 0.2)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="best", ncol=1)
    return fig


def build_figure_06(model: ModelCatalog, paper_model: PaperModelCatalog | None = None) -> plt.Figure:
    halo_table = build_halo_level_table(model)
    bins = _regular_logmass_bin_edges(halo_table["logMh_z0"].to_numpy(dtype=float), 0.25)

    fig, axes = plt.subplots(2, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 6.8), sharex=True)
    for population, colour in [("blue", "tab:blue"), ("red", "tab:red")]:
        sample = model.survivors.loc[model.survivors["population"] == population].copy()
        time_summary = _binned_median_halo_quantiles(_halo_system_quantiles(sample, "t_form_gyr"), bins, min_halos=1)
        form_halo_summary = _binned_median_halo_quantiles(_halo_system_quantiles(sample, "logMh_form"), bins, min_halos=1)

        if not time_summary.empty:
            axes[0].fill_between(
                _mass_from_logmh(time_summary["x"]),
                time_summary["q25"],
                time_summary["q75"],
                facecolor=colour,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.2,
            )
            axes[0].plot(_mass_from_logmh(time_summary["x"]), time_summary["median"], c=colour, lw=1.8, label=population)
        if not form_halo_summary.empty:
            axes[1].fill_between(
                _mass_from_logmh(form_halo_summary["x"]),
                form_halo_summary["q25"],
                form_halo_summary["q75"],
                facecolor=colour,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.2,
            )
            axes[1].plot(_mass_from_logmh(form_halo_summary["x"]), form_halo_summary["median"], c=colour, lw=1.8, label=population)
        if paper_model is not None:
            paper_sample = paper_model.survivors.loc[paper_model.survivors["population"] == population].copy()
            paper_time_summary = _binned_median_halo_quantiles(_halo_system_quantiles(paper_sample, "t_form_gyr"), bins, min_halos=1)
            paper_form_halo_summary = _binned_median_halo_quantiles(_halo_system_quantiles(paper_sample, "logMh_form"), bins, min_halos=1)
            if not paper_time_summary.empty:
                axes[0].plot(_mass_from_logmh(paper_time_summary["x"]), paper_time_summary["median"], c=colour, ls="--", lw=1.4, label=f"Choksi+2018 {population}")
            if not paper_form_halo_summary.empty:
                axes[1].plot(_mass_from_logmh(paper_form_halo_summary["x"]), paper_form_halo_summary["median"], c=colour, ls="--", lw=1.4, label=f"Choksi+2018 {population}")

    time_summary_all = _binned_median_halo_quantiles(_halo_system_quantiles(model.survivors, "t_form_gyr"), bins, min_halos=1)
    form_halo_summary_all = _binned_median_halo_quantiles(_halo_system_quantiles(model.survivors, "logMh_form"), bins, min_halos=1)
    if not time_summary_all.empty:
        axes[0].plot(_mass_from_logmh(time_summary_all["x"]), time_summary_all["median"], c="black", lw=1.8, label="all")
    if not form_halo_summary_all.empty:
        axes[1].plot(_mass_from_logmh(form_halo_summary_all["x"]), form_halo_summary_all["median"], c="black", lw=1.8, label="all")
    if paper_model is not None:
        paper_time_summary_all = _binned_median_halo_quantiles(_halo_system_quantiles(paper_model.survivors, "t_form_gyr"), bins, min_halos=1)
        paper_form_halo_summary_all = _binned_median_halo_quantiles(_halo_system_quantiles(paper_model.survivors, "logMh_form"), bins, min_halos=1)
        if not paper_time_summary_all.empty:
            axes[0].plot(_mass_from_logmh(paper_time_summary_all["x"]), paper_time_summary_all["median"], c="black", ls="--", lw=1.4, label="Choksi+2018 all")
        if not paper_form_halo_summary_all.empty:
            axes[1].plot(_mass_from_logmh(paper_form_halo_summary_all["x"]), paper_form_halo_summary_all["median"], c="black", ls="--", lw=1.4, label="Choksi+2018 all")

    axes[0].set_ylabel(r"$t_{\rm form}$ [Gyr]")
    axes[1].set_ylabel(r"$\log_{10}(M_{h,{\rm form}}/M_{\odot})$")
    axes[1].set_xlabel(r"$M_{h}(z=0)\,[M_{\odot}]$")
    x_min = 10.0 ** (float(halo_table["logMh_z0"].min()) - 0.05)
    x_max = 10.0 ** (float(halo_table["logMh_z0"].max()) + 0.05)
    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
        ax.grid(True, alpha=0.3, linestyle=":", which="both")
        ax.legend(frameon=False, loc="best", ncol=1)
    axes[0].set_xlim(1.0e11, 10.0**14.5)
    return fig


def build_figure_07(model: ModelCatalog, obs: ObsCatalog, paper_model: PaperModelCatalog | None = None) -> plt.Figure:
    halo_table = build_halo_level_table(model)
    halo_edges = _unique_bin_edges(halo_table["logMh_z0"].to_numpy(dtype=float), 3)
    halo_labels = ["low", "mid", "high"]
    halo_bin_map: Dict[int, str] = {}
    for _, row in halo_table.iterrows():
        for idx, (left, right) in enumerate(zip(halo_edges[:-1], halo_edges[1:])):
            include = row["logMh_z0"] <= right if idx == len(halo_edges) - 2 else row["logMh_z0"] < right
            if row["logMh_z0"] >= left and include:
                halo_bin_map[int(row["hid_z0"])] = halo_labels[min(idx, len(halo_labels) - 1)]
                break
    model_sample = model.survivors.copy()
    model_sample["halo_mass_bin"] = model_sample["hid_z0"].map(halo_bin_map)

    obs_gc = _build_obs_gc_population(obs)
    mass_bins = np.arange(5.0, 7.41, 0.2)
    bin_styles = {"low": ":", "mid": "--", "high": "-."}

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.8, 4.8))
    for population, colour in [("blue", "tab:blue"), ("red", "tab:red")]:
        subset = model_sample.loc[model_sample["population"] == population].copy()
        summary = _binned_quantiles(subset["logM_final"], subset["feh"], mass_bins, min_count=100)
        if not summary.empty:
            ax.fill_between(
                _mass_from_logmh(summary["x"]),
                summary["q25"],
                summary["q75"],
                facecolor=colour,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.25,
                label=f"{population} model IQR",
            )
        for halo_bin, linestyle in bin_styles.items():
            track = _binned_mean(
                subset.loc[subset["halo_mass_bin"] == halo_bin, "logM_final"],
                subset.loc[subset["halo_mass_bin"] == halo_bin, "feh"],
                mass_bins,
                min_count=50,
            )
            if not track.empty:
                label = f"{population} {halo_bin} $M_h$"
                ax.plot(_mass_from_logmh(track["x"]), track["mean"], c=colour, ls=linestyle, lw=1.3, alpha=0.95, label=label)
        obs_track = _binned_mean(
            obs_gc.loc[obs_gc["population"] == population, "logM_gc_proxy"],
            obs_gc.loc[obs_gc["population"] == population, "feh"],
            mass_bins,
            min_count=50,
        )
        if not obs_track.empty:
            ax.plot(_mass_from_logmh(obs_track["x"]), obs_track["mean"], c=colour, lw=2.0, label=f"obs {population}")
        if paper_model is not None:
            paper_subset = paper_model.survivors.loc[paper_model.survivors["population"] == population].copy()
            paper_summary = _binned_quantiles(paper_subset["logM_final"], paper_subset["feh"], mass_bins, min_count=50)
            if not paper_summary.empty:
                ax.plot(_mass_from_logmh(paper_summary["x"]), paper_summary["median"], c=colour, ls="--", lw=1.6, label=f"Choksi+2018 {population}")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    ax.set_xlabel(r"$M_{\rm GC}\,[M_{\odot}]$")
    ax.set_ylabel(r"[Fe/H]")
    ax.set_xlim(1.0e5, 7.0e6)
    ax.set_ylim(-2.0, 0.0)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="best", ncol=2, fontsize=8)
    return fig


def build_figure_08(model: ModelCatalog, paper_model: PaperModelCatalog | None = None) -> plt.Figure:
    sample = model.survivors.copy()
    halo_bins = np.arange(11.0, 15.01, 0.25)
    mass_bin_specs = [
        (5.0, 5.5, "-", r"$5.0 < \log_{10}(M/M_{\odot}) < 5.5$"),
        (5.5, 6.0, "--", r"$5.5 < \log_{10}(M/M_{\odot}) < 6.0$"),
        (6.0, 6.5, "-.", r"$6.0 < \log_{10}(M/M_{\odot}) < 6.5$"),
        (6.5, 7.5, ":", r"$6.5 < \log_{10}(M/M_{\odot})$"),
    ]
    quantity_info = [
        ("logMh_form", r"$M_{h}(t_{\rm form})\,[M_{\odot}]$"),
        ("logMstar_form", r"$M_{\ast}(t_{\rm form})\,[M_{\odot}]$"),
        ("logMgas_form", r"$M_{g}(t_{\rm form})\,[M_{\odot}]$"),
    ]

    fig, axes = plt.subplots(3, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 8.3), sharex=True)
    for ax, (quantity, ylabel) in zip(axes, quantity_info):
        for population, colour in [("blue", "tab:blue"), ("red", "tab:red")]:
            subset = sample.loc[sample["population"] == population].copy()
            for low, high, linestyle, _ in mass_bin_specs:
                mass_sel = (subset["logM_final"] >= low) & (subset["logM_final"] < high)
                summary = _binned_quantiles(subset.loc[mass_sel, "logMh_z0"], subset.loc[mass_sel, quantity], halo_bins, min_count=50)
                if summary.empty:
                    continue
                ax.plot(_mass_from_logmh(summary["x"]), _mass_from_logmh(summary["median"]), c=colour, ls=linestyle, lw=1.5)
                if paper_model is not None:
                    paper_subset = paper_model.survivors.loc[paper_model.survivors["population"] == population].copy()
                    paper_mass_sel = (paper_subset["logM_final"] >= low) & (paper_subset["logM_final"] < high)
                    paper_summary = _binned_quantiles(paper_subset.loc[paper_mass_sel, "logMh_z0"], paper_subset.loc[paper_mass_sel, quantity], halo_bins, min_count=50)
                    if not paper_summary.empty:
                        ax.plot(_mass_from_logmh(paper_summary["x"]), _mass_from_logmh(paper_summary["median"]), c=colour, ls=linestyle, lw=1.0, alpha=0.45)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle=":", which="both")

    mass_handles = [mpl.lines.Line2D([], [], c="black", ls=linestyle, lw=1.5, label=label) for _, _, linestyle, label in mass_bin_specs]
    colour_handles = [
        mpl.lines.Line2D([], [], c="tab:blue", ls="-", lw=1.8, label="blue"),
        mpl.lines.Line2D([], [], c="tab:red", ls="-", lw=1.8, label="red"),
    ]
    mass_legend = axes[0].legend(handles=mass_handles, frameon=False, loc="upper left", ncol=1, fontsize=8)
    axes[0].add_artist(mass_legend)
    axes[0].legend(handles=colour_handles, frameon=False, loc="lower right", ncol=1, fontsize=8)
    axes[-1].set_xlabel(r"$M_{h}(z=0)\,[M_{\odot}]$")
    axes[-1].xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    axes[-1].set_xlim(1.0e11, 10.0**14.5)
    return fig


def _density_threshold_for_enclosed_fraction(density: np.ndarray, enclosed_fraction: float) -> float:
    flat = np.sort(np.asarray(density, dtype=float).ravel())[::-1]
    flat = flat[flat > 0.0]
    if len(flat) == 0:
        return 0.0
    cdf = np.cumsum(flat) / np.sum(flat)
    idx = min(np.searchsorted(cdf, enclosed_fraction), len(flat) - 1)
    return float(flat[idx])


def _build_mpb_track(model: ModelCatalog, hid_z0: int) -> pd.DataFrame:
    track = model.mpb.loc[model.mpb["subhalo_id_z0"] == int(hid_z0)].copy()
    if track.empty:
        return track
    idx = track.groupby("SnapNum")["logMh_msun_h"].idxmax()
    track = track.loc[idx].sort_values("Redshift", ascending=False).reset_index(drop=True)
    track["M_halo"] = np.power(10.0, track["logMh_msun_h"].to_numpy(dtype=float))
    track["M_star"] = stellar_mass_from_halo_mass(track["M_halo"].to_numpy(dtype=float), track["Redshift"].to_numpy(dtype=float))
    track["logM_star"] = np.log10(np.clip(track["M_star"].to_numpy(dtype=float), 1.0e-30, None))
    track["feh"] = metallicity_mmr(track["logM_star"].to_numpy(dtype=float), track["Redshift"].to_numpy(dtype=float))
    return track


def _build_cluster_fraction_table(model: ModelCatalog, feh_edges: np.ndarray) -> pd.DataFrame:
    rows: List[dict] = []
    for hid, grp in model.catalog.groupby("hid_z0", sort=True):
        track = _build_mpb_track(model, int(hid))
        if track.empty:
            continue

        interp_df = pd.DataFrame(
            {
                "feh": track["feh"].to_numpy(dtype=float),
                "mstar": track["M_star"].to_numpy(dtype=float),
            }
        )
        interp_df = interp_df.loc[np.isfinite(interp_df["feh"]) & np.isfinite(interp_df["mstar"]) & (interp_df["mstar"] > 0.0)].copy()
        if interp_df.empty:
            continue
        interp_df = interp_df.sort_values("feh").reset_index(drop=True)
        interp_df["mstar"] = np.maximum.accumulate(interp_df["mstar"].to_numpy(dtype=float))
        interp_df["feh_round"] = interp_df["feh"].round(6)
        interp_df = (
            interp_df.groupby("feh_round", as_index=False)
            .agg(feh=("feh", "mean"), mstar=("mstar", "max"))
            .sort_values("feh")
            .reset_index(drop=True)
        )

        mstar_edges = np.interp(
            feh_edges,
            interp_df["feh"].to_numpy(dtype=float),
            interp_df["mstar"].to_numpy(dtype=float),
            left=0.0,
            right=float(interp_df["mstar"].iloc[-1]),
        )
        mstar_bins = np.diff(mstar_edges)

        survivors = model.survivors.loc[model.survivors["hid_z0"] == int(hid)].copy()
        survivor_feh = survivors["feh"].to_numpy(dtype=float)
        survivor_mass = survivors["m_final_msun"].to_numpy(dtype=float)
        cluster_bins = np.zeros(len(feh_edges) - 1, dtype=float)
        for i, (left, right) in enumerate(zip(feh_edges[:-1], feh_edges[1:])):
            if i == len(feh_edges) - 2:
                select = (survivor_feh >= left) & (survivor_feh <= right)
            else:
                select = (survivor_feh >= left) & (survivor_feh < right)
            cluster_bins[i] = float(np.sum(survivor_mass[select]))

        field_bins = mstar_bins - cluster_bins
        ratio = np.full(len(field_bins), np.nan, dtype=float)
        valid_ratio = field_bins > 0.0
        ratio[valid_ratio] = cluster_bins[valid_ratio] / field_bins[valid_ratio]
        for i, (left, right) in enumerate(zip(feh_edges[:-1], feh_edges[1:])):
            rows.append(
                {
                    "hid_z0": int(hid),
                    "logMh_z0": float(grp["logMh_z0"].iloc[0]),
                    "feh_left": float(left),
                    "feh_right": float(right),
                    "feh_center": float(0.5 * (left + right)),
                    "m_gc_bin_msun": float(cluster_bins[i]),
                    "m_field_bin_msun": float(field_bins[i]),
                    "ratio_defined": bool(valid_ratio[i]),
                    "ratio": float(ratio[i]) if np.isfinite(ratio[i]) else np.nan,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "hid_z0",
            "logMh_z0",
            "feh_left",
            "feh_right",
            "feh_center",
            "m_gc_bin_msun",
            "m_field_bin_msun",
            "ratio_defined",
            "ratio",
        ],
    ).astype(
        {
            "hid_z0": int,
            "logMh_z0": float,
            "feh_left": float,
            "feh_right": float,
            "feh_center": float,
            "m_gc_bin_msun": float,
            "m_field_bin_msun": float,
            "ratio_defined": bool,
            "ratio": float,
        }
    )


def build_figure_09(model: ModelCatalog, obs: ObsCatalog, final_redshift: float, paper_model: PaperModelCatalog | None = None) -> plt.Figure:
    model_feh = model.formed["feh"].to_numpy(dtype=float)
    model_age = cosmic_time_gyr(float(final_redshift)) - cosmic_time_gyr(model.formed["zform"].to_numpy(dtype=float))
    age_range = [6.0, 14.0]
    feh_range = [FEH_MIN - 0.3, 0.5]
    density, xedges, yedges = np.histogram2d(model_feh, model_age, bins=[64, 64], range=[feh_range, age_range])
    density = gaussian_filter(density, sigma=1.1)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcenters, ycenters, indexing="ij")

    contour_specs = [(0.97, 97), (0.75, 75), (0.50, 50), (0.10, 10)]
    contour_levels = [
        (_density_threshold_for_enclosed_fraction(density, enclosed_fraction), label)
        for enclosed_fraction, label in contour_specs
    ]
    contour_levels = [(level, label) for level, label in contour_levels if level > 0.0]
    contour_levels.sort(key=lambda item: item[0])
    levels = [item[0] for item in contour_levels]
    level_labels = {level: label for level, label in contour_levels}

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 4.9))
    if levels:
        fill_colours = ["#f2f2f2", "#d9d9d9", "#bfbfbf", "#a6a6a6"][-len(levels) :]
        ax.contourf(X, Y, density, levels=levels + [float(np.max(density))], colors=fill_colours)
        contour = ax.contour(X, Y, density, levels=levels, colors="black", linewidths=1.2)
        ax.clabel(contour, fmt=level_labels, inline=True, fontsize=9)
    if paper_model is not None:
        paper_feh = paper_model.survivors["feh"].to_numpy(dtype=float)
        paper_age = paper_model.survivors["cluster_age_gyr"].to_numpy(dtype=float)
        paper_density, _, _ = np.histogram2d(paper_feh, paper_age, bins=[64, 64], range=[feh_range, age_range])
        paper_density = gaussian_filter(paper_density, sigma=1.1)
        paper_levels = []
        for enclosed_fraction, _label in contour_specs:
            level = _density_threshold_for_enclosed_fraction(paper_density, enclosed_fraction)
            if level > 0.0:
                paper_levels.append(level)
        paper_levels = sorted(set(paper_levels))
        if paper_levels:
            ax.contour(X, Y, paper_density, levels=paper_levels, colors="0.35", linewidths=1.1, linestyles="--")
            ax.plot([], [], c="0.35", ls="--", lw=1.1, label="Choksi+2018")

    ax.errorbar(
        obs.mw_age_metallicity["feh"],
        obs.mw_age_metallicity["age_gyr"],
        yerr=obs.mw_age_metallicity["age_err_gyr"],
        fmt="o",
        ms=5.2,
        c="#4c63ff",
        ecolor="#4c63ff",
        elinewidth=1.0,
        capsize=0.0,
        alpha=0.75,
        label="Galactic",
    )
    ax.errorbar(
        obs.lmc_age_metallicity["feh"],
        obs.lmc_age_metallicity["age_gyr"],
        yerr=[
            obs.lmc_age_metallicity["age_err_lo_gyr"].to_numpy(dtype=float),
            obs.lmc_age_metallicity["age_err_hi_gyr"].to_numpy(dtype=float),
        ],
        fmt="o",
        ms=5.5,
        c="#3ca44a",
        ecolor="#3ca44a",
        elinewidth=1.0,
        capsize=0.0,
        alpha=0.75,
        label="LMC",
    )
    ax.set_xlabel(r"[Fe/H]")
    ax.set_ylabel("Age [Gyr]")
    ax.set_xlim(feh_range)
    ax.set_ylim(age_range)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="lower left", ncol=1)
    return fig


def build_figure_10(model: ModelCatalog, obs: ObsCatalog) -> plt.Figure:
    _ = obs
    feh_edges = np.arange(-2.0, 0.01, 0.2)
    ratio_table = _build_cluster_fraction_table(model, feh_edges)
    ratio_table = ratio_table.loc[np.isfinite(ratio_table["ratio"]) & (ratio_table["ratio"] > 0.0)].copy()

    summary = (
        ratio_table.groupby("feh_center", sort=True)["ratio"]
        .agg(
            q25=lambda values: float(np.quantile(values.to_numpy(dtype=float), 0.25)),
            median=lambda values: float(np.quantile(values.to_numpy(dtype=float), 0.50)),
            q75=lambda values: float(np.quantile(values.to_numpy(dtype=float), 0.75)),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=STD_DPI, figsize=(6.4, 5.2))
    ax.fill_between(
        summary["feh_center"],
        summary["q25"],
        summary["q75"],
        facecolor="#7b7ce6",
        edgecolor="none",
        linewidth=0.0,
        alpha=0.55,
    )

    mass_bin_styles = [
        (11.5, 12.5, (0.0, (1.0, 2.2)), r"$11.5 < \log_{10}(M_h/M_{\odot}) < 12.5$"),
        (12.5, 13.5, (0.0, (4.0, 4.0)), r"$12.5 < \log_{10}(M_h/M_{\odot}) < 13.5$"),
        (13.5, 14.5, (0.0, (3.0, 3.0, 1.0, 3.0)), r"$13.5 < \log_{10}(M_h/M_{\odot}) < 14.5$"),
    ]
    for low, high, linestyle, label in mass_bin_styles:
        subset = ratio_table.loc[(ratio_table["logMh_z0"] > low) & (ratio_table["logMh_z0"] < high)].copy()
        if subset.empty:
            continue
        trend = (
            subset.groupby("feh_center", sort=True)["ratio"]
            .agg(lambda values: float(np.quantile(values.to_numpy(dtype=float), 0.50)))
            .reset_index(name="median")
        )
        ax.plot(trend["feh_center"], trend["median"], c="black", lw=1.4, linestyle=linestyle, label=label)

    ax.set_xlabel(r"[Fe/H]")
    ax.set_ylabel(r"$M_{\rm GC}/M_{\rm field}$")
    ax.set_xlim(-2.0, 0.0)
    ax.set_yscale("log")
    ax.set_ylim(1.0e-4, 1.0)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    ax.legend(frameon=False, loc="lower left", ncol=1)
    return fig


def _parse_figures(value: str | None) -> List[int]:
    if value is None or value.strip() == "":
        return sorted(FIGURE_STEMS)
    figures: List[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        fig = int(chunk)
        if fig not in FIGURE_STEMS:
            raise ValueError(f"Unsupported figure number: {fig}")
        figures.append(fig)
    return sorted(set(figures))


parser = argparse.ArgumentParser(description="Reproduce the Choksi+2018 figure suite from one local High-z SMBHs output directory.")
parser.add_argument(
    "--out_dir",
    type=Path,
    default=DEFAULT_OUT_DIR,
    help="Model output directory containing the root allcat file, mpb_from_fixed_trees.csv, ns*/, and run_metadata.json.",
)
parser.add_argument(
    "--plot_dir",
    type=Path,
    default=None,
    help="Plot output directory. Defaults to <out_dir>/_plots_Choksi+2018.",
)
parser.add_argument("--ns-value", type=float, default=NS_VALUE_DEFAULT, help="N_s value to plot.")
parser.add_argument("--figures", type=str, default=None, help="Optional comma-separated subset, e.g. 1,2,5.")
parser.add_argument("--final-z", type=float, default=None, help="Optional final redshift override. Defaults to run_metadata.json when present.")
args = parser.parse_args()

out_dir = args.out_dir.resolve()
allcat_template, mpb_path = _resolve_model_inputs_from_out_dir(out_dir)
model_root = out_dir
plot_dir = args.plot_dir.resolve() if args.plot_dir is not None else (out_dir / "_plots_Choksi+2018").resolve()
plot_dir.mkdir(parents=True, exist_ok=True)

run_metadata = _load_run_metadata(allcat_template)
final_redshift = float(args.final_z) if args.final_z is not None else float(run_metadata.get("final_redshift", 0.0))
_ = final_redshift

_apply_plot_style()
observations = load_observations()
model_catalog = build_model_catalog(allcat_template, mpb_path, args.ns_value)
paper_model = load_choksi_paper_model()
selected_figures = _parse_figures(args.figures)

figure_builders = {
    1: lambda: build_figure_01(model_catalog, observations, paper_model),
    2: lambda: build_figure_02(model_catalog, observations, paper_model),
    3: lambda: build_figure_03(model_catalog, observations, paper_model),
    4: lambda: build_figure_04(model_catalog, observations),
    5: lambda: build_figure_05(model_catalog, observations, paper_model),
    6: lambda: build_figure_06(model_catalog, paper_model),
    7: lambda: build_figure_07(model_catalog, observations, paper_model),
    8: lambda: build_figure_08(model_catalog, paper_model),
    9: lambda: build_figure_09(model_catalog, observations, final_redshift, paper_model),
    10: lambda: build_figure_10(model_catalog, observations),
}

for fig_num in selected_figures:
    fig = figure_builders[fig_num]()
    path = plot_dir / f"Fig.{fig_num:02d}_{FIGURE_STEMS[fig_num]}.png"
    fig.savefig(path, dpi=STD_DPI, bbox_inches="tight")
    plt.close(fig)
