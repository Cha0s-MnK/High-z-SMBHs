#!/usr/bin/env python3

"""Standalone analytical Gao+2023 GC evolution.

This variant keeps the legacy event-driven scheduler and RK4 orbital decay.
The background density is evaluated analytically inside the RK4 substeps, so
there is no lookup-table evolution mode.

- deposited-mass summaries are maintained incrementally instead of recomputing
  ``np.sum(depo, axis=...)`` inside the inner scheduler loops.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import special


PI = math.pi
kpc = 3.0856775814913673e19 # [m]
ALEX_VC_CONV = 65.7677131526
t_universe = 13.780 # DESI 2024 + BAO [Gyr]
Omega_m0 = 0.307 # DESI 2024 + BAO
H_0 = 67.97 # Hubble constant (DESI 2024 + BAO) [(km/s)/Mpc]
Omega_Lambda0 = 1.0 - Omega_m0 # DESI 2024 + BAO
MPC_IN_KM = 3.0856775814913673e19
Gyr = 365.25 * 24.0 * 3600.0 * 1.0e9 # [s]
LCDM_H0_GYR_INV = (H_0 / MPC_IN_KM) * Gyr
LCDM_TIME_PREFAC_GYR = 2.0 / (3.0 * LCDM_H0_GYR_INV * math.sqrt(Omega_Lambda0))
LCDM_ASINH_RATIO = math.sqrt(Omega_Lambda0 / Omega_m0)
EPS = 1.0e-30
STATUS_ALIVE = 1
STATUS_EXHAUSTED = -1
STATUS_TORN = -2
STATUS_SUNK = -3
STATUS_WANDERER = -4
STATUS_WANDERER_SUNK = -5
TIDAL_STRIPPING_FRAGIONE2019 = "Fragione+2019"
TIDAL_STRIPPING_CHOKSI2018 = "Choksi+2018"
TIDAL_STRIPPING_CHOICES = (TIDAL_STRIPPING_FRAGIONE2019, TIDAL_STRIPPING_CHOKSI2018)
CHOKSI_TIDAL_P = 0.5
CHOKSI_TISO_COEFF_GYR = 17.0

FINAL_GC_HEADER = "\n".join([
    "gc_index status m_final_msun m_init_msun lookback_time_final_gyr lookback_time_init_gyr r_final_kpc r_init_kpc",
    ("rows: one GC per input GCini row; lookback times are measured from the configured final redshift; "
     "status = 1 alive, -1 exhausted, -2 torn, -3 sunk_to_center, -4 IMBH_wanderer, -5 sunk_wanderer"),])

DEPOS_HEADER = "\n".join([
    "lookback_time_gyr bin_index r_inner_kpc r_outer_kpc m_depo_total_msun m_star_no_evo_msun m_star_with_evo_msun",
    "rows: one radial bin per saved coarse time block; lookback times are measured from the configured final redshift; the same lookback_time is repeated for all bins in a block",])

TRACE_HEADER = "\n".join([
    "trace_index phase gc_index status t_cosmic_gyr redshift r_kpc m_gc_msun bin_index",
    "rows: one orbit-trace sample from the existing event-driven evolution; redshift follows the solver's own time-redshift convention",])


@dataclass
class Tunables:
    # Maximum per-GC timestep in Gyr after all adaptive limits are applied.
    dt_max: float = 0.1
    # Number of coarse time blocks between the Big Bang and the chosen final epoch.
    t_div: int = 100
    # Number of logarithmic radial bins used for deposited-mass bookkeeping.
    binnub: int = 100
    # Minimum base timescale in Gyr allowed for the ts_m and ts_r timestep floors.
    t_limit: float = 1.0e-2
    # Radius in kpc below which a cluster is tagged as sunk to the center.
    r_sink: float = 1.0e-3
    # Little-h used in the halo virial-radius and spin conversions.
    h: float = 0.704
    # Inner radius floor in kpc for radial binning and background-density calls.
    r_min: float = 1.0e-3


def _numeric_rows(path: Path) -> np.ndarray:
    """Read whitespace-delimited numeric rows, ignoring comments and blanks."""

    rows: List[List[float]] = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                vals = [float(v) for v in s.split()]
            except ValueError:
                continue
            if len(vals) > 0:
                rows.append(vals)
    if not rows:
        return np.zeros((0, 0), dtype=float)
    ncol = max(len(r) for r in rows)
    out = np.zeros((len(rows), ncol), dtype=float)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def _read_haloevo_mpb(path: Path) -> np.ndarray:
    """Read the monotonic MPB-like block from one halo-evolution table."""

    rows: List[List[float]] = []
    last_val: Optional[float] = None
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                vals = [float(v) for v in parts]
            except ValueError:
                continue
            if len(vals) < 9:
                continue
            cur_check = vals[0]
            # Fixed tree files place extra side branches after the MPB block.
            # The MPB itself is monotonic in retained halo mass, so the first
            # drop marks the hand-off to non-MPB rows.
            if last_val is not None and cur_check < last_val:
                break
            rows.append(vals[:9])
            last_val = cur_check
    if not rows:
        return np.zeros((0, 9), dtype=float)
    return np.asarray(rows, dtype=float)


def read_haloevo_mpb(path: Path) -> np.ndarray:
    """Public wrapper for the MPB parser used by plotting-ready summaries."""

    return _read_haloevo_mpb(path)


def f_x_SMHM(x: float, z: float) -> float:
    a = 1.0 / (1.0 + z)
    nu = math.exp(- 4.0 * a * a)
    alpha = - 1.412 + 0.731 * (a - 1.0) * nu
    delta = 3.508 + (2.608 * (a - 1.0) - 0.043 * z) * nu
    gamma = 0.316 + (1.319 * (a - 1.0) + 0.279 * z) * nu
    return - math.log10(10.0 ** (alpha * x) + 1.0) + delta * (math.log10(1.0 + math.exp(x))) ** gamma / (1.0 + math.exp(0.1**x))

def Omega_m(z: float) -> float:
    """flat LambdaCDM matter density parameter"""

    z_use = max(float(z), 0.0)
    zp1_cubed = (1.0 + z_use) ** 3
    return Omega_m0 * zp1_cubed / (1.0 - Omega_m0 + Omega_m0 * zp1_cubed)

def CosmicTimeGyr2Redshift(t_gyr: float) -> float:
    """flat LambdaCDM cosmic age to redshift conversion without radiation"""

    t = max(float(t_gyr), 1.0e-9)
    sinh_val = math.sinh(t / LCDM_TIME_PREFAC_GYR)
    if sinh_val <= 0.0:
        return 0.0
    return max((LCDM_ASINH_RATIO / sinh_val) ** (2.0 / 3.0) - 1.0, 0.0)

def rvir_kpc(Mhalo_1e9Msun: float, t_l_gyr: float, tun: Tunables) -> float:
    t = max(float(t_l_gyr), 1.0e-9)
    z = CosmicTimeGyr2Redshift(t)
    Omega_m_z = Omega_m(z)
    return 163.0 / tun.h * (Mhalo_1e9Msun / 1.0e3 * tun.h) ** (1.0 / 3.0) / (
        (Omega_m0 * (18.0 * PI * PI + 82.0 * (Omega_m_z - 1.0) - 39.0 * (Omega_m_z - 1.0) ** 2) / Omega_m_z / 200.0) ** (1.0 / 3.0) * (1.0 + z))


def Mstar_1e9Msun_SMHM(Mhalo_1e9Msun: float, t_l_gyr: float) -> float:
    t = max(float(t_l_gyr), 1.0e-9)
    z = CosmicTimeGyr2Redshift(t)
    a = 1.0 / (1.0 + z)
    nu = math.exp(- 4.0 * a * a)
    epsilon = 10.0 ** (- 1.777 - 0.006 * (a - 1.0) * nu - 0.119 * (a - 1.0))
    M1 = 10.0 ** (11.514 - (1.793 * (a - 1.0) + 0.251 * z) * nu)
    return epsilon * M1 * 10.0 ** (f_x_SMHM(math.log10(Mhalo_1e9Msun * 1.0e9 / M1), z) - f_x_SMHM(0.0, z)) / 1.0e9


def _sersic_shape_coeffs(sersic_n: float) -> Tuple[float, float]:
    n = max(float(sersic_n), 1.0e-6)
    sersic_p = 1.0 - 0.6097 / n + 0.05563 / (n * n)
    sersic_b = 2.0 * n - 1.0 / 3.0 + 0.009876 / n
    return sersic_p, sersic_b


def rho_bg_sw1(
    r_kpc: float,
    spin_norm: float,
    Mv_1e9Msun: float,
    t_l_gyr: float,
    sersic_n: float,
    tun: Tunables,
) -> float:
    r = max(r_kpc, 1.0e-9)
    sersic_p, sersic_b = _sersic_shape_coeffs(2.2)

    c = 9.354 / ((max(Mv_1e9Msun, 1.0e-30) * tun.h / 1.0e3) ** 0.094)
    R_s = rvir_kpc(Mv_1e9Msun, t_l_gyr, tun) / max(c, 1.0e-30)
    dphidr_dm = Mv_1e9Msun * (math.log(1.0 + r / R_s) / (r * r) - 1.0 / ((R_s + r) * r))

    spinprm = spin_norm / math.sqrt(
        2.0 * 6.67 * rvir_kpc(Mv_1e9Msun, t_l_gyr, tun) * kpc * Mv_1e9Msun * 2.0e28
    )
    sersic_re = max(spinprm * rvir_kpc(Mv_1e9Msun, t_l_gyr, tun) / math.sqrt(2.0), 1.0e-12)
    sersic_z = sersic_b * (r / sersic_re) ** (1.0 / 2.2)
    m_ser_r = Mstar_1e9Msun_SMHM(Mv_1e9Msun, t_l_gyr) * float(special.gammainc(2.2 * (3.0 - sersic_p), sersic_z))
    vc_bg = math.sqrt(max(r * dphidr_dm + m_ser_r / r, 0.0))
    return vc_bg * vc_bg / ((4.0 / 3.0) * PI * r * r)

def Redshift2CosmicTimeGyr(redshift: float) -> float:
    """Flat LCDM cosmic time in Gyr.

    For a spatially flat matter+Lambda cosmology, the age at redshift ``z`` has the analytic form

        t(z) = 2 / (3 H0 sqrt(Omega_L)) * asinh(sqrt(Omega_L / Omega_M) / (1+z)^(3/2))

    which is exact under the flat-LCDM assumption.
    """

    z = max(float(redshift), 0.0)
    return LCDM_TIME_PREFAC_GYR * math.asinh(LCDM_ASINH_RATIO / ((1.0 + z) ** 1.5))


def swf(t_gyr: float) -> float:
    t_safe = max(float(t_gyr), 1.0e-12)
    x = math.log10(t_safe) + 9.0
    return max(0.0, -(x * x) / 100.0 + 0.288 * x - 1.42)


def validate_tidal_stripping_mode(mode: str) -> str:
    if mode not in TIDAL_STRIPPING_CHOICES:
        allowed = ", ".join(TIDAL_STRIPPING_CHOICES)
        raise ValueError(f"tidal_stripping must be one of: {allowed}")
    return mode


def fragione_tidal_stripping_rate(
    m_gc_1e5: float,
    r_kpc: float,
    v_kms: float,
    prconst: float,
) -> float:
    p_r = prconst * r_kpc / max(v_kms, EPS)
    return (2.0 ** (2.0 / 3.0)) * 0.1 * (m_gc_1e5 ** (1.0 / 3.0)) / max(p_r, EPS)


def choksi_tidal_stripping_rate(m_gc_1e5: float) -> float:
    m = max(float(m_gc_1e5), EPS)
    m_over_2e5 = m / 2.0
    t_tid_gyr = 10.0 * (m_over_2e5 ** (2.0 / 3.0)) * CHOKSI_TIDAL_P
    t_iso_gyr = CHOKSI_TISO_COEFF_GYR * m_over_2e5
    return m / max(min(t_tid_gyr, t_iso_gyr), EPS)


def assign_bin_fast(
    r_kpc: float,
    r_min: float,
    log_r_min: float,
    inv_log_span: float,
    binnub: int,
) -> int:
    if r_kpc < r_min or inv_log_span <= 0.0:
        return 1
    frac = (math.log10(max(r_kpc, r_min)) - log_r_min) * inv_log_span
    b = 1 + int(math.floor(frac * (binnub - 1)))
    return max(1, min(binnub, b))


def cluster_halfmass_density(M_GC_1e5Msun: float) -> float:
    rho_h = 1.0e3 * (M_GC_1e5Msun**2)
    if M_GC_1e5Msun < 1.0:
        return 1.0e3
    if M_GC_1e5Msun > 10.0:
        return 1.0e5
    return rho_h


def vc_kms(Mencl_1e5Msun: float, r_kpc: float, rho_bkgd: float) -> float:
    r = max(r_kpc, 1.0e-12)
    rho_GC = Mencl_1e5Msun / ((4.0 / 3.0) * PI * (r**3)) / 1.0e4
    return math.sqrt(max((rho_bkgd + rho_GC) * (4.0 / 3.0) * PI * (r**2), 0.0)) * ALEX_VC_CONV


def _prefix_from_sumgc_total(m_sumgc_total: np.ndarray) -> np.ndarray:
    """Prefix sums of deposited mass by radial bin for fast enclosed-mass queries."""

    prefix = np.empty(len(m_sumgc_total) + 1, dtype=float)
    prefix[0] = 0.0
    prefix[1:] = np.cumsum(m_sumgc_total[:, 0], dtype=float)
    return prefix


def _enclosed_mass_before_bin_from_prefix(bin_index: int, prefix: np.ndarray) -> float:
    if bin_index <= 1:
        return 0.0
    return float(prefix[bin_index - 1])


def _deposit_delta_partial(dM_gc: float, dM_gc_sw: float) -> np.ndarray:
    """Deposit the mass removed during one finite timestep."""

    return np.array([dM_gc + dM_gc_sw, dM_gc, dM_gc], dtype=float)


def _deposit_amount(
    i: int,
    bin_index: int,
    dm: float,
    depo: np.ndarray,
    m_sumbin_total: np.ndarray,
    m_sumgc_total: np.ndarray,
) -> None:
    """Deposit a stellar-mass amount into one radial bin in-place."""

    if dm <= 0.0:
        return
    bi = bin_index - 1
    delta = np.array([dm, dm, dm], dtype=float)
    depo[i, bi, :] += delta
    m_sumbin_total[i, :] += delta
    m_sumgc_total[bi, :] += delta


def _deposit_full_mass(
    i: int,
    bin_index: int,
    m_gc: np.ndarray,
    depo: np.ndarray,
    m_sumbin_total: np.ndarray,
    m_sumgc_total: np.ndarray,
) -> None:
    """Move the remaining bound cluster mass into one radial bin in-place.

    A fully disrupted GC adds the same remaining mass to all three deposited
    channels because there is no surviving bound component left to distinguish.
    """

    dm = float(m_gc[i])
    _deposit_amount(i, bin_index, dm, depo, m_sumbin_total, m_sumgc_total)
    m_gc[i] = 0.0


def rk4_rdot_analytic(
    M_GC_1e5Msun: float,
    r_kpc: float,
    dt_gyr: float,
    rho_bg_current: float,
    m_enclosed_current_1e5: float,
    prefix_snapshot: np.ndarray,
    r_min: float,
    log_r_min: float,
    inv_log_span: float,
    binnub: int,
    *,
    spin_now: float,
    masshalo: float,
    t_l_gyr: float,
    sersic_n: float,
    tun: Tunables,
) -> float:
    """RK4 estimate of the radial inspiral rate using analytic rho_bg calls."""

    if dt_gyr <= 0.0:
        return 0.0

    v_c_kms = vc_kms(m_enclosed_current_1e5, r_kpc, rho_bg_current)
    if (r_kpc <= 0.0) or (v_c_kms <= 0.0):
        return 1.0e10
    dot_r = M_GC_1e5Msun / (0.45 * r_kpc * v_c_kms)
    k1 = dt_gyr * dot_r

    def _substep_rdot(rr: float) -> float:
        if rr <= 0.0:
            return 1.0e10
        rk_bin = assign_bin_fast(rr, r_min, log_r_min, inv_log_span, binnub)
        m_enclose = _enclosed_mass_before_bin_from_prefix(rk_bin, prefix_snapshot)
        rho_bg = rho_bg_sw1(rr, spin_now, masshalo, t_l_gyr, sersic_n, tun)
        v = vc_kms(m_enclose, rr, rho_bg)
        if v <= 0.0:
            return 1.0e10
        return M_GC_1e5Msun / (0.45 * rr * v)

    r2 = r_kpc - 0.5 * k1
    if r2 <= 0.0:
        k2 = 1.0e10
        k3 = 1.0e10
        k4 = 1.0e10
    else:
        k2 = dt_gyr * _substep_rdot(r2)
        r3 = r_kpc - 0.5 * k2
        if r3 <= 0.0:
            k3 = 1.0e10
            k4 = 1.0e10
        else:
            k3 = dt_gyr * _substep_rdot(r3)
            r4 = r_kpc - k3
            if r4 <= 0.0:
                k4 = 1.0e10
            else:
                k4 = dt_gyr * _substep_rdot(r4)
    return (k1 + 2.0 * k2 + 2.0 * k3 + k4) / (6.0 * max(dt_gyr, EPS))


def evolve_single_halo(
    ts_m: float,
    ts_r: float,
    gcini_path: Path,
    depos_path: Path,
    gcfin_path: Path,
    haloevo_path: Optional[Path] = None,
    tun: Optional[Tunables] = None,
    verbose: bool = True,
    *,
    sersic_n: float = 2.2,
    final_redshift: float = 0.0,
    df: int = 1,
    tidal_stripping: str = TIDAL_STRIPPING_FRAGIONE2019,
    trace_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evolve one halo's GC system from formation to z=0.

    The state is stored in legacy units for direct compatibility with the
    historical tables:

    - GC masses are kept internally in units of 1e5 Msun
    - times are cosmic time in Gyr
    - radii are kpc
    """

    tun = tun or Tunables()
    if haloevo_path is None:
        raise ValueError("haloevo_path is required")
    if sersic_n <= 0.0:
        raise ValueError("sersic_n must be positive")
    if final_redshift < 0.0:
        raise ValueError("final_redshift must be non-negative")
    if int(df) not in (0, 1):
        raise ValueError("df must be 0 or 1")
    df_enabled = bool(int(df))
    tidal_stripping = validate_tidal_stripping_mode(tidal_stripping)

    gc_init = _numeric_rows(gcini_path)
    if gc_init.size == 0:
        raise ValueError(f"No usable GC rows found in {gcini_path}")

    n_gc = gc_init.shape[0]
    if gc_init.shape[1] < 10:
        raise ValueError("GCini rows must have at least 10 columns")

    # Modern GCini rows use the 13-column formation catalog.
    m_gc_init = 10.0 ** (gc_init[:, 6] - 5.0)
    r_gc_init = gc_init[:, 9].astype(float)
    t_gc_init = np.array([Redshift2CosmicTimeGyr(z) for z in gc_init[:, 7]], dtype=float)
    if gc_init.shape[1] > 12:
        m_imbh = np.asarray(gc_init[:, 12], dtype=float) / 1.0e5
    else:
        m_imbh = np.zeros(n_gc, dtype=float)

    t_end = Redshift2CosmicTimeGyr(final_redshift)
    if np.any(t_gc_init > t_end + 1.0e-10):
        raise ValueError(
            "GCini contains clusters formed after the requested final_redshift. "
            "Regenerate the formation catalog with the same final redshift."
        )

    if np.any(~np.isfinite(m_imbh)):
        raise ValueError("IMBH masses must be finite.")
    if np.any(m_imbh < 0.0):
        raise ValueError("IMBH masses must be non-negative.")
    if np.any(m_imbh > m_gc_init + 1.0e-12):
        raise ValueError("IMBH masses cannot exceed their parent GC initial masses.")

    m_gc = m_gc_init.copy()
    r_gc = r_gc_init.copy()
    t_gc = t_gc_init.copy()
    status = np.full(n_gc, STATUS_ALIVE, dtype=int)
    is_wanderer = m_imbh >= (m_gc - 1.0e-12)
    m_gc[is_wanderer] = m_imbh[is_wanderer]

    r_min = tun.r_min
    r_max = max(float(np.max(r_gc_init)), r_min * 1.0001)
    log_r_min = math.log10(r_min)
    log_r_max = math.log10(r_max)
    inv_log_span = 0.0 if log_r_max <= log_r_min else 1.0 / (log_r_max - log_r_min)
    if tun.binnub == 1:
        bin_edges = np.array([0.0, r_max], dtype=float)
    else:
        frac = np.arange(tun.binnub, dtype=float) / float(tun.binnub - 1)
        edges = 10.0 ** (log_r_min + (log_r_max - log_r_min) * frac)
        bin_edges = np.concatenate(([0.0], edges))
    bin_gc = np.array(
        [assign_bin_fast(rr, r_min, log_r_min, inv_log_span, tun.binnub) for rr in r_gc],
        dtype=int,
    )

    depo = np.zeros((n_gc, tun.binnub, 3), dtype=float)
    m_sumbin_total = np.zeros((n_gc, 3), dtype=float)
    m_sumgc_total = np.zeros((tun.binnub, 3), dtype=float)

    prconst = 100.0

    halo = _read_haloevo_mpb(haloevo_path)
    if halo.shape[0] == 0:
        raise ValueError(f"No usable halo rows found in {haloevo_path}")
    mhalo = 10.0 ** (halo[:, 0] - 9.0)
    bg_time = np.array([Redshift2CosmicTimeGyr(z) for z in halo[:, 5]], dtype=float)
    spin_norm = np.sqrt(halo[:, 6] ** 2 + halo[:, 7] ** 2 + halo[:, 8] ** 2) * kpc / tun.h * 1.0e3

    tposini = int(math.floor(float(np.min(t_gc_init)) / max(t_end, EPS) * tun.t_div))
    tposini = max(0, min(tun.t_div, tposini))
    tpos = 1 + tposini
    snap_pos = 0
    start = time.time()
    dt_gc = np.full(n_gc, t_end, dtype=float)
    mdot_td = np.zeros(n_gc, dtype=float)
    rdot_df = np.zeros(n_gc, dtype=float)

    if depos_path.exists():
        depos_path.unlink()
    with depos_path.open("w") as fdep:
        fdep.write("# " + DEPOS_HEADER.replace("\n", "\n# ") + "\n")
    trace_fh = None
    if trace_path is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if trace_path.exists():
            trace_path.unlink()
        trace_fh = trace_path.open("w")
        trace_fh.write("# " + TRACE_HEADER.replace("\n", "\n# ") + "\n")
    trace_index = 0

    spin_now = float(spin_norm[0]) if len(spin_norm) > 0 else 0.0
    masshalo = float(mhalo[0]) if len(mhalo) > 0 else 0.0
    t_l_block = 0.0

    def remaining_stellar_mass(i: int) -> float:
        return max(float(m_gc[i] - m_imbh[i]), 0.0)

    def current_status_for_trace(i: int) -> int:
        if status[i] == STATUS_ALIVE and is_wanderer[i]:
            return STATUS_WANDERER
        return int(status[i])

    def write_trace_row(i: int, phase: str) -> None:
        nonlocal trace_index
        if trace_fh is None:
            return
        trace_index += 1
        t_now = float(t_gc[i])
        trace_fh.write(
            f"{trace_index:d} {phase} {i + 1:d} {current_status_for_trace(i):d} "
            f"{t_now:.10e} {CosmicTimeGyr2Redshift(t_now):.10e} "
            f"{r_gc[i]:.10e} {1.0e5 * m_gc[i]:.10e} {int(bin_gc[i]):d}\n"
        )

    def cap_bound_mass_loss(i: int, dm: float) -> float:
        loss = min(max(float(dm), 0.0), float(m_gc[i]))
        if m_imbh[i] > 0.0:
            loss = min(loss, max(float(m_gc[i] - m_imbh[i]), 0.0))
        return loss

    def deposit_remaining_stars(i: int, bin_index: int) -> None:
        _deposit_amount(i, bin_index, remaining_stellar_mass(i), depo, m_sumbin_total, m_sumgc_total)

    def enter_wanderer(i: int, bin_index: int, *, deposit_stars: bool) -> None:
        if status[i] != STATUS_ALIVE:
            return
        if m_imbh[i] <= 0.0:
            return
        if deposit_stars:
            deposit_remaining_stars(i, bin_index)
        m_gc[i] = m_imbh[i]
        is_wanderer[i] = True

    def sink_bound_gc(i: int, bin_index: int) -> None:
        if m_imbh[i] > 0.0:
            deposit_remaining_stars(i, bin_index)
            m_gc[i] = 0.0
        else:
            _deposit_full_mass(i, bin_index, m_gc, depo, m_sumbin_total, m_sumgc_total)
        status[i] = STATUS_SUNK

    def rho_bg_current_block(r_now: float) -> float:
        rr = max(float(r_now), 1.0e-12)
        return rho_bg_sw1(rr, spin_now, masshalo, t_l_block, sersic_n, tun)

    def rho_components(
        r_now: float,
        bin_index: int,
        prefix_snapshot: np.ndarray,
    ) -> Tuple[float, float, float]:
        m_enclose = _enclosed_mass_before_bin_from_prefix(bin_index, prefix_snapshot)
        rho_bg = rho_bg_current_block(r_now)
        rho_tot = rho_bg + m_enclose / ((4.0 / 3.0) * PI * (max(r_now, 1.0e-12) ** 3)) / 1.0e4
        return m_enclose, rho_bg, rho_tot

    def current_rdot(
        mass_df: float,
        i: int,
        prefix_snapshot: np.ndarray,
    ) -> float:
        if not df_enabled:
            return 0.0
        b_now = assign_bin_fast(r_gc[i], r_min, log_r_min, inv_log_span, tun.binnub)
        m_enclose, _, rho_tot = rho_components(r_gc[i], b_now, prefix_snapshot)
        return rk4_rdot_analytic(
            M_GC_1e5Msun=mass_df,
            r_kpc=r_gc[i],
            dt_gyr=dt_gc[i],
            rho_bg_current=rho_tot,
            m_enclosed_current_1e5=m_enclose,
            prefix_snapshot=prefix_snapshot,
            r_min=r_min,
            log_r_min=log_r_min,
            inv_log_span=inv_log_span,
            binnub=tun.binnub,
            spin_now=spin_now,
            masshalo=masshalo,
            t_l_gyr=t_l_block,
            sersic_n=sersic_n,
            tun=tun,
        )

    def prepare_gc_step(
        i: int,
        prefix_snapshot: np.ndarray,
        t_r: float,
    ) -> None:
        """Prepare one active GC or IMBH wanderer against the current deposit field."""

        dt_gc[i] = t_end
        if status[i] != STATUS_ALIVE:
            return
        if m_gc_init[i] <= 1.0e-2:
            return
        if t_gc[i] >= t_r:
            return
        if (not is_wanderer[i]) and (m_imbh[i] > 0.0) and (m_gc[i] <= m_imbh[i] + 1.0e-12):
            enter_wanderer(i, int(bin_gc[i]), deposit_stars=False)

        b = int(bin_gc[i])
        m_enclose, _, rho_tot = rho_components(r_gc[i], b, prefix_snapshot)

        if not is_wanderer[i]:
            if cluster_halfmass_density(m_gc[i]) < rho_tot:
                if m_imbh[i] > 0.0:
                    enter_wanderer(i, b, deposit_stars=True)
                else:
                    status[i] = STATUS_TORN
                    _deposit_full_mass(i, b, m_gc, depo, m_sumbin_total, m_sumgc_total)
                    write_trace_row(i, "prep")
                    return

        v = vc_kms(m_enclose, r_gc[i], rho_tot)

        if is_wanderer[i]:
            mdot_td[i] = 0.0
            if v <= 0.0 or not df_enabled:
                rdot_df[i] = 0.0
                dt_gc[i] = min(max(t_r - t_gc[i], 0.0), tun.dt_max)
                return
            dot_r = m_imbh[i] / (0.45 * max(r_gc[i], EPS) * max(v, EPS))
            dt_orb = ts_r * r_gc[i] / max(dot_r, EPS)
            if t_gc[i] + dt_orb > t_r:
                dt_orb = t_r - t_gc[i]
            elif dt_orb < ts_r * tun.t_limit:
                dt_orb = ts_r * tun.t_limit
            dt_gc[i] = min(dt_orb, tun.dt_max)
            rdot_df[i] = current_rdot(m_imbh[i], i, prefix_snapshot)
            return

        if tidal_stripping == TIDAL_STRIPPING_CHOKSI2018:
            mdot_td[i] = choksi_tidal_stripping_rate(m_gc[i])
        else:
            if v <= 0.0:
                mdot_td[i] = 0.0
                rdot_df[i] = 0.0
                dt_gc[i] = min(max(t_r - t_gc[i], 0.0), tun.dt_max)
                return
            mdot_td[i] = fragione_tidal_stripping_rate(m_gc[i], r_gc[i], v, prconst)

        dtm = ts_m * m_gc[i] / max(mdot_td[i], EPS)
        if t_gc[i] + dtm > t_r:
            dtm = t_r - t_gc[i]
        elif dtm < ts_m * tun.t_limit:
            dtm = ts_m * tun.t_limit

        if v <= 0.0:
            rdot_df[i] = 0.0
            dt_gc[i] = min(dtm, tun.dt_max)
            return

        if not df_enabled:
            rdot_df[i] = 0.0
            dt_gc[i] = min(dtm, tun.dt_max)
            return

        dot_r = m_gc[i] / (0.45 * max(r_gc[i], EPS) * max(v, EPS))
        dt_orb = ts_r * r_gc[i] / max(dot_r, EPS)
        if t_gc[i] + dt_orb > t_r:
            dt_orb = t_r - t_gc[i]
        elif dt_orb < ts_r * tun.t_limit:
            dt_orb = ts_r * tun.t_limit

        dt_gc[i] = min(dtm, dt_orb, tun.dt_max)
        rdot_df[i] = current_rdot(m_gc[i], i, prefix_snapshot)

    for i in range(n_gc):
        write_trace_row(i, "init")

    while tpos <= tun.t_div:
        t_l = t_end * (tpos - 1) / tun.t_div
        t_r = t_end * tpos / tun.t_div
        t_l_block = t_l

        while snap_pos < len(bg_time) and bg_time[snap_pos] <= t_l:
            snap_pos += 1
        if snap_pos == 0:
            masshalo = float(mhalo[0])
            spin_now = float(spin_norm[0])
        elif snap_pos >= len(mhalo):
            masshalo = float(mhalo[-1])
            spin_now = float(spin_norm[-1])
        else:
            t0 = bg_time[snap_pos - 1]
            t1 = bg_time[snap_pos]
            if t1 == t0:
                masshalo = float(mhalo[snap_pos - 1])
            else:
                masshalo = float(
                    mhalo[snap_pos - 1]
                    + (mhalo[snap_pos] - mhalo[snap_pos - 1]) * (t_l - t0) / (t1 - t0)
                )
            spin_now = float(spin_norm[snap_pos - 1])
        prefix_snapshot = _prefix_from_sumgc_total(m_sumgc_total)
        for i in range(n_gc):
            prepare_gc_step(i, prefix_snapshot, t_r)

        next_i = int(np.argmin(t_gc + dt_gc))

        while (t_gc[next_i] + dt_gc[next_i]) < t_r:
            prefix_snapshot = _prefix_from_sumgc_total(m_sumgc_total)
            i = next_i
            dM_gc_sw = 0.0
            dM_gc = 0.0

            if not is_wanderer[i]:
                delta_swf = swf(t_gc[i] + dt_gc[i] - t_gc_init[i]) - swf(t_gc[i] - t_gc_init[i])
                denom = m_gc[i] + m_sumbin_total[i, 2]
                if denom > 0.0 and delta_swf != 0.0:
                    for l in range(tun.binnub):
                        dM_star = m_gc_init[i] * depo[i, l, 2] / denom * delta_swf
                        new_val = max(0.0, depo[i, l, 2] - dM_star)
                        removed = depo[i, l, 2] - new_val
                        depo[i, l, 2] = new_val
                        m_sumbin_total[i, 2] -= removed
                        m_sumgc_total[l, 2] -= removed
                    dM_gc_sw = m_gc_init[i] * m_gc[i] / denom * delta_swf

                dM_gc_sw = cap_bound_mass_loss(i, dM_gc_sw)
                m_gc[i] -= dM_gc_sw
                if (m_imbh[i] > 0.0) and (m_gc[i] <= m_imbh[i] + 1.0e-12):
                    m_gc[i] = m_imbh[i]
                    is_wanderer[i] = True

                if not is_wanderer[i]:
                    dM_gc = cap_bound_mass_loss(i, dt_gc[i] * mdot_td[i])
                    m_gc[i] -= dM_gc
                    if (m_imbh[i] > 0.0) and (m_gc[i] <= m_imbh[i] + 1.0e-12):
                        m_gc[i] = m_imbh[i]
                        is_wanderer[i] = True

            if dM_gc > 0.0 or dM_gc_sw > 0.0:
                bi = int(bin_gc[i]) - 1
                delta = _deposit_delta_partial(dM_gc, dM_gc_sw)
                depo[i, bi, :] += delta
                m_sumbin_total[i, :] += delta
                m_sumgc_total[bi, :] += delta

            dR = dt_gc[i] * (current_rdot(m_imbh[i], i, prefix_snapshot)
                             if is_wanderer[i] else rdot_df[i])
            r_gc[i] = max(0.0, r_gc[i] - dR)
            bin_gc[i] = assign_bin_fast(r_gc[i], r_min, log_r_min, inv_log_span, tun.binnub)
            t_gc[i] += dt_gc[i]

            b = int(bin_gc[i])
            if is_wanderer[i]:
                if r_gc[i] <= tun.r_sink:
                    dt_gc[i] = t_end
                    status[i] = STATUS_WANDERER_SUNK
                    m_gc[i] = 0.0
                else:
                    prepare_gc_step(i, prefix_snapshot, t_r)
            else:
                _, _, rho_tot = rho_components(r_gc[i], b, prefix_snapshot)
                rho_h = cluster_halfmass_density(m_gc[i])

                if m_gc[i] <= 0.0:
                    dt_gc[i] = t_end
                    status[i] = STATUS_EXHAUSTED
                    m_gc[i] = 0.0
                elif r_gc[i] <= tun.r_sink:
                    dt_gc[i] = t_end
                    sink_bound_gc(i, b)
                elif rho_h < rho_tot:
                    dt_gc[i] = t_end
                    if m_imbh[i] > 0.0:
                        enter_wanderer(i, b, deposit_stars=True)
                        prepare_gc_step(i, prefix_snapshot, t_r)
                    else:
                        status[i] = STATUS_TORN
                        _deposit_full_mass(i, b, m_gc, depo, m_sumbin_total, m_sumgc_total)
                else:
                    prepare_gc_step(i, prefix_snapshot, t_r)

            write_trace_row(i, "event")
            next_i = int(np.argmin(t_gc + dt_gc))

        prefix_snapshot = _prefix_from_sumgc_total(m_sumgc_total)
        sumbin_stage_col2 = m_sumbin_total[:, 2].copy()
        for i in range(n_gc):
            if (t_gc[i] >= t_r) or (status[i] != STATUS_ALIVE) or (m_gc_init[i] <= 1.0e-2):
                continue

            dM_gc_sw = 0.0
            dM_gc = 0.0
            if not is_wanderer[i]:
                delta_swf = swf(t_gc[i] + dt_gc[i] - t_gc_init[i]) - swf(t_gc[i] - t_gc_init[i])
                denom = m_gc[i] + sumbin_stage_col2[i]
                if denom > 0.0 and delta_swf != 0.0:
                    for l in range(tun.binnub):
                        dM_star = m_gc_init[i] * depo[i, l, 2] / denom * delta_swf
                        depo[i, l, 2] = depo[i, l, 2] - dM_star
                        m_sumbin_total[i, 2] -= dM_star
                        m_sumgc_total[l, 2] -= dM_star
                    dM_gc_sw = m_gc_init[i] * m_gc[i] / denom * delta_swf

                dM_gc_sw = cap_bound_mass_loss(i, dM_gc_sw)
                m_gc[i] -= dM_gc_sw
                if (m_imbh[i] > 0.0) and (m_gc[i] <= m_imbh[i] + 1.0e-12):
                    m_gc[i] = m_imbh[i]
                    is_wanderer[i] = True

                if not is_wanderer[i]:
                    dM_gc = cap_bound_mass_loss(i, dt_gc[i] * mdot_td[i])
                    m_gc[i] -= dM_gc
                    if (m_imbh[i] > 0.0) and (m_gc[i] <= m_imbh[i] + 1.0e-12):
                        m_gc[i] = m_imbh[i]
                        is_wanderer[i] = True

            if dM_gc > 0.0 or dM_gc_sw > 0.0:
                bi = int(bin_gc[i]) - 1
                delta = _deposit_delta_partial(dM_gc, dM_gc_sw)
                depo[i, bi, :] += delta
                m_sumbin_total[i, :] += delta
                m_sumgc_total[bi, :] += delta

            dR = dt_gc[i] * (current_rdot(m_imbh[i], i, prefix_snapshot)
                             if is_wanderer[i] else rdot_df[i])
            r_gc[i] = max(0.0, r_gc[i] - dR)
            bin_gc[i] = assign_bin_fast(r_gc[i], r_min, log_r_min, inv_log_span, tun.binnub)
            t_gc[i] = t_r

            b = int(bin_gc[i])
            if is_wanderer[i]:
                if r_gc[i] <= tun.r_sink:
                    dt_gc[i] = t_end
                    status[i] = STATUS_WANDERER_SUNK
                    m_gc[i] = 0.0
            else:
                _, _, rho_tot = rho_components(r_gc[i], b, prefix_snapshot)
                rho_h = cluster_halfmass_density(m_gc[i])

                if m_gc[i] <= 0.0:
                    dt_gc[i] = t_end
                    status[i] = STATUS_EXHAUSTED
                    m_gc[i] = 0.0
                elif r_gc[i] <= tun.r_sink:
                    dt_gc[i] = t_end
                    sink_bound_gc(i, b)
                elif rho_h < rho_tot:
                    dt_gc[i] = t_end
                    if m_imbh[i] > 0.0:
                        enter_wanderer(i, b, deposit_stars=True)
                    else:
                        status[i] = STATUS_TORN
                        _deposit_full_mass(i, b, m_gc, depo, m_sumbin_total, m_sumgc_total)

            write_trace_row(i, "coarse")

        with depos_path.open("a") as fdep:
            for l in range(tun.binnub):
                fdep.write(
                    f"{t_end - t_r:.10e} {l + 1:d} {bin_edges[l]:.10e} {bin_edges[l + 1]:.10e} "
                    f"{1.0e5 * m_sumgc_total[l, 0]:.10e} {1.0e5 * m_sumgc_total[l, 1]:.10e} "
                    f"{1.0e5 * m_sumgc_total[l, 2]:.10e}\n"
                )

        if verbose:
            print(f"{tpos - tposini:5d} / {tun.t_div - tposini:5d}  runtime={time.time() - start:8.3f} s")
        tpos += 1

    status_out = status.copy()
    status_out[(status == STATUS_ALIVE) & is_wanderer] = STATUS_WANDERER

    with gcfin_path.open("w") as fgc:
        fgc.write("# " + FINAL_GC_HEADER.replace("\n", "\n# ") + "\n")
        for i in range(n_gc):
            fgc.write(
                f"{i + 1:d} {int(status_out[i]):d} {1.0e5 * m_gc[i]:.10e} {1.0e5 * m_gc_init[i]:.10e} "
                f"{t_end - t_gc[i]:.10e} {t_end - t_gc_init[i]:.10e} {r_gc[i]:.10e} {r_gc_init[i]:.10e}\n"
            )

    finalGCs_array = np.column_stack((
        np.arange(1, n_gc + 1, dtype=float),
        status_out.astype(float),
        1.0e5 * m_gc,
        1.0e5 * m_gc_init,
        t_end - t_gc,
        t_end - t_gc_init,
        r_gc,
        r_gc_init,
    ))
    if trace_fh is not None:
        trace_fh.close()

    return finalGCs_array, depo


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analytical Python rewrite of the active Gao+2023 GC evolution path")
    p.add_argument("ts_m", type=float, help="mass-loss timestep factor")
    p.add_argument("ts_r", type=float, help="radial-decay timestep factor")
    p.add_argument("gcini", type=Path, help="GC initial table")
    p.add_argument("depos", type=Path, help="deposit-profile output file")
    p.add_argument("gcfin", type=Path, help="final-GC output file")
    p.add_argument("haloevo", type=Path, help="halo evolution table")
    p.add_argument("--DF", dest="df", type=int, choices=[0, 1], default=1, help="if 1, enable dynamical-friction orbital decay; if 0, keep radii fixed against DF")
    p.add_argument("--tidal_stripping", choices=TIDAL_STRIPPING_CHOICES, default=TIDAL_STRIPPING_FRAGIONE2019, help="continuous tidal-stripping prescription: current Fragione+2019 local-orbit rate or Choksi+2018 fixed-P disruption rate")
    p.add_argument("--ns", type=float, default=2.2, help="Sersic index N_s used in the background stellar profile")
    p.add_argument("--final-redshift", type=float, default=0.0, help="stop the evolution at this redshift instead of z=0")
    p.add_argument("--trace", type=Path, default=None, help="optional orbit-trace output path")
    p.add_argument("--quiet", action="store_true", help="disable progress prints")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    evolve_single_halo(
        ts_m=args.ts_m,
        ts_r=args.ts_r,
        gcini_path=args.gcini,
        depos_path=args.depos,
        gcfin_path=args.gcfin,
        haloevo_path=args.haloevo,
        verbose=not args.quiet,
        df=args.df,
        tidal_stripping=args.tidal_stripping,
        sersic_n=args.ns,
        final_redshift=args.final_redshift,
        trace_path=args.trace,
    )


if __name__ == "__main__":
    main()
