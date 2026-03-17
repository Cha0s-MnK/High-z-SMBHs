# Licensed under BSD-3-Clause License - see LICENSE

"""
IMBH seeding model for GC formation outputs.

This module implements a compact GC-level IMBH estimator based on:

- Eq. (7): cluster mass-size relation.
- Eq. (9): IMBH mass fit in the calibrated simulation range.
- Eq. (10): high-surface-density extrapolation fit.

When a GC radius is available, the projected half-mass surface density is computed directly from the Plummer projected half-mass
radius. This keeps the estimator aligned with the fitted quantity in Eqs. (9) and (10), namely Sigma_h.

Reference:
Rantala et al. (2026), "FROST-CLUSTERS -- III. Metallicity-dependent intermediate mass black hole formation by runaway collisions
in dense star clusters".
"""

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np


Number = Union[float, np.ndarray]


@dataclass(frozen=True)
class IMBHModelConfig:
    """Configuration for the IMBH seeding model."""

    # Global on/off switch for IMBH mass estimation in the GC pipeline.
    enabled: bool = False

    # Eq. (7) parameters
    fh: float = 0.125
    beta: float = 0.180
    r4_pc: float = 2.365
    # Eq. (7) uses the mass scale 10^4 Msun, while the prefactor is fh * R4 / 1.3.
    mass_pivot_msun: float = 1.0e4

    # Metallicity input mode: "feh" means [Fe/H] = log10(Z/Zsun)
    metallicity_kind: str = "feh"
    z_ratio_min: float = 1.0e-2
    z_ratio_max: float = 1.0

    # Shared high-density threshold from the upper Eq. (9) table value.
    lg_Sigma_max: float = 5.22
    min_imbh_mass_msun: float = 100.0


class IMBHModel:
    """IMBH model wrapper used by the GC formation workflow."""

    _PLUMMER_RH3D_TO_RHPROJ = 1.305

    # Table 4 limits and coefficients for Eq. (9)
    _EQ9_BREAKS = (0.126, 0.398)
    _EQ9 = {
        "A1": np.array([-1790.07, 9707.84, 1147.68]),
        "A2": np.array([-7392.65, 2627.71, -721.18]),
        "B1": np.array([162.46, -1015.72, -166.92]),
        "B2": np.array([829.42, -211.38, 126.15]),
        "C1": np.array([4734.11, -23585.20, -2002.25]),
        "C2": np.array([16556.82, -7814.04, 471.59]),
        "S1": np.array([0.0386, 0.91, 0.91]),
        "S2": np.array([4.53, 5.22, 5.22]),
    }

    # Table 5 limits and coefficients for Eq. (10)
    _EQ10_BREAKS = (0.079, 0.316)
    _EQ10 = {
        "D1": np.array([-37.37, -922.15, -611.27]),
        "D2": np.array([1452.33, 466.71, 628.40]),
        "E1": np.array([81.66, 4242.58, 2620.25]),
        "E2": np.array([-6892.57, -2280.02, -3137.93]),
    }

    def __init__(self, config: IMBHModelConfig = IMBHModelConfig()):
        self.config = config

    @classmethod
    def from_params(cls, params: Dict) -> "IMBHModel":
        """Create an IMBH model from the global params dictionary."""

        cfg = IMBHModelConfig(
            enabled=bool(params.get("imbh_model", False)),
            fh=params.get("imbh_fh", 0.125),
            beta=params.get("imbh_beta", 0.180),
            r4_pc=params.get("imbh_r4_pc", 2.365),
            mass_pivot_msun=params.get("imbh_mass_pivot_msun", 1.0e4),
            metallicity_kind=params.get("imbh_metallicity_kind", "feh"),
            z_ratio_min=params.get("imbh_z_ratio_min", 1.0e-2),
            z_ratio_max=params.get("imbh_z_ratio_max", 1.0),
            lg_Sigma_max=params.get("imbh_lg_Sigma_max", 5.22),
            min_imbh_mass_msun=params.get("imbh_min_mass_msun", 100.0),
        )
        return cls(cfg)

    @classmethod
    def is_enabled(cls, params: Dict) -> bool:
        """Return whether IMBH estimation should run for this parameter set."""
        return bool(params.get("imbh_model", False))

    @staticmethod
    def _as_1d(arr: Number) -> (np.ndarray, bool):
        # Normalize scalar/array inputs so the public methods can share one
        # vectorized implementation and restore scalars on return.
        out = np.asarray(arr, dtype=float)
        scalar = out.ndim == 0
        if scalar:
            out = out.reshape(1)
        return out, scalar

    @staticmethod
    def _broadcast_pair(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        # This module accepts either one mass with many metallicities, or vice
        # versa, so we do a minimal manual broadcast instead of requiring exact
        # shape matches everywhere.
        if x.size == y.size:
            return x, y
        if x.size == 1:
            return np.full_like(y, x[0], dtype=float), y
        if y.size == 1:
            return x, np.full_like(x, y[0], dtype=float)
        raise ValueError("Inputs are not broadcast-compatible.")

    @staticmethod
    def _piecewise_linear(log_z_ratio: np.ndarray, z_ratio: np.ndarray,
                          z_breaks, k: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Tables 4 and 5 in Rantala+2026 are piecewise-linear in log(Z/Zsun).
        out = np.empty_like(log_z_ratio)
        m0 = z_ratio < z_breaks[0]
        m1 = (z_ratio >= z_breaks[0]) & (z_ratio < z_breaks[1])
        m2 = ~(m0 | m1)
        out[m0] = k[0] * log_z_ratio[m0] + b[0]
        out[m1] = k[1] * log_z_ratio[m1] + b[1]
        out[m2] = k[2] * log_z_ratio[m2] + b[2]
        return out

    def metallicity_to_z_ratio(self, metallicity: Number) -> Number:
        """Convert metallicity input to Z/Zsun.

        If `metallicity_kind` is "feh", metallicity is interpreted as [Fe/H].
        If it is "z_ratio", input is interpreted directly as Z/Zsun.
        """

        met, scalar = self._as_1d(metallicity)
        if self.config.metallicity_kind == "feh":
            z_ratio = np.power(10.0, met)
        elif self.config.metallicity_kind == "z_ratio":
            z_ratio = np.array(met, copy=True)
        else:
            raise ValueError("Unknown metallicity kind: %s" % self.config.metallicity_kind)
        if scalar:
            return float(z_ratio[0])
        return z_ratio

    def radius_eq7(self, cluster_mass_msun: Number) -> Number:
        """Eq. (7): 3D half-mass radius in pc."""

        mass, scalar = self._as_1d(cluster_mass_msun)
        mass_safe = np.clip(mass, 1e-30, None)
        r_h_pc = (
            self.config.fh
            * self.config.r4_pc
            / 1.3
            * np.power(mass_safe / self.config.mass_pivot_msun, self.config.beta)
        )
        if scalar:
            return float(r_h_pc[0])
        return r_h_pc

    @classmethod
    def projected_half_mass_radius_plummer(cls, r_h_pc: Number) -> Number:
        """Convert Plummer 3D half-mass radius to projected half-mass radius."""

        radius, scalar = cls._as_1d(r_h_pc)
        projected = radius / cls._PLUMMER_RH3D_TO_RHPROJ
        if scalar:
            return float(projected[0])
        return projected

    @classmethod
    def sigma_h_from_mass_radius(
        cls,
        cluster_mass_msun: Number,
        r_h_pc: Number,
    ) -> Number:
        """Projected half-mass surface density from GC mass and radius.

        The input ``r_h_pc`` is always interpreted as the 3D half-mass radius.
        For a Plummer profile, Sigma_h = M / (2 pi R_h,proj^2) with
        R_h,proj = r_h / 1.305.
        """

        mass, scalar_m = cls._as_1d(cluster_mass_msun)
        radius, scalar_r = cls._as_1d(r_h_pc)
        mass, radius = cls._broadcast_pair(mass, radius)
        projected = cls.projected_half_mass_radius_plummer(radius)

        sigma_h = np.zeros_like(mass)
        m = projected > 0
        sigma_h[m] = mass[m] / (2.0 * np.pi * np.power(projected[m], 2))
        if scalar_m and scalar_r:
            return float(sigma_h[0])
        return sigma_h

    def _eq9_coeffs(self, z_ratio: np.ndarray):
        """Interpolate the Eq. (9) coefficient set at the requested metallicity."""

        log_z = np.log10(np.clip(z_ratio, 1e-30, None))
        A = self._piecewise_linear(log_z, z_ratio, self._EQ9_BREAKS, self._EQ9["A1"], self._EQ9["A2"])
        B = self._piecewise_linear(log_z, z_ratio, self._EQ9_BREAKS, self._EQ9["B1"], self._EQ9["B2"])
        C = self._piecewise_linear(log_z, z_ratio, self._EQ9_BREAKS, self._EQ9["C1"], self._EQ9["C2"])
        log_sigma_crit = self._piecewise_linear(
            log_z, z_ratio, self._EQ9_BREAKS, self._EQ9["S1"], self._EQ9["S2"]
        )
        return A, B, C, log_sigma_crit

    def _eq10_coeffs(self, z_ratio: np.ndarray):
        """Interpolate the Eq. (10) coefficient set at the requested metallicity."""

        log_z = np.log10(np.clip(z_ratio, 1e-30, None))
        D = self._piecewise_linear(log_z, z_ratio, self._EQ10_BREAKS, self._EQ10["D1"], self._EQ10["D2"])
        E = self._piecewise_linear(log_z, z_ratio, self._EQ10_BREAKS, self._EQ10["E1"], self._EQ10["E2"])
        return D, E

    def imbh_mass_eq9(self, sigma_h_msun_pc2: np.ndarray, z_ratio: np.ndarray) -> np.ndarray:
        """Eq. (9): IMBH mass fit inside the calibrated density range."""

        A, B, C, log_sigma_crit = self._eq9_coeffs(z_ratio)
        log_sigma_h = np.log10(np.clip(sigma_h_msun_pc2, 1e-30, None))
        sigma_crit = np.power(10.0, log_sigma_crit)
        # Below the critical density the fit is defined to contribute zero IMBH mass.
        active = sigma_h_msun_pc2 >= sigma_crit
        mass = A * log_sigma_h + B * np.power(log_sigma_h, 2) + C
        mass = np.where(active, mass, 0.0)
        mass = np.where(np.isfinite(mass), mass, 0.0)
        return np.clip(mass, 0.0, None)

    def imbh_mass_eq10(self, sigma_h_msun_pc2: np.ndarray, z_ratio: np.ndarray) -> np.ndarray:
        """Eq. (10): high-density extrapolation for IMBH mass.

        The caller is responsible for selecting only the high-Sigma_h regime.
        """

        D, E = self._eq10_coeffs(z_ratio)
        log_sigma_h = np.log10(np.clip(sigma_h_msun_pc2, 1e-30, None))
        mass = D * log_sigma_h + E
        mass = np.where(np.isfinite(mass), mass, 0.0)
        return np.clip(mass, 0.0, None)

    def imbh_mass_from_sigma_metallicity(self, sigma_h_msun_pc2: Number,
                                         z_ratio: Number) -> Number:
        """Estimate IMBH mass from surface density and metallicity ratio.

        The branch uses one shared high-density transition at log10(Sigma_h)=5.22.
        """

        sigma_h, scalar_s = self._as_1d(sigma_h_msun_pc2)
        z_ratio_arr, scalar_z = self._as_1d(z_ratio)
        sigma_h, z_ratio_arr = self._broadcast_pair(sigma_h, z_ratio_arr)

        z_clip = np.clip(z_ratio_arr, self.config.z_ratio_min, self.config.z_ratio_max)
        log_sigma_h = np.log10(np.clip(sigma_h, 1e-30, None))
        use_eq10 = log_sigma_h >= self.config.lg_Sigma_max

        imbh_mass = self.imbh_mass_eq9(sigma_h, z_clip)
        if np.any(use_eq10):
            imbh_mass = np.array(imbh_mass, copy = True)
            imbh_mass[use_eq10] = self.imbh_mass_eq10(sigma_h[use_eq10], z_clip[use_eq10])
        imbh_mass = np.where(imbh_mass >= self.config.min_imbh_mass_msun, imbh_mass, 0.0)

        if scalar_s and scalar_z:
            return float(imbh_mass[0])
        return imbh_mass

    def estimate_for_gc(
        self,
        cluster_mass_msun: Number,
        metallicity: Number,
    ) -> Dict[str, Number]:
        """Full GC-level estimate for one GC or a broadcast-compatible GC array.

        The GC half-mass radius is always computed internally from Eq. (7).
        """

        mass, scalar_m = self._as_1d(cluster_mass_msun)
        met, scalar_z = self._as_1d(metallicity)
        radius = np.asarray(self.radius_eq7(mass), dtype=float)
        scalar_r = scalar_m
        mass, radius, met = np.broadcast_arrays(mass, radius, met)
        r_h_3d_pc = radius.astype(float, copy=False)

        sigma_h = self.sigma_h_from_mass_radius(mass, r_h_3d_pc)

        z_ratio_raw = self.metallicity_to_z_ratio(met)
        if np.isscalar(z_ratio_raw):
            z_ratio_raw = np.array([z_ratio_raw], dtype=float)
        z_ratio = np.clip(z_ratio_raw, self.config.z_ratio_min, self.config.z_ratio_max)
        imbh_mass = self.imbh_mass_from_sigma_metallicity(sigma_h, z_ratio)

        if np.isscalar(imbh_mass):
            imbh_mass = np.array([imbh_mass], dtype=float)

        output = {
            "r_h_pc": r_h_3d_pc,
            "sigma_h_msun_pc2": sigma_h,
            "z_ratio_input": z_ratio_raw,
            "z_ratio_used": z_ratio,
            "imbh_mass_msun": imbh_mass,
        }
        if scalar_m and scalar_r and scalar_z:
            return {k: float(v[0]) for k, v in output.items()}
        return output
