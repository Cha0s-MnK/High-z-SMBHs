#!/usr/bin/env python3
# Licensed under BSD-3-Clause License - see LICENSE

"""
Validate the IMBH fitting model against Eq. (7) and Fig. 8 of the paper.

python3 /home/subonan/Gao+2023/my/imbh_validate.py --output /home/subonan/Gao+2023/plots
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from IMBH import IMBHModel, IMBHModelConfig


STD_DPI = 300
RNG_SEED = 20260311

FH_VALUES = [0.125, 0.184, 0.269, 0.395, 0.580]

FH_LOGMASS_RANGES = {
    0.125: (4.95, 6.45),
    0.184: (4.98, 6.48),
    0.269: (5.00, 6.55),
    0.395: (5.60, 7.10),
    0.580: (5.95, 7.40),
}


def apply_plot_style() -> None:
    """Apply the local Gao+2023 plotting style with a TeX fallback."""

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
                "text.latex.preamble": r"\usepackage{amsmath}",
            }
        )
    else:
        plt.rcParams.update({"text.usetex": False})


def build_models_by_fh(fh_values: list[float]) -> dict[float, IMBHModel]:
    """Create one IMBH model per Eq. (7) normalization."""

    models = {}
    for fh in fh_values:
        cfg = IMBHModelConfig(enabled=True, metallicity_kind="z_ratio", fh=fh)
        models[fh] = IMBHModel(cfg)
    return models


def generate_demo_gcs(models_by_fh: dict[float, IMBHModel]) -> list[dict[str, float]]:
    """Build GC seed examples directly from Eq. (7) for the five f_h values."""

    rng = np.random.default_rng(RNG_SEED)
    seeds = []
    for fh in FH_VALUES:
        model = models_by_fh[fh]
        log_mass_min, log_mass_max = FH_LOGMASS_RANGES[fh]
        log_mass_edges = np.linspace(log_mass_min, log_mass_max, 11)
        accepted_masses = []
        for seed_index, (left, right) in enumerate(zip(log_mass_edges[:-1], log_mass_edges[1:]), start=1):
            accepted = False
            for _ in range(400):
                log_mass_msun = rng.uniform(left, right)
                mass_msun = 10.0 ** log_mass_msun
                if any(abs(np.log10(mass_msun / prev_mass)) < 0.025 for prev_mass in accepted_masses):
                    continue
                r_h_pc = float(model.radius_eq7(mass_msun))
                for _ in range(120):
                    z_ratio = 10.0 ** rng.uniform(-2.0, 0.0)
                    result = model.estimate_for_gc(mass_msun, z_ratio)
                    if result["imbh_mass_msun"] > 0.0:
                        accepted_masses.append(mass_msun)
                        seeds.append(
                            {
                                "name": f"fh{fh:.3f}_seed{seed_index}",
                                "fh": fh,
                                "seed_index": seed_index,
                                "mass_msun": mass_msun,
                                "r_h_pc": r_h_pc,
                                "z_ratio": z_ratio,
                            }
                        )
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                raise RuntimeError(f"Failed to generate an IMBH-forming seed for f_h={fh:.3f}")
    return seeds


def save_demo_csv(models_by_fh: dict[float, IMBHModel], demo_gcs: list[dict[str, float]], output_dir: Path) -> Path:
    """Run a few GC demo seeds and save the results as CSV."""

    output_path = output_dir / "imbh_demo_gc_examples.csv"
    fieldnames = [
        "name",
        "fh",
        "seed_index",
        "mass_msun",
        "r_h_pc",
        "r_proj_h_pc",
        "z_ratio",
        "sigma_h_msun_pc2",
        "imbh_mass_msun",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for case in demo_gcs:
            model = models_by_fh[case["fh"]]
            result = model.estimate_for_gc(
                case["mass_msun"],
                case["z_ratio"],
            )
            writer.writerow(
                {
                    "name": case["name"],
                    "fh": case["fh"],
                    "seed_index": case["seed_index"],
                    "mass_msun": case["mass_msun"],
                    "r_h_pc": case["r_h_pc"],
                    "r_proj_h_pc": model.projected_half_mass_radius_plummer(case["r_h_pc"]),
                    "z_ratio": case["z_ratio"],
                    "sigma_h_msun_pc2": result["sigma_h_msun_pc2"],
                    "imbh_mass_msun": result["imbh_mass_msun"],
                }
            )
            print(
                f"{case['name']}: f_h={case['fh']:.3f}, M={case['mass_msun']:.2e} Msun, "
                f"r_h={case['r_h_pc']:.2f} pc, Z/Zsun={case['z_ratio']:.2f}, "
                f"Sigma_h={result['sigma_h_msun_pc2']:.2e}, "
                f"IMBH={result['imbh_mass_msun']:.1f} Msun"
            )
    return output_path


def make_eq7_plot(models_by_fh: dict[float, IMBHModel], demo_gcs: list[dict[str, float]], output_dir: Path) -> Path:
    """Plot the Eq. (7) GC mass-size relation with generated GC seeds only."""

    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(FH_VALUES), vmax=max(FH_VALUES))
    masses = np.array([case["mass_msun"] for case in demo_gcs], dtype=float)
    radii = np.array([case["r_h_pc"] for case in demo_gcs], dtype=float)
    x_min = masses.min() / 1.35
    x_max = masses.max() * 1.35
    y_min = radii.min() / 1.35
    y_max = radii.max() * 1.35
    mass_grid = np.logspace(np.log10(x_min), np.log10(x_max), 500)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(7.2, 4.6), dpi=STD_DPI)
    for fh in FH_VALUES:
        color = cmap(norm(fh))
        model = models_by_fh[fh]
        ax.plot(
            mass_grid,
            model.radius_eq7(mass_grid),
            color=color,
            lw=2.0,
            alpha=0.95,
        )
    for case in demo_gcs:
        ax.scatter(
            case["mass_msun"],
            case["r_h_pc"],
            marker="o",
            s=48,
            facecolor=cmap(norm(case["fh"])),
            edgecolor="black",
            linewidth=0.45,
            zorder=6,
        )

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.5, label="Eq. 7 curves"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=6,
            lw=0,
            label="Generated GC seeds",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Cluster mass $M_{cl}\ [{\rm M_{\odot}}]$")
    ax.set_ylabel(r"3D half-mass radius $r_h\ [{\rm pc}]$")
    ax.set_title(r"Eq. 7 Mass-Size Relation Check")
    ax.text(
        0.98,
        0.03,
        "Solid lines: Eq. 7 implementation\nColored circles: generated GC seeds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.015)
    colorbar.set_label(r"$f_h$")
    colorbar.set_ticks(FH_VALUES)
    colorbar.set_ticklabels([f"{value:.3f}" for value in FH_VALUES])

    output_path = output_dir / "eq7_gc_mass_size_relation.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def make_fig8_like_plot(model: IMBHModel, demo_gcs: list[dict[str, float]], output_dir: Path) -> Path:
    """Plot the parametrized IMBH fit in the style of Fig. 8."""

    log_sigma = np.linspace(np.log10(4.0e3), np.log10(3.2e6), 500)
    log_z_ratio = np.linspace(-2.0, 0.0, 400)
    sigma = 10.0 ** log_sigma
    z_ratio = 10.0 ** log_z_ratio
    sigma_grid, z_grid = np.meshgrid(sigma, z_ratio)
    imbh_mass = model.imbh_mass_from_sigma_metallicity(sigma_grid, z_grid)
    mass_masked = np.ma.masked_less(imbh_mass, model.config.min_imbh_mass_msun)

    mass_cmap = plt.cm.magma.copy()
    mass_cmap.set_bad(color="white")
    mass_norm = LogNorm(vmin=100.0, vmax=4.0e3)

    fig = plt.figure(constrained_layout=True, figsize=(7.8, 4.1), dpi=STD_DPI)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.05])
    ax = fig.add_subplot(gs[0, 0])
    cax_mass = fig.add_subplot(gs[0, 1])
    mesh = ax.pcolormesh(
        log_sigma,
        log_z_ratio,
        mass_masked,
        shading="auto",
        cmap=mass_cmap,
        norm=mass_norm,
    )
    contours = ax.contour(
        log_sigma,
        log_z_ratio,
        np.where(imbh_mass > 0.0, imbh_mass, np.nan),
        levels=[100.0, 300.0, 1000.0, 3000.0],
        colors="white",
        linewidths=[1.3, 1.0, 1.3, 1.0],
    )
    ax.clabel(contours, inline=True, fmt=r"%d $M_{\odot}$", fontsize=8)
    ax.axvline(
        model.config.lg_Sigma_max,
        color="white",
        ls="--",
        lw=1.2,
        alpha=0.8,
    )
    ax.text(
        model.config.lg_Sigma_max + 0.03,
        -1.98,
        "Eq. 9 / Eq. 10 splice",
        color="white",
        fontsize=8,
        rotation=90,
        va="bottom",
        ha="left",
    )

    gc_estimates = [model.estimate_for_gc(case["mass_msun"], case["z_ratio"]) for case in demo_gcs]
    gc_sigma = np.array([estimate["sigma_h_msun_pc2"] for estimate in gc_estimates], dtype=float)
    gc_z = np.array([case["z_ratio"] for case in demo_gcs], dtype=float)
    gc_imbh_mass = np.array([estimate["imbh_mass_msun"] for estimate in gc_estimates], dtype=float)
    ax.scatter(
        np.log10(gc_sigma),
        np.log10(gc_z),
        c=gc_imbh_mass,
        cmap=mass_cmap,
        norm=mass_norm,
        s=36,
        edgecolors="black",
        linewidths=0.6,
        zorder=7,
    )

    ax.set_xlim(4.5, 7.0)
    ax.set_ylim(log_z_ratio.min(), log_z_ratio.max())
    ax.set_xticks([4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
    ax.set_yticks([-2.0, -1.5, -1.0, -0.5, 0.0])
    ax.set_xlabel(r"$\log_{10}\left(\Sigma_h / {\rm M_{\odot}\,pc^{-2}}\right)$")
    ax.set_ylabel(r"$\log_{10}(Z/Z_{\odot})$")
    ax.set_title(r"Fig. 8-Like Parametrized IMBH Fit")

    cbar = fig.colorbar(mesh, cax=cax_mass)
    cbar.set_label(r"IMBH mass $M_{\bullet}\ [{\rm M_{\odot}}]$")

    output_path = output_dir / "fig8_like_imbh_param_model.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "plots",
        help="Directory that will receive the plots and demo CSV.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the validation workflow."""

    args = parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    models_by_fh = build_models_by_fh(FH_VALUES)
    demo_gcs = generate_demo_gcs(models_by_fh)
    base_model = IMBHModel(IMBHModelConfig(enabled=True, metallicity_kind="z_ratio"))
    csv_path = save_demo_csv(models_by_fh, demo_gcs, output_dir)
    eq7_plot = make_eq7_plot(models_by_fh, demo_gcs, output_dir)
    fig8_plot = make_fig8_like_plot(base_model, demo_gcs, output_dir)

    print(f"Saved demo CSV: {csv_path}")
    print(f"Saved Eq. 7 plot: {eq7_plot}")
    print(f"Saved Fig. 8-like plot: {fig8_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
