# High-z SMBHs

This repository is the current working branch derived from `/home/subonan/Gao+2023`. It extends the Gao+2023 globular cluster (GC) model toward a "GC to IMBH to high-$z$ SMBHs" workflow and now provides the active Python implementation for GC formation, GC evolution, IMBH seeding, batch execution, and figure reproduction.

## New Features

### External Illustris-Dark tree workflow

Compared with the original Gao+2023 repository, this project now has an explicit external tree pipeline for Milky Way (MW) and M31-like targets in Illustris-Dark. The maintained workflow covers target selection, full-tree download, conversion from raw SubLink trees to Gao-compatible fixed-tree files, and validation before any GC physics is run. In practice, this means the model no longer depends on one hard-wired bundled tree sample: the runner can ingest alternative corrected tree directories through `--tree-dir`.

### Python GC-evolution workflow

The original Python-plus-Fortran split has been replaced by an active Python evolution path centered on `src/evoGC_fast.py`. Relative to `/home/subonan/Gao+2023`, the current workflow exposes the background model, timestep controls, final redshift, and Sersic-index scan directly through `my/run.py`, while keeping the formation stage tied to the Gao-style tree and GC catalog logic. The current pipeline is also easier to inspect and compare because one command now rebuilds formation catalogs, runs halo-by-halo evolution, merges outputs, and can optionally switch to the analysis-oriented `--analy 1` mode.

### IMBH extension

The main scientific extension beyond Gao+2023 is the IMBH path. `src/IMBH.py` adds formation-time IMBH seeding tied to GC structural properties, and the formation catalogs now store GC radius, surface density, metallicity, and IMBH seed mass for downstream use. Halo-level summaries also track SMBH-proxy quantities from sunk GC and IMBH channels. This is still a first bridge from GC evolution to SMBH-oriented diagnostics rather than a full black-hole growth model with accretion and merger physics.

### Improved outputs and analysis support

The output layout is now organized by `N_s`, with merged `finalGCs` and `depos` products, halo-summary tables, machine-readable run metadata, and direct figure reproduction through `my/plot.py`. The repository also includes `my/imbh_validate.py` for checking the IMBH-side calibration. Compared with the original Gao layout, the emphasis here is on a cleaner batch workflow and outputs that are easier to audit, compare, and reuse in later SMBH-focused analysis.

## Repository Layout

- `data/`: reference tables used by the model, plus the bundled fixed-tree sample. External corrected tree directories can also be supplied at runtime through `--tree-dir`.
- `data/fixed_trees_large_spin/`: bundled Gao-compatible fixed-tree input set.
- `src/main_spatial.py`: GC formation stage based on the Gao/Choksi-style model.
- `src/evoGC_fast.py`: active Python GC evolution solver.
- `src/IMBH.py`: IMBH seeding module used at GC formation.
- `src/schechter_interp.py`: Schechter-sampling support for GC initial masses.
- `src/smhm.py`: stellar-mass-halo-mass helper functions.
- `my/run.py`: end-to-end batch runner for formation, evolution, merging, and optional plotting.
- `my/plot.py`: figure reproduction script for the current output layout.
- `my/imbh_validate.py`: validation helper for the IMBH parametrized model.
- `papers/`: method papers and reference PDFs used for the project.
- `plots/`: project figures and plotting artifacts kept in the repository.
- `tex/`: manuscript and note material.

## Typical Run

```bash
cd "/home/subonan/High-z SMBHs"
python3 my/run.py \
  --gao-root "$PWD" \
  --tree-dir /lingshan/disk3/subonan/Illustris-Dark/data/fixed_trees_large_spin_dark \
  --output /lingshan/disk3/subonan/_outputs/High-z-SMBHs_MW64_analy \
  --clear-output \
  --p2 6.75 \
  --p3 0.5 \
  --mpb-only 0 \
  --mc 12.0 \
  --run-all 0 \
  --n-halos 64 \
  --log-mh-min 11.8865 \
  --log-mh-max 12.0969 \
  --exclude_halo 55,65,86 \
  --bgsw 1 \
  --ts-m 0.5 \
  --ts-r 0.5 \
  --analy 1 \
  --final-z 0.0 \
  --IMBH 1 \
  --ns-values 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0 \
  --jobs 4 \
  --ns-jobs 4 \
  --plot
```

Prefer running from the repository root because the project path contains spaces and the relative `my/run.py` entry point is the least error-prone form.

- `--run-all 1` processes the full tree set, while `--run-all 0` activates the mass window and `--n-halos` selection.
- `--jobs` controls halo-level parallelism inside one `N_s` run, and `--ns-jobs` controls how many `N_s` pipelines are run concurrently.
- `--exclude_halo` removes specific `z=0` host halo IDs before the mass filter and halo count are applied.
- `--plot` writes figure outputs under `<output>/_plots/`.
- Temporary work directories are created under the system temp area and removed automatically at the end of the run.
- Each Sersic index writes to its own `ns*/` directory, while merged products stay at the output top level.
- The main merged outputs are `finalGCs_all.dat`, `depos_all.dat`, `haloSummary_all.csv`, `python_evo_summary.csv`, and `run_metadata.json`.

## Main Run Parameters

The active workflow no longer uses the legacy Gao `input.txt` interface. The main controls now live in `my/run.py`.

### Path and output control

- `--gao-root`: repository root used to locate `src/` and bundled `data/`.
- `--tree-dir`: optional fixed-tree input directory; if omitted, the runner uses `<gao-root>/data/fixed_trees_large_spin`.
- `--output`: output directory for the whole run.
- `--clear-output`: remove existing files in the output directory before writing fresh results.

### Formation-model parameters

- `--p2`: GC formation-efficiency normalization in `M_GC = 3e-5 * p2 * M_gas / f_b`.
- `--p3`: threshold in `((Delta M_h / M_h) / Delta t)` above which a formation event is triggered.
- `--mpb-only`: if `1`, form GCs only on the main progenitor branch; if `0`, include all retained branches in the fixed tree.
- `--mc`: `log10(M_c / Msun)` for the Schechter cutoff mass in the GC initial-mass function.
- `--run-all`: if `1`, process all halos in the selected tree directory.
- `--log-mh-min`: lower bound on retained host-halo `log10(M_h)` when `--run-all 0`.
- `--log-mh-max`: upper bound on retained host-halo `log10(M_h)` when `--run-all 0`.
- `--n-halos`: maximum number of halos to keep when `--run-all 0`.
- `--exclude_halo`: comma-separated `z=0` halo IDs to skip before filtering and counting.

### Evolution and scan parameters

- `--bgsw`: background model in `src/evoGC_fast.py`; `1` uses the evolving host halo, `0` uses a fixed MW-like Sersic background, and `-1` uses a fixed MW-like bulge+disk+DM background.
- `--ts-m`: adaptive mass-loss timestep factor.
- `--ts-r`: adaptive orbital-decay timestep factor.
- `--analy`: if `1`, use analytic background-density evaluation instead of the cached lookup-table mode.
- `--final-z`: final redshift where the formation survivor cut and orbit evolution stop.
- `--IMBH`: if `1`, enable IMBH seeding in `src/main_spatial.py`; if `0`, write zero IMBH-related columns.
- `--ns-values`: comma-separated list of Sersic indices to run.
- `--jobs`: parallel halo-evolution workers per `N_s`.
- `--ns-jobs`: concurrent `N_s` pipelines.
- `--plot`: run `my/plot.py` automatically after the simulation.
- `--quiet`: reduce progress logging.

### Internal `evoGC_fast.py` tunables

These are not exposed as `my/run.py` flags, but they still define the evolution grid and deposited-mass bookkeeping:

- `T_UNIVERSE_GYR = 13.799`: Universe-age constant used by the approximate cosmic-time and redshift conversions.
- `dt_max = 0.1` and `t_div = 100`: cap the adaptive step size and define the coarse cosmic-time blocks.
- `binnub = 100`, `r_min = 1.0e-3`, and `r_sink = 1.0e-3`: set the deposited-profile radial binning and the sink radius.
- `t_limit = 1.0e-2` and `mdot_iso_mw = 2/17`: set the minimum adaptive timescale floor and the fixed-MW lower bound on tidal mass loss.

## Figure Reproduction

```bash
cd "/home/subonan/High-z SMBHs"
python3 my/plot.py \
  --allcat /lingshan/disk3/subonan/_outputs/High-z-SMBHs_MW64_analy/allcat_s-0_p2-6.75_p3-0.5.txt \
  --mpb /lingshan/disk3/subonan/_outputs/High-z-SMBHs_MW64_analy/mpb_from_fixed_trees.csv \
  --ns-values 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0 \
  --output /lingshan/disk3/subonan/_outputs/High-z-SMBHs_MW64_analy/_plots \
  --gao-fig2-dir /lingshan/disk3/subonan/_outputs/Gao+2023Mod_origMW64
```

- `--allcat`: root allcat template path, or one per-`N_s` allcat file from which the script resolves the other `ns*/` tables.
- `--mpb`: path to `mpb_from_fixed_trees.csv`.
- `--ns-values`: set of Sersic indices to include in the figure suite.
- `--output`: destination directory for figure PNGs and the manifest.
- `--gao-fig2-dir`: optional Gao+2023 merged-output directory used for the Figure 2 comparison overlay.
- `--no-observables`: optional switch to disable observational overlays.

The plotting script writes the figure PNGs plus `_plots/figure_manifest.csv`.

## Output Schema

The output directory has two persistent layers:

- top level: merged summaries shared across all `N_s`
- `ns*/`: per-Sersic-index outputs such as `ns0p5/`, `ns1p0/`, and `ns4p0/`

Temporary work directories are transient, created under the system temp area, and removed automatically. They are not part of the published output schema.

### Top-level outputs

#### `allcat_s-0_p2-..._p3-....txt`

Convenience copy of one per-`N_s` formation catalog and the historical single-file entry point for `my/plot.py`. Each row is one formed GC, and the row order matches the corresponding `finalGCs_ns*.dat` tables.

Columns:
- `hid_z0`, `logMh_z0`, `logMstar_z0`
- `logMh_form`, `logMstar_form`, `logM_form`
- `zform`, `feh`, `isMPB`
- `subfind_form`, `snap_form`
- `r_galaxy_kpc`, `gc_radius_pc`, `sigma_h_msun_pc2`, `imbh_mass_msun`

#### `mpb_from_fixed_trees.csv`

Compact main-progenitor-branch table rebuilt from the selected fixed-tree directory. It is used mainly by `my/plot.py` for halo-history diagnostics such as `z_hm`.

Columns:
- `subhalo_id_z0`
- `SnapNum`
- `Redshift`
- `logMh_msun_h`
- `SubhaloSpin_x`, `SubhaloSpin_y`, `SubhaloSpin_z`

#### `python_evo_summary.csv`

Compact per-GC summary across all `N_s` values, useful for quick QA without rereading the merged `finalGCs` tables.

Columns:
- `ns`
- `hid_z0`
- `status`
- `m_final_msun`
- `r_final_kpc`

Status codes:
- `1`: alive at the final redshift
- `-1`: exhausted to zero mass
- `-2`: tidally torn apart
- `-3`: sunk into the galaxy center
- `-4`: surviving IMBH wanderer at the final redshift
- `-5`: IMBH wanderer sunk into the galaxy center

#### `finalGCs_all.dat`

Merged final-GC table across all `N_s` runs. Each row corresponds to one GC from one halo and one Sersic-index run.

Columns:
- `ns`
- `halo_id_z0`
- `gc_index_halo`
- `status`
- `m_final_msun`
- `log10_m_final_msun`
- `m_init_msun`
- `lookback_time_final_gyr`
- `lookback_time_init_gyr`
- `r_final_kpc`
- `r_init_kpc`
- `gc_radius_pc`
- `sigma_h_msun_pc2`
- `feh`
- `imbh_mass_msun`

#### `depos_all.dat`

Merged deposited-mass profile table across all `N_s` runs.

Columns:
- `ns`
- `halo_id_z0`
- `lookback_time_gyr`
- `bin_index`
- `r_inner_kpc`
- `r_outer_kpc`
- `m_depo_total_msun`
- `m_star_no_evo_msun`
- `m_star_with_evo_msun`

#### `haloSummary_all.csv`

Halo-level summary across all `N_s` runs, including status counts, total GC masses, and SMBH-proxy quantities built from sunk GC and IMBH channels.

Columns:
- `hid_z0`
- `logMh_z0`
- `n_gc_total`
- `n_alive`
- `n_wanderer`
- `n_exhausted`
- `n_torn`
- `n_sunk_gc`
- `n_sunk_wanderer`
- `n_sunk`
- `m_gc_init_total_msun`
- `m_gc_final_total_msun`
- `m_imbh_seed_total_msun`
- `m_smbh_gc_sunk_msun`
- `m_smbh_wanderer_sunk_msun`
- `m_smbh_est_msun`
- `ns`

#### `run_metadata.json`

Machine-readable record of the main run configuration used to build the output directory.

Keys surfaced in the README:
- `final_redshift`
- `bgsw`
- `ts_m`
- `ts_r`
- `analy`
- `p2`
- `p3`
- `mc`
- `IMBH`
- `mpb_only`
- `run_all`
- `log_mh_min`
- `log_mh_max`
- `n_halos`
- `exclude_halo`
- `ns_values`

### Per-`N_s` outputs

Each `N_s` writes to its own directory such as `ns0p5/`, `ns1p0/`, `ns1p5/`, `ns2p0/`, `ns2p5/`, `ns3p0/`, `ns3p5/`, and `ns4p0/`.

#### `allcat_nsXpY_s-0_p2-..._p3-....txt`

Formation catalog for one `N_s`. It uses the same columns as the top-level `allcat_s-...txt` file.

#### `finalGCs_nsXpY.dat`

Published final-GC table for one `N_s`. It uses the same columns as `finalGCs_all.dat` except for the leading `ns` column.

#### `depos_nsXpY.dat`

Published deposited-mass profile table for one `N_s`. It uses the same columns as `depos_all.dat` except for the leading `ns` column.

#### `haloSummary_nsXpY.csv`

Halo-level summary for one `N_s`. It uses the same columns as `haloSummary_all.csv` except for the trailing `ns` column.

### Plot outputs

When `my/plot.py` is run, it writes:

- `_plots/Fig.XX_*.png`: one rendered PNG per reproduced figure.
- `_plots/figure_manifest.csv`: one row per generated figure with `figure`, `path`, and `observables` columns.
