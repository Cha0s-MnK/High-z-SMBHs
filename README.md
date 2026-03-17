# Gao+2023

Python-only workflow for the Gao et al. (2023) globular-cluster model in this
repository. The active workflow uses only raw files under `data/` as simulation inputs.

## Active layout

- `data/fixed_trees_large_spin/`: input halo trees
- `data/mass_loss.txt`: stellar-evolution mass-loss table
- `data/snaps2redshifts.txt`: snapshot to redshift table
- `src/main_spatial.py`: GC formation stage
- `src/evoGC_fast.py`: active standalone Python evolution module
- `src/schechter_interp.py`, `src/smhm.py`: support modules used by the model
- `my/run.py`: end-to-end batch runner
- `my/plot.py`: figure reproduction script
- `src/old/`: parked legacy and reference source files, including the old `evoGC.py`

## Run the full 248-halo model

This command rebuilds the outputs from scratch for the full `N_s` list:

```bash
nohup python3 /home/subonan/Gao+2023/my/run.py \
  --gao-root /home/subonan/Gao+2023 \
  --output /lingshan/disk3/subonan/_outputs/Gao+2023_new --clear-output \
  --run-all 1 \
  --final-z 0.0 --ns-values 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0 \
  --jobs 8 --ns-jobs 8 \
  --plot > /home/subonan/Gao+2023/my/run.log 2>&1 &
python3 /home/subonan/Gao+2023/my/run.py \
  --gao-root /home/subonan/Gao+2023 \
  --tree-dir /lingshan/disk3/subonan/Illustris-Dark/data/fixed_trees_large_spin_dark \
  --output /lingshan/disk3/subonan/_outputs/Gao+2023_MW512 --clear-output \
  --run-all 0 --n-halos 512 --log-mh-min 11.88 --log-mh-max 12.1 \
  --final-z 0.0 --ns-values 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0 \
  --jobs 4 --ns-jobs 8 \
  --plot
python3 /home/subonan/Gao+2023/my/run.py \
  --gao-root /home/subonan/Gao+2023 \
  --output /lingshan/disk3/subonan/_outputs/Gao+2023_MW+M31 --clear-output \
  --run-all 0 --n-halos 32 --log-mh-min 11.84 --log-mh-max 12.4 \
  --final-z 7.5 --ns-values 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0 \
  --jobs 2 --ns-jobs 8
```

Notes:

- `--run-all 1` selects all 248 halos available in the raw tree set.
- `--jobs` controls parallel halo evolution per `N_s` run. Adjust it to your
  machine.
- `--ns-jobs` controls how many `N_s` values are processed concurrently.
- `--plot` runs `my/plot.py` automatically after the model finishes and writes
  figures to `<output>/_plots/`.
- Temporary staging directories are created under the system temp area and
  always removed automatically at the end of the run.
- Each `N_s` now writes to its own subdirectory such as `ns0p5/`, `ns1p0/`,
  ..., `ns4p0/`. Shared summary files stay at the top level.
- Published GC-evolution tables are now merged-only:
  `ns*/finalGCs_ns*.dat`, `ns*/depos_ns*.dat`, plus top-level
  `finalGCs_all.dat` and `depos_all.dat`.
- `finalGCs_ns*.dat` now subsumes the old `logm` and `rgal` outputs.
- `haloSummary_ns*.csv` and `haloSummary_all.csv` now store halo-level status counts
  and the SMBH estimate from sunk GC IMBH seeds.
- Outputs are written to `/lingshan/disk3/subonan/_outputs/Gao+2023`.

## Run Parameters

The active workflow no longer uses the legacy `input.txt` file. The old
`input.txt` columns are now controlled directly with `my/run.py` flags:

- `--p2`: GC formation-efficiency normalization. In `src/main_spatial.py` the
  total GC mass formed in one event is `M_GC = 3e-5 * p2 * M_gas / f_b`.
- `--p3`: halo growth-rate threshold for triggering a GC formation event. A
  branch node forms GCs only when `((Delta M_h / M_h) / Delta t) > p3`.
- `--mpb-only`: if `1`, only the main progenitor branch is searched for GC
  formation events; if `0`, all retained branches in the fixed tree are used.
- `--mc`: `log10(M_c / Msun)` for the Schechter cutoff mass in the GC initial
  mass function.
- `--run-all`: if `1`, process every halo in `data/fixed_trees_large_spin/`;
  if `0`, use the mass window and halo count flags below.
- `--log-mh-min`: lower bound on the retained host-halo log mass used for halo
  selection when `--run-all 0`.
- `--log-mh-max`: upper bound on the retained host-halo log mass used for halo
  selection when `--run-all 0`.
- `--n-halos`: maximum number of halos to keep when `--run-all 0`.
- `--final-z`: redshift where both the formation-stage survivor cut and the
  orbit-evolution stage stop. `0.0` means the present day.

Other important Python-controlled runtime parameters are:

- `--bgsw`: background model for `src/evoGC_fast.py`. `1` uses the evolving
  host halo, `0` uses a fixed MW-like Sersic background, and `-1` uses a fixed
  MW-like bulge+disk+DM background.
- `--ts-m`: adaptive mass-loss timestep factor in `src/evoGC_fast.py`.
- `--ts-r`: adaptive orbital-decay timestep factor in `src/evoGC_fast.py`.
- `--ns-values`: comma-separated list of Sersic indices to run.
- `--jobs`: number of parallel halo-evolution workers inside one `N_s` run.
- `--ns-jobs`: number of `N_s` runs processed concurrently.
- `--plot`: automatically run `my/plot.py` after the simulation and write the
  figures under the run output directory at `_plots/`.
- `--clear-output`: delete existing files in the output directory before
  writing new results.

### Internal `evoGC_fast.py` Tunables

`src/evoGC_fast.py` also has a small internal `Tunables` dataclass. These are
not exposed as `my/run.py` flags right now, but they control the time grid and
the deposited-mass bookkeeping inside the orbit solver:

- `T_UNIVERSE_GYR = 13.799`: fixed solver constant for the age of the Universe
  in Gyr. The legacy approximation `t(z) = T_UNIVERSE_GYR / (1 + z)^{1.5}` and
  its inverse now use this module constant directly instead of a `Tunables`
  field.

- `dt_max = 0.1`: hard upper limit on one adaptive GC step in Gyr. Even if the
  mass-loss and orbital-decay estimates allow a larger step, the solver clips
  the step to this value.
- `t_div = 100`: number of coarse cosmic-time blocks between the Big Bang and
  the chosen final epoch. The background-density lookup is rebuilt once per
  block, and deposited profiles are written once per block.
- `binnub = 100`: number of logarithmic radial bins used for deposited-mass
  profiles. These bins run from `r_min` up to the largest initial GC radius in
  the halo.
- `t_limit = 1.0e-2`: minimum base timescale in Gyr used when the adaptive
  timestep floors from `ts_m` and `ts_r` would otherwise become too small.
- `r_sink = 1.0e-3`: sink radius in kpc. A GC that reaches `r <= r_sink` is
  marked as `sunk_to_center`.
- `h = 0.704`: dimensionless Hubble parameter used in the virial-radius and
  halo-spin conversions in the evolving-background mode.
- `mdot_iso_mw = 2/17`: lower bound on the tidal-disruption mass-loss rate,
  applied only in fixed MW-like Sersic mode (`--bgsw 0`). Its internal unit is
  `1e5 Msun / Gyr`.
- `r_min = 1.0e-3`: inner radial floor in kpc for deposit bins and for the
  cached background-density lookup grid. This avoids `log10(r)` problems at
  the origin and keeps the innermost deposit bin finite.

## Reproduce the figures

After the simulation finishes, run:

```bash
python /home/subonan/Gao+2023/my/plot.py \
  --allcat /lingshan/disk3/subonan/_outputs/Gao+2023_MW/allcat_s-0_p2-6.75_p3-0.5.txt \
  --mpb /lingshan/disk3/subonan/_outputs/Gao+2023_MW/mpb_from_fixed_trees.csv \
  --ns-values 0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0 \
  --output /lingshan/disk3/subonan/_outputs/Gao+2023_MW/_plots
```

The script writes the current 10-figure subset plus
`_plots/figure_manifest.csv`.

## Output schema

The output directory has two layers:

- top level: shared summaries across all `N_s`
- `ns*/`: per-Sersic-index outputs such as `ns0p5/`, `ns1p0/`, ..., `ns4p0/`

### Top-level outputs

#### `allcat_s-0_p2-..._p3-....txt`

Meaning:
- convenience copy of one per-`N_s` allcat table
- serves as the historical single-file entry point for `my/plot.py`

Rows:
- one formed GC per row
- row order defines the alignment for the matching `finalGCs_ns*.dat` files

Columns:
- `hid_z0`: halo ID at `z=0`
- `logMh_z0`: `log10` halo mass at `z=0`
- `logMstar_z0`: `log10` stellar mass at `z=0` from the SMHM relation
- `logMh_form`: `log10` halo mass at the GC formation event
- `logMstar_form`: `log10` stellar mass at the GC formation event
- `logM_form`: `log10` initial GC mass at formation
- `zform`: GC formation redshift
- `feh`: GC metallicity `[Fe/H]`
- `isMPB`: `1` if the formation subhalo is on the main progenitor branch, else `0`
- `subfind_form`: formation subhalo ID
- `snap_form`: nearest snapshot number to `zform`
- `r_galaxy_kpc`: assigned initial galactocentric radius in kpc
- `gc_radius_pc`: formation-time GC half-mass radius used by the IMBH model
- `sigma_h_msun_pc2`: formation-time GC half-mass surface density used by the IMBH model
- `imbh_mass_msun`: IMBH seed mass assigned at GC formation

#### `mpb_from_fixed_trees.csv`

Meaning:
- compact MPB table rebuilt from `data/fixed_trees_large_spin`
- used by `my/plot.py` to estimate `z_hm` and other halo-history quantities

Rows:
- one halo-history row per `z=0` halo and snapshot on its main branch

Columns:
- `subhalo_id_z0`
- `SnapNum`
- `Redshift`
- `logMh_msun_h`
- `SubhaloSpin_x`
- `SubhaloSpin_y`
- `SubhaloSpin_z`

#### `python_evo_summary.csv`

Meaning:
- compact per-GC summary across all `N_s` values
- useful for quick QA without rereading the merged `finalGCs` tables

Rows:
- one GC row per `N_s`

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

#### `finalGCs_all.dat`

Meaning:
- merged final-GC table across all `N_s` runs

Rows:
- one GC row from one halo and one `N_s`
- grouped by `N_s`, then halo, then halo-local GC order

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

Meaning:
- merged deposited-mass table across all `N_s` runs

Rows:
- one deposited radial-bin row from one halo and one `N_s`

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

Meaning:
- halo-level summary across all `N_s` runs
- includes both the total formed IMBH seed mass and the sunk-BH sum used as the SMBH estimate

Rows:
- one halo row per `N_s`

Columns:
- `hid_z0`
- `logMh_z0`
- `n_gc_total`
- `n_alive`
- `n_exhausted`
- `n_torn`
- `n_sunk`
- `m_gc_init_total_msun`
- `m_gc_final_total_msun`
- `m_imbh_seed_total_msun`
- `m_smbh_est_msun`
- `ns`

#### `run_metadata.json`

Meaning:
- machine-readable record of the run configuration used to build the output directory

Keys:
- `final_redshift`
- `bgsw`
- `ts_m`
- `ts_r`
- `p2`
- `p3`
- `mc`
- `mpb_only`
- `run_all`
- `log_mh_min`
- `log_mh_max`
- `n_halos`
- `ns_values`

### Per-`N_s` outputs

Each `N_s` writes to its own directory:

- `ns0p5/`
- `ns1p0/`
- `ns1p5/`
- `ns2p0/`
- `ns2p5/`
- `ns3p0/`
- `ns3p5/`
- `ns4p0/`

#### `allcat_nsXpY_s-0_p2-..._p3-....txt`

Meaning:
- formation catalog for one `N_s`
- main per-`N_s` GC catalog used by the plotting script

Rows:
- one formed GC per row

Columns:
- same as the top-level `allcat_s-...txt`

#### `finalGCs_nsXpY.dat`

Meaning:
- published final-GC table for one `N_s`
- replaces the old separate `logm` and `rgal` outputs

Rows:
- one GC row per `allcat_ns...` row
- row order matches the corresponding `allcat_ns...txt` exactly

Columns:
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

#### `depos_nsXpY.dat`

Meaning:
- published deposited-mass profile table for one `N_s`
- combines all halos for that `N_s`

Rows:
- one deposited radial-bin row from one halo at that `N_s`

Columns:
- `halo_id_z0`
- `lookback_time_gyr`
- `bin_index`
- `r_inner_kpc`
- `r_outer_kpc`
- `m_depo_total_msun`
- `m_star_no_evo_msun`
- `m_star_with_evo_msun`

#### `haloSummary_nsXpY.csv`

Meaning:
- halo-level summary for one `N_s`

Rows:
- one halo row for that `N_s`

Columns:
- `hid_z0`
- `logMh_z0`
- `n_gc_total`
- `n_alive`
- `n_exhausted`
- `n_torn`
- `n_sunk`
- `m_gc_init_total_msun`
- `m_gc_final_total_msun`
- `m_imbh_seed_total_msun`
- `m_smbh_est_msun`

### Temporary directories

By default, temporary work directories are created under the system temp area and removed automatically.

If `--keep-temp` is enabled, two extra directories are kept under the output root:

- `_tmp_main_spatial/`: transient `main_spatial.py` outputs and logs such as `all_<Ns>.txt`, `z0_cat_<Ns>.txt`, and `main_spatial_ns*.log`
- `_tmp_gcini/`: per-halo temporary evolution inputs and intermediate halo-local outputs used before merging into `finalGCs_ns*.dat` and `depos_ns*.dat`

These temp directories no longer contain copied or linked copies of the raw input data.

### Plot outputs

When `my/plot.py` is run, it writes:

#### `_plots/Fig.XX_*.png`

Meaning:
- one rendered PNG per reproduced figure

#### `_plots/figure_manifest.csv`

Rows:
- one row per generated figure

Columns:
- `figure`
- `path`
- `observables`
