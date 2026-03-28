# High-z-SMBHs compared with original Gao+2023

This section briefly introduces what is new in the *High-z SMBHs* project, compared with the original *Gao+2023* MSP project at `/home/subonan/Gao+2023/`. The part about MSP is totally removed here because the present project is no longer about GC-carried MSP $\gamma$-rays; instead it is being turned into a GC-to-IMBH-to-high-$z$-SMBH workflow.

## Data Downloading from Illustris

We build a totally new data access flow in folder `/lingshan/disk3/subonan/Illustris-Dark/`. *High-z-SMBHs* now has an explicit external pipeline to select, download, convert, and validate Illustris-1-Dark trees before they are fed into the model.

### 1_select_targets: select $z = 0$ MW/M31-like central haloes

This stage replaces the old static tree assumption with a reproducible target-selection step. It reads Illustris-1-Dark group catalog fields, selects candidate $z = 0$ central haloes in the desired Milky Way (MW) / M 31 mass range, and writes a stable target manifest.

### 2_download_full_trees: download full SubLink trees instead of relying on bundled samples

The second stage downloads the raw full SubLink merger trees for the selected target haloes. The API download itself is part of the maintained workflow, and the emphasis is on full trees rather than only MPB-like histories.

### 3_convert_full_trees_to_fixed_dat: convert raw trees into Gao-compatible corrected tree files

The third stage converts the downloaded HDF5 trees into the corrected fixed-tree format needed by the GC model. This workflow is clean and explicit: the converted trees live outside the repo, the header is made clear, and the resulting tree directory can be passed into the model through a flexible `--tree-dir` parameter.

### 4_validate_fixed_trees: validate tree structure before running GC physics

The fourth stage is a dedicated validation step. It checks that the converted trees have the expected schema, branch structure, and monotonicity properties before they are consumed by the GC model. The tree-making pipeline is part of the scientific workflow, so validation is first-class rather than implicit.

## Rewritten of the evolution of GCs

We totally rewrite the Fortran evolution of GCs files to Python.
 
### Stellar evolution

Stellar evolution is generally not changed at all physically; here we just emphasize the way we do stellar evolution again. The core mass-loss prescription is still the same age-based GC stellar-wind loss inherited from the Gao+2023 / GCevo logic, but the implementation is now in Python and coupled directly to the deposited stellar-mass bookkeeping. Another important difference from the original project is that the deposited profile now focuses on mass components relevant for NSCs and BH growth.

### Tidal disruption

Tidal disruption is generally not changed at all at the model level; here we just emphasize the way we do Tidal Disruption again. The main physical ingredients are still the same direct tidal-tear criterion and CGL18-style stripping logic, but the code path is cleaner and more configurable in *High-z-SMBHs*. In particular, the modern solver now reads configurable tree directories, uses the input `N_s` value consistently in the Sersic background instead of hardcoding one index.

### Tidal Tearing

### Dynamical friction (DF)

We do the Dynamical Friction of GCs in a more efficient way. Compared with the original Gao+2023 repository, which keeps the main orbital-evolution engine in Fortran, the new Python solver uses cached radial background-density lookup tables, RK4 orbital decay, cleaner cosmic-time/redshift helpers, and a batch wrapper in `my/run.py` that parallelizes halo evolution. The result is not a different physical idea, but a more maintainable and inspectable implementation that is easier to connect to new BH physics.

## IMBH in GCs

We hope to connect BHs in GCs to the high-z SMBHs. This whole section is new relative to the original `/home/subonan/Gao+2023/`, which had no active IMBH module in the GC formation stage and instead focused on MSP-related observables.

### IMBH formation

We follow the paper `/home/subonan/High-z-SMBHs/my/FROST-CLUSTERS – III. Metallicity-dependent IMBH formation by runaway collisions in dense star clusters.pdf` to do IMBH formation inside each GC. Concretely, *High-z-SMBHs* adds `src/IMBH.py`, validates the parameterized model with dedicated scripts and plots, and seeds an IMBH exactly once at GC birth inside `src/main_spatial.py`. The formation catalog now stores each GC’s formation-time radius, surface density, metallicity, and IMBH seed mass, which is a major extension beyond the original Gao+2023 output schema.

### IMBH evolution

Up to now we do not evolve the IMBHs formed in GCs with accretion or BH-BH merger physics. The present implementation simply keeps the IMBH mass fixed after GC formation and carries it through the GC evolution outputs. If a GC later sinks to the centre, the sunk IMBH seeds can be summarized at halo level as a first SMBH proxy, but there is still no dedicated wanderer / post-disruption IMBH orbital module yet. This is therefore a first bridge from GC formation to high-$z$ SMBHs, not the final BH growth model.