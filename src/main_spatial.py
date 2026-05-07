"""Legacy Gao+2023 GC formation stage.

This script reads one fixed merger tree per target halo, identifies GC-forming
events from halo growth along each branch, samples a cluster initial mass
function for each event, assigns galactocentric radii, and writes:

- `all_<Ns>.txt`: every formed GC, whether it survives to the configured final
  redshift or not
- `z0_cat_<Ns>.txt`: only the subset that survives the analytic disruption
  pass; the filename is historical and is still used even when `final_z > 0`

Although the implementation is legacy-style and fairly stateful, the rough
flow is:
1. load one corrected tree from the configured fixed-tree directory
2. walk every retained branch node and identify rapid-growth events
3. form GCs and assign their birth radii
4. apply the weak-field survival estimate to decide which reach the configured
   final redshift
"""

import argparse
import csv
from dataclasses import dataclass
import numpy as np
from scipy import interpolate
import time
import os
import sys
from pathlib import Path
import schechter_interp
import smhm
from IMBH import IMBHModel, IMBHModelConfig
np.random.seed(1)

#use same cosmology as Illustris
h100 = .704
omega_matter = 0.2726
omega_lambda = 1. - omega_matter
fb = .167 #cosmic baryon fraction


#model parameters, as defined in CGL18
mmr_slope = .35 #mass slope of galaxy mass-metallicity relation
mmr_turnover = 10.5 #pivot in the power-law of mass-metallicity relation
sigma_m = 0.3 #MMR scatter, in dex
max_feh = 0.3 #max [Fe/H]
sigma_gas = 0.3 #cold gas fraction scatter, in dex
log_Mmin = 5.0 #min cluster mass to draw from CIMF
tdep = 0.3 #scaling of the gas depletion time with redshift, tdep \propto (1+z)^(-alpha), alpha as here
mmr_evolution = .9 #redshift slope of the galaxy mass-metallicity relation
pr = 0.5 #normalized period of rotation; t_tid \propto P
miso = (pr/1.7)**3 #mass where tiso < ttid
TREE_LOOKUP_BASENAME = "id_lookup_large_dark.csv"
METAL_CHOKSI2018 = "Choksi+2018"
METAL_CHEN_GNEDIN2024 = "Chen&Gnedin2024"
METAL_CHOICES = (METAL_CHOKSI2018, METAL_CHEN_GNEDIN2024)
ACCRETED_BARYON_MURATOV_GNEDIN2010 = "Muratov&Gnedin2010"
ACCRETED_BARYON_CHEN_GNEDIN2023 = "Chen&Gnedin2023"
ACCRETED_BARYON_CHOICES = (ACCRETED_BARYON_MURATOV_GNEDIN2010, ACCRETED_BARYON_CHEN_GNEDIN2023)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Legacy Gao+2023 GC formation stage.")
    parser.add_argument("ns", type=float, help="Sersic index N_s")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Path to the raw Gao+2023 data directory.",
    )
    parser.add_argument(
        "--tree-dir",
        type=Path,
        default=None,
        help="Optional fixed-tree directory. Defaults to <data-dir>/fixed_trees_large_spin.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where all_<Ns>.txt and z0_cat_<Ns>.txt are written.",
    )
    parser.add_argument("--p2", type=float, default=6.75, help="GC formation-efficiency normalization")
    parser.add_argument("--p3", type=float, default=0.5, help="halo growth-rate threshold for triggering GC formation")
    parser.add_argument("--mpb-only", type=int, default=0, help="restrict the tree walk to the main progenitor branch only")
    parser.add_argument("--lg_cut-off_mass", dest="lg_cut_off_mass", type=float, default=12.0, help="log10 Schechter cutoff mass Mc in Msun")
    parser.add_argument("--metal", choices=METAL_CHOICES, default=METAL_CHOKSI2018, help="stellar mass-metallicity relation")
    parser.add_argument("--accreted_baryon", choices=ACCRETED_BARYON_CHOICES, default=ACCRETED_BARYON_MURATOV_GNEDIN2010, help="accreted-baryon fraction limiter")
    parser.add_argument("--run-all", type=int, default=1, help="run all halos if 1, otherwise use the halo count and mass window below")
    parser.add_argument("--log-mh-min", type=float, default=11.5, help="minimum descendant z=0 host halo log mass for selection")
    parser.add_argument("--log-mh-max", type=float, default=12.5, help="maximum descendant z=0 host halo log mass for selection")
    parser.add_argument("--n-halos", type=int, default=10, help="number of halos to keep when --run-all=0")
    parser.add_argument("--IMBH", type=int, choices=[0, 1], default=1, help="enable the IMBH seeding module if 1, otherwise keep IMBH-related columns at zero")
    parser.add_argument("--final-z", "--final-redshift", dest="final_redshift", type=float, default=0.0, help="final redshift where the formation/survival stage stops; halo selection remains tied to the descendant z=0 host")
    return parser


args = _build_arg_parser().parse_args()

# Sersic index
# ns = 2.2
ns = float(args.ns)
nsStr = f"{ns:.1f}"

data_dir = args.data_dir.resolve()
output_dir = args.output_dir.resolve()
output_dir.mkdir(parents = True, exist_ok = True)

massloss_path = data_dir / "mass_loss.txt"
snaps_path = data_dir / "snaps2redshifts.txt"
treedir = args.tree_dir.resolve() if(args.tree_dir is not None) else (data_dir / "fixed_trees_large_spin")
for required_path in (massloss_path, snaps_path, treedir):
    if(not required_path.exists()):
        raise FileNotFoundError("Required Gao+2023 input path not found: " + str(required_path))

fm = open(massloss_path)
flost, t_solar, t_subsolar = np.loadtxt(fm, usecols = (1,2,3), unpack = True)

ssub = interpolate.interp1d(t_subsolar, flost, kind = 'linear')
ssolar = interpolate.interp1d(t_solar, flost, kind = 'linear')


snaps2 = np.loadtxt(snaps_path, unpack = True)


cat = open(output_dir / ('z0_cat_'+nsStr+'.txt'), 'w') #historical filename for the survivor catalog
allcat = open(output_dir / ('all_'+nsStr+'.txt'), 'w') #full catalog of all GCs

p2 = float(args.p2)
p3 = float(args.p3)
mpb_only = bool(args.mpb_only)
lg_cut_off_mass = float(args.lg_cut_off_mass)
metal_model = args.metal
accreted_baryon_model = args.accreted_baryon
run_all = bool(args.run_all)
log_mh_min = float(args.log_mh_min)
log_mh_max = float(args.log_mh_max)
N = int(args.n_halos)
final_redshift = float(args.final_redshift)
final_epoch_label = "z=0" if(abs(final_redshift) < 1.0e-12) else ("z=" + str(np.round(final_redshift, 5)))
host_epoch_label = "z=0"

cat.write('#model parameters: p2, p3, mpb_only, lg_cut_off_mass, metal, accreted_baryon, z_final = ' +  str(p2) + " " +  str(p3) +  " " + str(mpb_only) + " " +  str(lg_cut_off_mass) + " " + str(metal_model) + " " + str(accreted_baryon_model) + " " + str(final_redshift)  + "\n")
cat.write(
    str("#haloID")
    + " | "
    + str('logMh(' + host_epoch_label + ')')
    + " | "
    + str("logM*(" + host_epoch_label + ")")
    + " | "
    + str('logMh(zform)')
    + " | "
    + str("logM*(zform)")
    + " | "
    + str("logM(" + final_epoch_label + ")")
    + " | "
    + str("logM(zform)")
    + " | "
    + str("zform")
    + " | "
    + str("feh")
    + " | "
    + str("isMPB")
    + " | "
    + str("halo ID @ formation")
    + " | "
    + str("rGalaxy (kpc)")
    + " | "
    + str("GC radius (pc)")
    + " | "
    + str("Sigma_h (Msun/pc^2)")
    + " | "
    + str("IMBH mass (Msun)")
    + "\n"
)

allcat.write('#model parameters: p2, p3, mpb_only, lg_cut_off_mass, metal, accreted_baryon, z_final = ' +  str(p2) + " " +  str(p3) +  " " + str(mpb_only) + " " +  str(lg_cut_off_mass) + " " + str(metal_model) + " " + str(accreted_baryon_model) + " " + str(final_redshift)  + "\n")
allcat.write(
    "#haloID | logMh(" + host_epoch_label + ") | haloID @ form | logMh(tform) | logM*(tform) | "
    "logMgas(tform) | logMcl(tform) | zform | [Fe/H] | rGalaxy (kpc) | "
    "GC radius (pc) | Sigma_h (Msun/pc^2) | IMBH mass (Msun)\n"
)

#initialize all the interpolation tables for use with Schechter function
schechter_interp.init(10**lg_cut_off_mass)
mgc_to_mmax = schechter_interp.generate(10**lg_cut_off_mass)
alpha = -2.0
ug52 = schechter_interp.upper_gamma2(5.0)

# First-step GC IMBH seeding: seed exactly once at GC formation, using the
# Eq. (7) cluster radius relation and the GC metallicity after intrinsic
# cluster-to-cluster scatter has been applied.
imbh_model = IMBHModel(IMBHModelConfig(enabled = bool(args.IMBH)))

# Define globular cluster class to store data about each cluster.
class GC :
    def __init__(self, mass, originHaloMass, origin_redshift, metallicity, osm, omgas, is_mpb, idform,
                 gc_radius_pc = 0.0, gc_sigma_h_msun_pc2 = 0.0, imbh_mass_msun = 0.0) :
        self.mass = mass
        self.originHaloMass = originHaloMass
        self.origin_redshift = origin_redshift
        self.metallicity = metallicity
        self.origin_sm = osm
        self.origin_mgas = omgas
        self.is_mpb = is_mpb
        self.idform = idform
        self.rGalaxy = 0.0
        self.gc_radius_pc = gc_radius_pc
        self.gc_sigma_h_msun_pc2 = gc_sigma_h_msun_pc2
        self.imbh_mass_msun = imbh_mass_msun
    def assign_rGalaxy(self, radius):
        self.rGalaxy = radius


def seed_imbh_properties(cluster_mass, metallicity):
    if(not imbh_model.config.enabled):
        return 0.0, 0.0, 0.0

    estimate = imbh_model.estimate_for_gc(cluster_mass, metallicity)
    gc_radius_pc = float(estimate["r_h_pc"])
    sigma_h_msun_pc2 = float(estimate["sigma_h_msun_pc2"])
    imbh_mass_msun = float(estimate["imbh_mass_msun"])

    if(not np.isfinite(gc_radius_pc) or gc_radius_pc < 0):
        gc_radius_pc = 0.0
    if(not np.isfinite(sigma_h_msun_pc2) or sigma_h_msun_pc2 < 0):
        sigma_h_msun_pc2 = 0.0
    if(not np.isfinite(imbh_mass_msun) or imbh_mass_msun < 0):
        imbh_mass_msun = 0.0

    return gc_radius_pc, sigma_h_msun_pc2, imbh_mass_msun

def soft_step(x, y):
    return (1.0 + (2.0**(y/3.0) - 1.0) * x**y)**(-3.0/y)


def muratov_gnedin2010_fin_norm(Mh, z):
    mchar_baryon = 3.6e9*np.exp(-.6*(1+z))/h100
    mchar_baryon_min = 1.5e10*(180**(-.5))/(smhm.E(z)*h100)
    if(mchar_baryon < mchar_baryon_min):
        mchar_baryon = mchar_baryon_min
    return 1/((1+(mchar_baryon/Mh))**3)


def chen_gnedin2023_fin_norm(Mh, z):
    z_rei = 6.0
    gamma = 15.0
    beta = z_rei*(np.log(1.82e3*np.exp(-0.63*z_rei)) - 1.0)**(-1.0/gamma)
    mchar_baryon = 1.69e10*np.exp(-0.63*z - np.logaddexp(0.0, (z/beta)**gamma))
    return soft_step(mchar_baryon/Mh, 2.0)


def accreted_baryon_fin_norm(Mh, z):
    if(accreted_baryon_model == ACCRETED_BARYON_MURATOV_GNEDIN2010):
        return muratov_gnedin2010_fin_norm(Mh, z)
    if(accreted_baryon_model == ACCRETED_BARYON_CHEN_GNEDIN2023):
        return chen_gnedin2023_fin_norm(Mh, z)
    raise ValueError("Unknown accreted_baryon model: " + str(accreted_baryon_model))


# Calculate gas mass given stellar mass, halo mass, redshift using scaling relations. Double power law for SM-Mg relation, then scale with redshift. Revise if gas fraction exceeds accreted baryon fraction. As described in Choksi, Gnedin, and Li (2018).
def gasMass(SM, Mh, z) :
    slope = 0.33
    if(SM < 1e9):
        slope = 0.19
    log_ratio = 0.05 - 0.5 -  slope*(np.log10(SM) - 9.0) #log10(Mg/M*)
    if(z < 3): #fg saturates at z > 3
        if(z < 2):
            log_ratio += (3.0-tdep)*np.log10((1.+z)/3.) + (3.0-tdep)*np.log10(3.) #strong ssfr evolution at z < 2
        else:
            log_ratio += (1.7-tdep)*np.log10((1.+z)/3.) + (3.0-tdep)*np.log10(3.) #weak ssfr evolution at z > 2 (Lilly+)
    else:
        log_ratio += (1.7-tdep)*np.log10((1.+3)/3.) + (3.0-tdep)*np.log10(3.) #weak ssfr evolution at z > 2
    log_ratio += np.random.normal(0, 0.3)
    ratio = 10**log_ratio
    Mg = SM*ratio
    fstar = SM/(fb*Mh)
    fgas = Mg/(fb*Mh)
    fin = accreted_baryon_fin_norm(Mh, z)

    if(fstar+fgas > fin):
        fgas = fin-fstar
        Mg = fgas*fb*Mh
    return Mg


def cap_metallicity(fe_h):
    # The project uses a slightly supersolar saturation for both supported MZRs.
    if(fe_h > max_feh):
        fe_h = max_feh
    return fe_h


# Galaxy stellar mass-metallicity relation, with additional redshift evolution, as described in Choksi, Gnedin, and Li (2018).
def MMR_choksi2018(SM, z):
    local = mmr_slope*(np.log10(SM) - mmr_turnover)
    evolution = mmr_evolution*np.log10(1+z)
    fe_h = local - evolution
    return cap_metallicity(fe_h)


def MMR_chen_gnedin2024(SM, z):
    fe_h = 0.3*np.log10(SM/1.0e9) - 1.0*np.log10(1+z) - 0.5
    return cap_metallicity(fe_h)


def MMR(SM, z):
    if(metal_model == METAL_CHOKSI2018):
        return MMR_choksi2018(SM, z)
    if(metal_model == METAL_CHEN_GNEDIN2024):
        return MMR_chen_gnedin2024(SM, z)
    raise ValueError("Unknown metal model: " + str(metal_model))


import scipy.special as special
import scipy.integrate as integrate
from scipy.interpolate import interp1d

def rho_sersic(R, A, Re, ns):
    return A * np.exp( -2 * ns * (R/Re)**(1./ns) )

def mtot_sersic(A,Re,ns):
    return 4*np.pi*A/8**ns * ns**(1-3*ns) * Re**3 * special.gamma(3*ns)

def Mr_frac_sersic_inverse(fm, ns):
    # give the mass fraction of total enclosed mass
    # return radius of the exactly location in unit of re
    p = 1.0 - 0.6097/ns + 0.05563/ns/ns
    b = 2.*ns - 1/3. + 0.009876/ns
    ZZ = special.gammaincinv(ns*(3.-p), fm)
    return (ZZ/b)**ns

def gc_sersic_sampling(gc_list, mass_sum, halomass, redshift, jsp, ns):
    """
    sample GC spatial distribution within galactic disk with a Sersic profile
    """
    rgal_min_kpc = 1.0e-3
    rgal_max_kpc = 1.0e4
    rVir = smhm.virialRadius(halomass, redshift) # in pc
    Hz = smhm.H0*smhm.E(redshift) # in Myr^-1

    fallback_outer_kpc = 0.5*rVir/1e3
    #fallback_outer_kpc = 0.1*rVir/1e3
    if((not np.isfinite(fallback_outer_kpc)) or (fallback_outer_kpc <= 0.0)):
        print(
            "[gc_sersic_sampling] invalid fallback outer radius "
            + f"(rVir={rVir}, halomass={halomass}, z={redshift}); using 1.0 kpc",
            file = sys.stderr,
        )
        fallback_outer_kpc = 1.0
    fallback_outer_kpc = float(np.clip(fallback_outer_kpc, rgal_min_kpc, rgal_max_kpc))

    # jsp in (kpc/h)*(km/s) similar to (kpc/h)*(pc/Myr)
    Re = jsp*h100/20./Hz/rVir # in kpc
    if((not np.isfinite(Re)) or (Re <= 0.0)):
        print(
            "[gc_sersic_sampling] invalid Sersic scale radius "
            + f"(Re={Re}, jsp={jsp}, Hz={Hz}, rVir={rVir}, halomass={halomass}, z={redshift}); "
            + f"using fallback Re={fallback_outer_kpc} kpc",
            file = sys.stderr,
        )
        Re = fallback_outer_kpc
    Re = float(np.clip(Re, rgal_min_kpc, rgal_max_kpc))

    # Sersic profile based on inverse incomplete Gamma function, see eq. (A2) in Terzic & Graham 2005
    if((not np.isfinite(mass_sum)) or (mass_sum <= 0.0)):
        print(
            "[gc_sersic_sampling] invalid total GC mass budget "
            + f"(mass_sum={mass_sum}, halomass={halomass}, z={redshift}); "
            + "placing all GCs at the fallback outer radius",
            file = sys.stderr,
        )
        for gc in gc_list:
            gc.assign_rGalaxy(fallback_outer_kpc)
        return

    m_tot = -0.5*gc_list[0].mass
    for gc in gc_list:
        if gc.is_mpb:
            # Main-branch clusters are ordered by cumulative formed mass so the
            # distribution matches the target enclosed Sersic mass profile.
            m_tot += gc.mass
            enclosed_mass_fraction_raw = m_tot/mass_sum
            if((not np.isfinite(enclosed_mass_fraction_raw)) or (enclosed_mass_fraction_raw <= 0.0) or (enclosed_mass_fraction_raw >= 1.0)):
                enclosed_mass_fraction = np.clip(enclosed_mass_fraction_raw, 1.0e-12, 1.0-1.0e-12) if np.isfinite(enclosed_mass_fraction_raw) else 1.0e-12
                print(
                    "[gc_sersic_sampling] invalid enclosed Sersic mass fraction "
                    + f"(fm={enclosed_mass_fraction_raw}, m_tot={m_tot}, mass_sum={mass_sum}, "
                    + f"halomass={halomass}, z={redshift}); clamping to {enclosed_mass_fraction}",
                    file = sys.stderr,
                )
            else:
                enclosed_mass_fraction = enclosed_mass_fraction_raw

            rGalaxy = Mr_frac_sersic_inverse(enclosed_mass_fraction, ns) * Re
            if((not np.isfinite(rGalaxy)) or (rGalaxy <= 0.0)):
                print(
                    "[gc_sersic_sampling] invalid Sersic-sampled GC radius "
                    + f"(rGalaxy={rGalaxy}, fm={enclosed_mass_fraction}, Re={Re}, "
                    + f"halomass={halomass}, z={redshift}); using fallback outer radius",
                    file = sys.stderr,
                )
                rGalaxy = fallback_outer_kpc
            rGalaxy = float(np.clip(rGalaxy, rgal_min_kpc, rgal_max_kpc))
            gc.assign_rGalaxy(rGalaxy)
        else:
            # Non-MPB clusters are treated as accreted satellites and placed at
            # a representative outer-halo radius rather than in the disk model.
            m_tot += gc.mass
            gc.assign_rGalaxy(fallback_outer_kpc)

    # uniform radius distribution
    # for gc in gc_list:
    #     if gc.is_mpb:
    #         rGalaxy = np.random.uniform(0, 5*Re)
    #         gc.assign_rGalaxy(rGalaxy)
    #     else:
    #         gc.assign_rGalaxy(0.5*rVir/1e3)

    return


def clusterFormation(Mg, halomass, redshift, metallicity, SM, is_mpb, hid, jj) :
    gc_list = []
    Mgc = 3e-5*p2*Mg/fb #total mass of all GCs formed in cluster formation event
    log_Mgc = np.log10(Mgc)
    if(log_Mgc < log_Mmin): #not enough mass to form a single cluster of mass Mmin
        return gc_list
    # First reserve the most massive cluster explicitly, then sample the rest
    # from the Schechter CIMF until the event budget Mgc is exhausted.
    # This mirrors the historical Fortran/IDL logic used in the original model.
    # calculate the cumulative distribution r(<M), and invert it numerically
    log_Mmax = mgc_to_mmax(log_Mgc)
    if(log_Mmax > log_Mgc):
        log_Mmax = log_Mgc
    Mmax = 10**log_Mmax
    mt = np.logspace(log_Mmin, log_Mmax, num = 500)

    maxGC_metallicity = metallicity + np.random.normal(0,sigma_m)
    gc_radius_pc, sigma_h_msun_pc2, imbh_mass_msun = seed_imbh_properties(Mmax, maxGC_metallicity)
    maxGC = GC(
        Mmax,
        halomass,
        redshift,
        maxGC_metallicity,
        SM,
        Mg,
        is_mpb,
        hid,
        gc_radius_pc = gc_radius_pc,
        gc_sigma_h_msun_pc2 = sigma_h_msun_pc2,
        imbh_mass_msun = imbh_mass_msun,
    )
    gc_list.append(maxGC)
    mass_sum = Mmax

    ntot = ug52 - schechter_interp.upper_gamma2(log_Mmax)
    cum = np.array([(ug52 - schechter_interp.upper_gamma2(np.log10(mv)))/ntot for mv in mt])

    r_to_m = interpolate.interp1d(cum, mt)
    mass_sum2 = Mmax
    while(mass_sum < Mgc):
        r = np.random.random()
        M = r_to_m(r)
        if(mass_sum+M > Mgc): #make sure the final cluster drawn doesn't exceed the total mass to be formed. it may produce some clusters below Mmin, but shouldn't really matter (will disrupt)
            M = Mgc-mass_sum
        mass_sum += M
        cluster_metallicity = metallicity + np.random.normal(0,sigma_m)
        gc_radius_pc, sigma_h_msun_pc2, imbh_mass_msun = seed_imbh_properties(M, cluster_metallicity)
        cluster = GC(
            M,
            halomass,
            redshift,
            cluster_metallicity,
            SM,
            Mg,
            is_mpb,
            hid,
            gc_radius_pc = gc_radius_pc,
            gc_sigma_h_msun_pc2 = sigma_h_msun_pc2,
            imbh_mass_msun = imbh_mass_msun,
        )
        gc_list.append(cluster)

    # Shuffle before assigning radii so the maxGC does not always
    # inherit the smallest radius purely because it was appended first.
    np.random.shuffle(gc_list)
    # sample spatial distribution of GCs within a Sersic disk
    gc_sersic_sampling(gc_list, mass_sum, halomass, redshift, jj, ns)
    return gc_list


#metallicity dependent stellar evolution, as calculated by Prieto & Gnedin
def massFraction(fe_h, t_alive):
    if(fe_h >= 0 ): #supersolar metallicity
        return ssolar(t_alive)
    elif(fe_h <= -0.69): #if Z<= 0.2Zsun
        return ssub(t_alive)
    else: #intermediate case: interpolate (in log-space)  between .2*Z_sun and Z_sun
        slope = (t_solar - t_subsolar)/0.69 #log10(0.2) = -0.69
        t_int= t_subsolar + slope*(fe_h - -0.69) #new intermediate MS lifetime table
        sint = interpolate.interp1d(t_int, flost, kind = 'linear')
        return sint(t_alive)
#disruption prescription as described in CGL18
def disruption(M0, fe_h, origin_redshift, tnow, use_weak = False): #if use_weak = True, allow for disruption in weak tidal field limit (see CGL18 for details)
    m0 = M0/2e5
    tform = smhm.cosmicTime(origin_redshift, units = 'yr')
    t_tid0 = 1e10*m0**(2./3)*pr
    nu_tid0 = 1./t_tid0

    k = 1-(2./3)*nu_tid0*(tnow - tform)
    k = max(k, 0) #can't raise negative number to fractional exponent; if 1-k < 0, the tidal-evolved mass at the final epoch would be below miso
    mtid_z0 = m0*k**1.5

    if(use_weak): #use dM/dt = -M/min(t_tid, t_iso), as in CGL18
        if(mtid_z0 >= miso):
            mf = mtid_z0
        else:
            tiso = tform + 3.0*(1 - (miso/m0)**(2./3))/(2*nu_tid0)
            mf = miso - (tnow - tiso)/(1.7e10)
        if(mf <= 0):
            return 0

    f_lost = massFraction(fe_h, tnow - tform) #stellar evolution
    if(use_weak):
        return mf*2e5*(1-f_lost)
    else:
        return mtid_z0*2e5*(1-f_lost)


@dataclass(frozen = True)
class TreeEntry:
    halo_id_z0: int
    path: Path


def _legacy_tree_entries(tree_dir):
    tree_entries = []
    for path in sorted(tree_dir.iterdir()):
        if(not path.is_file()):
            continue
        if(path.suffix.lower() not in (".txt", ".dat")):
            continue
        try:
            hid = int(path.stem)
        except ValueError:
            continue
        tree_entries.append(TreeEntry(halo_id_z0 = hid, path = path))
    return tree_entries


def _iter_tree_files(tree_dir):
    lookup_path = tree_dir / TREE_LOOKUP_BASENAME
    if(lookup_path.is_file()):
        tree_entries = []
        seen_halo_ids = set()
        with lookup_path.open("r", encoding = "utf-8", newline = "") as handle:
            for row in csv.DictReader(handle):
                try:
                    hid = int(row["halo_id_z0"])
                    basename = row["fixed_tree_basename"].strip()
                except (KeyError, ValueError) as exc:
                    raise RuntimeError("Malformed tree lookup row in " + str(lookup_path) + ": " + str(row)) from exc
                tree_path = tree_dir / basename
                if(not tree_path.is_file()):
                    raise FileNotFoundError("Tree lookup references missing fixed tree: " + str(tree_path))
                if(hid in seen_halo_ids):
                    raise RuntimeError("Duplicate halo_id_z0 " + str(hid) + " in tree lookup " + str(lookup_path))
                seen_halo_ids.add(hid)
                tree_entries.append(TreeEntry(halo_id_z0 = hid, path = tree_path))
        if(len(tree_entries) == 0):
            raise RuntimeError("Tree lookup exists but contains no usable rows: " + str(lookup_path))
        return tree_entries

    tree_entries = _legacy_tree_entries(tree_dir)
    if(len(tree_entries) == 0):
        raise RuntimeError("No usable fixed-tree files were found under " + str(tree_dir))
    return tree_entries


def loadTree(tree_path):
    m, fp, subid, mpi, redshifts, jsp = [], [], [], [], [], []
    with open(tree_path) as f:
        count = -1
        for line in f:
            count += 1
            if(count == 0):
                continue
            cols = line.split()
            m.append(float(cols[0])); fp.append(int(cols[1])); subid.append(int(cols[2])); mpi.append(int(cols[3])); redshifts.append(float(cols[5]));
            jsp.append(np.sqrt(float(cols[6])*float(cols[6])+float(cols[7])*float(cols[7])+float(cols[8])*float(cols[8])))
    if(len(m) == 0):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 0.0, -1
    m  = np.array(m); fp  = np.array(fp); subid = np.array(subid); mpi = np.array(mpi); redshifts = np.array(redshifts); jsp = np.array(jsp)
    mpbi = mpi[m == np.amax(m)][0]
    main_mask_full = mpi == mpbi
    if(np.any(main_mask_full)):
        msub_z0 = 10**np.amax(m[main_mask_full])
    else:
        msub_z0 = 10**np.amax(m)
    keep = redshifts >= final_redshift
    m = m[keep]; fp = fp[keep]; subid = subid[keep]; redshifts = redshifts[keep]; jsp = jsp[keep]; mpi = mpi[keep]
    if(len(m) == 0):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), msub_z0, mpbi
    if(mpb_only):
        # `mpb_only` reduces the event search to the main branch only. The
        # fixed tree files contain all retained branches by default.
        m = m[mpi == mpbi]; fp = fp[mpi == mpbi]; subid = subid[mpi == mpbi]; redshifts = redshifts[mpi == mpbi]; jsp = jsp[mpi == mpbi]; mpi = mpi[mpi == mpbi]
    if(len(m) == 0):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), msub_z0, mpbi
    m = 10**m
    return m, fp, subid, redshifts, jsp, mpi, msub_z0, mpbi


t0 = smhm.cosmicTime(final_redshift, units = 'yr')
#id_dict = loadDict()


num = -1
num_run = 0
for tree_entry in _iter_tree_files(treedir):
    hid_num = int(tree_entry.halo_id_z0)
    tree_path = tree_entry.path
    m, fp, subid, redshifts, jsp, mpi, msub_z0, mpbi = loadTree(tree_path)
    if(len(m) == 0):
        continue
    # if(run_all == False and msub  < 10**log_mh_min or msub > 10**log_mh_max or num_run >= N):
        # continue

    if (run_all == False):
        if (num_run >= N):
            continue
        # The selection cut is always applied to the descendant z=0 host mass,
        # while `final_redshift` only truncates the subsequent formation history.
        if (msub_z0  < 10**log_mh_min) or (msub_z0 > 10**log_mh_max):
            continue

    num_run += 1

    # Go through each halo along the tree and look for events satisfying Rm > p3.
    sm_arr = np.zeros(len(redshifts))
    clusters = []
    for i in range(0, len(m)) : #for each halo in the merger tree
        mass = m[i] #mass of this halo
        fpID = fp[i] #ID of the main progenitor
        jj = jsp[i]

        if(fpID == -1 or len(subid[subid == fpID]) == 0): #then we've reached the first point along this track of the tree
            sm_arr[i] = smhm.SMHM(mass, redshifts[i], scatter = True) #assign a "seed" stellar mass which we will grow self-consistently
            continue

        progIdx = np.where(subid == fpID)[0][0] #identify index of first progenitor in data
        progMass = m[progIdx] #get mass of fprogenitor

        ratio = mass/progMass - 1 #calculate merger ratio, Rm = dMh/Mh
        znow, zbefore = redshifts[i], redshifts[progIdx]
        dt = smhm.cosmicTime(znow, units = 'Gyr')- smhm.cosmicTime(zbefore, units = 'Gyr')
        ratio = ratio/dt #(dMh/Mh)/dt

        #evolve stellar mass self-consistently as described in CGL18
        sm1 = smhm.SMHM(mass, znow, scatter = False); sm2 = smhm.SMHM(progMass, zbefore, scatter = False)
        dsm = sm1-sm2
        a = 1./(1+znow)
        scatter = np.random.normal(0,.218 - .023*(a-1))
        SM = sm_arr[progIdx] + dsm*10**scatter
        if(SM < 0): #only happens in very weird cases at very high redshift
            SM = sm_arr[progIdx]
        sm_arr[i] = SM
        Mg = gasMass(SM, mass, znow)
        if(ratio > p3):  #if merger criterion satisfied
            metallicity = MMR(SM, znow)
            is_mpb = mpi[i] == mpbi
            # `subid[i]` records the halo hosting the formation event; later
            # stages use it to mark MPB vs accreted GCs in merged catalogs.
            clusters.extend(clusterFormation(Mg, mass, znow, metallicity, SM, is_mpb, subid[i], jj))
            continue

    clusters2, log_initial_masses = [], []
    for cluster in clusters:
        final_mass = disruption(cluster.mass, cluster.metallicity, cluster.origin_redshift, t0)
        if(final_mass > 0):
            log_initial_masses.append(np.log10(cluster.mass))
        evolved_cluster = GC(
            final_mass,
            cluster.originHaloMass,
            cluster.origin_redshift,
            cluster.metallicity,
            cluster.origin_sm,
            cluster.origin_mgas,
            cluster.is_mpb,
            cluster.idform,
            gc_radius_pc = cluster.gc_radius_pc,
            gc_sigma_h_msun_pc2 = cluster.gc_sigma_h_msun_pc2,
            imbh_mass_msun = cluster.imbh_mass_msun,
        )
        evolved_cluster.assign_rGalaxy(cluster.rGalaxy)
        clusters2.append(evolved_cluster)

    evolved_clusters = [cluster for cluster in clusters2 if cluster.mass > 0]

    # all GCs that form, regardless of survival -- for use w/ allcat.txt
    GC_mets = np.array([cluster.metallicity for cluster in clusters]); GC_masses = np.array([cluster.mass for cluster in clusters]); GC_log_masses = np.log10(GC_masses); GC_redshifts = np.array([cluster.origin_redshift for cluster in clusters])
    GC_idform = np.array([cluster.idform for cluster in clusters]); GC_mhost_tform = np.array([cluster.originHaloMass for cluster in clusters])
    GC_log_mhost_tform = np.log10(GC_mhost_tform); GC_log_mstar_tform = np.round(np.log10(np.array([cluster.origin_sm for cluster in clusters])), 3)
    GC_log_mgas_tform = np.round(np.log10(np.array([cluster.origin_mgas for cluster in clusters])), 3)
    GC_radius = np.array([cluster.rGalaxy for cluster in clusters])
    GC_gc_radius_pc = np.array([cluster.gc_radius_pc for cluster in clusters])
    GC_sigma_h = np.array([cluster.gc_sigma_h_msun_pc2 for cluster in clusters])
    GC_imbh_mass = np.array([cluster.imbh_mass_msun for cluster in clusters])

    # Repeat the same bookkeeping for the survivors at the chosen final redshift.
    GC_masses2 = np.array([cluster.mass for cluster in evolved_clusters]); GC_metallicity2 = np.array([cluster.metallicity for cluster in evolved_clusters])
    GC_redshifts2 = np.array([cluster.origin_redshift for cluster in evolved_clusters]); GC_mhost_tform2 = np.array([cluster.originHaloMass for cluster in evolved_clusters])
    GC_mstar_tform2 = np.array([cluster.origin_sm for cluster in evolved_clusters]); GC_ismpb2 = np.array([cluster.is_mpb for cluster in evolved_clusters])
    GC_idform2 = np.array([cluster.idform for cluster in evolved_clusters]); log_gc_masses2 = np.log10(GC_masses2)
    GC_radius2 = np.array([cluster.rGalaxy for cluster in evolved_clusters])
    GC_gc_radius_pc2 = np.array([cluster.gc_radius_pc for cluster in evolved_clusters])
    GC_sigma_h2 = np.array([cluster.gc_sigma_h_msun_pc2 for cluster in evolved_clusters])
    GC_imbh_mass2 = np.array([cluster.imbh_mass_msun for cluster in evolved_clusters])


    logmsub = np.log10(msub_z0);
    log_GC_mhost =  np.empty(len(GC_masses2))
    log_GC_mhost.fill(logmsub)
    logms = np.log10(smhm.SMHM(msub_z0, 0.0, scatter = False))

    # The historical host columns always refer to the descendant z=0 halo,
    # while GC survival and orbital evolution stop at `final_redshift`.
    # write to output files
    for i in range(len(GC_masses)): #all clusters
        allcat.write(
            str(hid_num)
            + " "
            + str(np.round(logmsub,5))
            + " "
            + str(GC_idform[i])
            + " "
            + str(np.round(GC_log_mhost_tform[i],5))
            + " "
            + str(np.round(GC_log_mstar_tform[i],5))
            + " "
            + str(np.round(GC_log_mgas_tform[i],5))
            + " "
            + str(np.round(GC_log_masses[i],5))
            + " "
            + str(np.round(GC_redshifts[i],5))
            + " "
            + str(np.round(GC_mets[i],5))
            + " "
            + str(np.round(GC_radius[i],5))
            + " "
            + str(np.round(GC_gc_radius_pc[i],5))
            + " "
            + str(np.round(GC_sigma_h[i],5))
            + " "
            + str(np.round(GC_imbh_mass[i],5))
            + "\n"
        )
    for i in range(len(GC_masses2)): #only those surviving to the chosen final redshift
        zform = GC_redshifts2[i]
        mh_tform = GC_mhost_tform2[i]; log_mh_tform = np.log10(mh_tform);
        sm_tform = GC_mstar_tform2[i]; log_sm_tform = np.log10(sm_tform);
        logm = log_gc_masses2[i]; logm_tform = log_initial_masses[i];
        cat.write(
            str(hid_num)
            +  " "
            + str(np.round(logmsub,5))
            + " "
            + str(np.round(logms, 5))
            + " "
            +  str(np.round(log_mh_tform, 5))
            + " "
            + str(np.round(log_sm_tform,5))
            + " "
            +  str(np.round(logm,5))
            + " "
            + str(np.round(logm_tform,5))
            + " "
            + str(np.round(zform, 5))
            + " "
            + str(np.round(GC_metallicity2[i],5))
            + " "
            + str(float(GC_ismpb2[i]))
            +  " "
            + str(GC_idform2[i])
            + " "
            + str(np.round(GC_radius2[i],5))
            + " "
            + str(np.round(GC_gc_radius_pc2[i],5))
            + " "
            + str(np.round(GC_sigma_h2[i],5))
            + " "
            + str(np.round(GC_imbh_mass2[i],5))
            + "\n"
        )
cat.close()
allcat.close()
if(num_run < N and run_all == False):
    print("requested", N, "halos, but there were only", num_run, "halos stored in the mass range you requested. Model was run on all available halos.\n")
print("all done!")
