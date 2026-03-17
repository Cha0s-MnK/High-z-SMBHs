"""Helpers for inverting the Schechter cluster initial mass function.

The formation code repeatedly needs the map

    total GC mass formed in one event -> maximum cluster mass in that event

for a Schechter CIMF. Building that inversion numerically every time would be
expensive, so this module tabulates the needed incomplete-gamma terms once and
then exposes interpolators used inside `main_spatial.py`.
"""

import numpy as np
from scipy import interpolate
import mpmath
import sys
#sys.path.insert(0, '/Users/nchoksi/gdrive/Astro/')
import smhm

log_mv = np.linspace(4.9, 8.6, num = 2000)
dlog_mv_inv = 1./(log_mv[1] - log_mv[0])
gamma_arr1, gamma_arr2 = np.zeros(len(log_mv)), np.zeros(len(log_mv))
s = 0.02
log_mmaxt = np.arange(5.01, 8.6, step = s)
def upper_gamma2(log_mvv): #linear interpolation of the upper incomplete gamma function for the case of a -2 power law
	"""Fast lookup for Gamma(alpha+1, M/Mc) with alpha=-2."""

	return smhm.lininterp(log_mvv, log_mv, gamma_arr2, dlog_mv_inv)
def upper_gamma1(log_mvv): #linear interpolation of the upper incomplete gamma function for the case of a -1 power law
	"""Fast lookup for Gamma(alpha+2, M/Mc) with alpha=-2."""

	return smhm.lininterp(log_mvv, log_mv, gamma_arr1, dlog_mv_inv)
def init(mc, alpha = -2.0):
	"""Populate the incomplete-gamma lookup tables for one cutoff mass `mc`."""

	for i in range(len(log_mv)):
		mvv = 10**log_mv[i]
		gamma_arr2[i] = mpmath.gammainc(alpha+1.0, mvv/mc) 
		gamma_arr1[i] = mpmath.gammainc(alpha+2.0, mvv/mc) 
	print("init_complete")
def generate(mc, alpha = -2.0, mmin = 1e5): #generates functions to interpolate Mgc(M0) and M0(Mmax); combine to give Mgc(Mmax), 
    """Return an interpolator from log10(total GC mass) to log10(Mmax).

    `init(mc)` must be called first so that `upper_gamma1/2` already hold the
    matching incomplete-gamma tables for the same Schechter cutoff mass.
    """

    ug51 = upper_gamma1(5.0)
    mgc = mc*np.array([(ug51 - upper_gamma1(log_mmaxtv))/upper_gamma2(log_mmaxtv) for log_mmaxtv in log_mmaxt])
    mgc_to_mmax = interpolate.interp1d(np.log10(mgc), log_mmaxt)
    return mgc_to_mmax
