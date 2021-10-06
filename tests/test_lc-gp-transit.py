"""
GP + transit example for Kepler 1627 / exoplanet case-studies docs.
"""
import exoplanet as xo

xo.utils.docs_setup()
print(f"exoplanet.__version__ = '{xo.__version__}'")

##########################################

import lightkurve as lk
import numpy as np, matplotlib.pyplot as plt

# Get long cadence light curves for all quarters. Median normalize all
# quarters, remove nans, and run a 5-sigma outlier clipping.
lcf = lk.search_lightcurve(
        "6184894", mission="Kepler", author="Kepler", cadence="long"
).download_all()
lc = lcf.stitch().remove_nans().remove_outliers()

# Require non-zero quality flags, since we have an abundance of data.
lc = lc[lc.quality == 0]

# Make sure that the data type is consistent
x = np.ascontiguousarray(lc.time.value, dtype=np.float64)
y = np.ascontiguousarray(lc.flux, dtype=np.float64)
yerr = np.ascontiguousarray(lc.flux_err, dtype=np.float64)
texp = np.median(np.diff(x))

# Normalize around zero for GP fitting.  Keep in units of relative flux, rather
# than say ppt.
mu = np.nanmedian(y)
y = (y / mu - 1)
yerr = yerr / mu

# Visualize the data.
# Plot #0: the full dataset
# Plot #1: a 100 day slice
# Plot #2: center it on the known Kepler ephemeris.
plt.plot(x, y, "k")
plt.xlim(x.min(), x.max())
plt.xlabel("time [days]")
plt.ylabel("relative flux [ppt]")
plt.title("Kepler 1627")
plt.savefig("temp0.png", bbox_inches='tight')

plt.plot(x, y, "k")
plt.xlabel("time [days]")
plt.ylabel("relative flux [ppt]")
plt.xlim([550,650])
plt.title("Kepler 1627")
plt.savefig("temp1.png", bbox_inches='tight')

plt.plot(x, y, "k")
plt.xlabel("time [days]")
plt.ylabel("relative flux")
plt.xlim([120.6,121]) # transit is here
plt.ylim([-30e-3,-5e-3])
plt.title("Kepler 1627 b")
plt.savefig("temp2.png", bbox_inches='tight')

##########################################
#
# Begin main block
#

import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess

from astropy import units as units, constants as const

def build_model(mask=None, start=None):

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    with pm.Model() as model:

        # Shared parameters
        mean = pm.Normal("mean", mu=0, sd=1, testval=0)

        # Stellar parameters.  These are usually determined from spectroscopy
        # and/or isochrone fits.  We set a bound on the R_star prior simply to
        # show how one might do this.
        logg_star = pm.Normal("logg_star", mu=4.53, sd=0.05)
        r_star = pm.Normal("r_star", mu=1.0, sd=0.018)

        # Here "factor" is defined s.t. factor * 10**logg / r_star = rho
        factor = 5.141596357654149e-05
        rho_star = pm.Deterministic(
            "rho_star", factor*10**logg_star / r_star
        )

        # Limb-darkening: adopt Kipping 2013.
        u_star = xo.QuadLimbDark("u_star")
        star = xo.LimbDarkLightCurve(u_star)

        # To get Rp/R*, fit for log(depth).  This requires an impact parameter
        # prior from 0 to 1, because otherwise there's a sqrt(1-b^2) in the
        # conversion that doesn't make sense.  See
        # https://github.com/exoplanet-dev/exoplanet/blob/e99d1bd68654f21efbbf8400a83889a470d2baf7/src/exoplanet/light_curves/limb_dark.py#L73

        b = pm.Uniform("b", lower=0, upper=1)

        log_depth = pm.Normal("log_depth", mu=np.log(1.8e-3), sigma=1)
        depth = pm.Deterministic("depth", tt.exp(log_depth))

        ror = pm.Deterministic(
            "ror", star.get_ror_from_approx_transit_depth(depth, b),
        )
        r_pl = pm.Deterministic(
            "r_pl", ror*r_star
        )

        # Orbital parameters for the planet.  Use mean values from Holczer+16.
        t0 = pm.Normal("t0", mu=120.790531, sd=0.02, testval=120.790531)
        period = pm.Normal("period", mu=7.202806, sd=0.01, testval=7.202806)

        # Let eccentricity float, for funsies.
        nplanets = 1
        ecs = pmx.UnitDisk(
            "ecs", shape=(2, nplanets),
            testval=0.01 * np.ones((2, nplanets))
        )
        ecc = pm.Deterministic(
            "ecc",
            tt.sum(ecs ** 2, axis=0)
        )
        omega = pm.Deterministic(
            "omega", tt.arctan2(ecs[1], ecs[0])
        )
        xo.eccentricity.vaneylen19(
            "ecc_prior",
            multi=False, shape=nplanets, fixed=True, observed=ecc
        )

        # Define the orbit model.
        orbit = xo.orbits.KeplerianOrbit(
            period=period,
            t0=t0,
            b=b,
            rho_star=rho_star,
            r_star=r_star,
            ecc=ecc,
            omega=omega
        )

        transit_model = (
            mean + tt.sum(
                star.get_light_curve(
                    orbit=orbit, r=r_pl, t=x, texp=texp), axis=-1
            )
        )

        # Convenience function for plotting.
        pm.Deterministic(
            'transit_pred', star.get_light_curve(
                orbit=orbit, r=r_pl, t=x[mask], texp=texp
            )
        )

        # Use the GP model from the stellar variability tutorial at
        # https://gallery.exoplanet.codes/en/latest/tutorials/stellar-variability/

        # A jitter term describing excess white noise
        log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)), sd=2)

        # The parameters of the RotationTerm kernel
        sigma_rot = pm.InverseGamma(
            "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1, 5)
        )
        # Rotation period is 2.6 days, from Lomb Scargle
        log_prot = pm.Normal(
            "log_prot", mu=np.log(2.606418), sd=0.02
        )
        prot = pm.Deterministic("prot", tt.exp(log_prot))
        log_Q0 = pm.Normal(
            "log_Q0", mu=0, sd=2
        )
        log_dQ = pm.Normal(
            "log_dQ", mu=0, sd=2
        )
        f = pm.Uniform(
            "f", lower=0.01, upper=1
        )

        # Set up the Gaussian Process model. See
        # https://celerite2.readthedocs.io/en/latest/tutorials/first/ for an
        # introduction. Here, we have a quasiperiodic term:
        kernel = terms.RotationTerm(
            sigma=sigma_rot,
            period=prot,
            Q0=tt.exp(log_Q0),
            dQ=tt.exp(log_dQ),
            f=f,
        )
        #
        # Note mean of the GP is defined here to be zero.
        #
        gp = GaussianProcess(
            kernel,
            t=x[mask],
            diag=yerr[mask]**2 + tt.exp(2 * log_jitter),
            quiet=True,
        )

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        gp.marginal("transit_obs", observed=y[mask]-transit_model)

        # Compute the GP model prediction for plotting purposes
        pm.Deterministic(
            "gp_pred", gp.predict(y[mask]-transit_model)
        )

        # Track planet radius in Jovian radii
        r_planet = pm.Deterministic(
            "r_planet", (ror*r_star)*( 1*units.Rsun/(1*units.Rjup) ).cgs.value
        )

        # Optimize the MAP solution.
        if start is None:
            start = model.test_point

        map_soln = start

        map_soln = pmx.optimize(
            start=map_soln,
            vars=[sigma_rot, f, prot, log_Q0, log_dQ]
        )
        map_soln = pmx.optimize(
            start=map_soln,
            vars=[log_depth, b, ecc, omega, t0, period, r_star, logg_star,
                  u_star, mean]
        )
        map_soln = pmx.optimize(start=map_soln)

    return model, map_soln

model, map_estimate = build_model()

##########################################

import matplotlib as mpl
from copy import deepcopy
# NOTE: maybe lightkurve has some analog of this function from astrobase?  See
# https://github.com/waqasbhatti/astrobase; you might need to
# ! pip install astrobase
from astrobase.lcmath import phase_bin_magseries

##########################################

def plot_light_curve(x, y, soln, mask=None):

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    plt.close("all")
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    ax = axes[0]

    if len(x[mask]) > int(2e4):
        # see https://github.com/matplotlib/matplotlib/issues/5907
        mpl.rcParams["agg.path.chunksize"] = 10000

    ax.scatter(x[mask], y[mask], c="k", s=0.5, rasterized=True,
               label="data", linewidths=0, zorder=42)
    gp_mod = soln["gp_pred"] + soln["mean"]
    ax.plot(x[mask], gp_mod, color="C2", label="MAP gp model",
            zorder=41, lw=0.5)
    ax.legend(fontsize=10)
    ax.set_ylabel("$f$")

    ax = axes[1]
    ax.plot(x[mask], y[mask] - gp_mod, "k", label="data - MAPgp")
    for i, l in enumerate("b"):
        mod = soln["transit_pred"][:, i]
        ax.plot(
            x[mask], mod, label="planet {0} [model under]".format(l),
            zorder=-10
        )
    ax.legend(fontsize=10, loc=3)
    ax.set_ylabel("$f_\mathrm{dtr}$")

    ax = axes[2]
    ax.plot(x[mask], y[mask] - gp_mod, "k", label="data - MAPgp")
    for i, l in enumerate("b"):
        mod = soln["transit_pred"][:, i]
        ax.plot(
            x[mask], mod, label="planet {0} [model over]".format(l)
        )
    ax.legend(fontsize=10, loc=3)
    ax.set_ylabel("$f_\mathrm{dtr}$ [zoom]")
    ymin = np.min(mod)-0.05*abs(np.min(mod))
    ymax = abs(ymin)
    ax.set_ylim([ymin, ymax])

    ax = axes[3]
    mod = gp_mod + np.sum(soln["transit_pred"], axis=-1)
    ax.plot(x[mask], y[mask] - mod, "k")
    ax.axhline(0, color="#aaaaaa", lw=1)
    ax.set_ylabel("residuals")
    ax.set_xlim(x[mask].min(), x[mask].max())
    ax.set_xlabel("time [days]")

    fig.tight_layout()

def doublemedian(x):
    return np.median(np.median(x, axis=0), axis=0)

def doublemean(x):
    return np.nanmean(np.nanmean(x, axis=0), axis=0)

def doublepctile(x, SIGMA=[2.5,97.5]):
    # [16, 84] for 1-sigma
    # flatten/merge cores and chains. then percentile over both.
    return np.percentile(
        np.reshape(
            np.array(x), (x.shape[0]*x.shape[1], x.shape[2])
        ),
        SIGMA, axis=0
    )

def get_ylimguess(y):
    ylow = np.nanpercentile(y, 0.1)
    yhigh = np.nanpercentile(y, 99.9)
    ydiff = (yhigh-ylow)
    ymin = ylow - 0.35*ydiff
    ymax = yhigh + 0.35*ydiff
    return [ymin,ymax]

def plot_phased_light_curve(
    x, y, yerr, soln, mask=None, from_trace=False, ylimd=None,
    binsize_minutes=20, map_estimate=None, fullxlim=False, BINMS=3,
    show_rms_err=True, hlines=None
):
    """
    Plot a phased light curve using either a MAP solution, or using the full
    MCMC posterior.  (Or both).  Overkill for a minimum-working-example, but
    this is what I had already cooked up.

    Args:

        soln (az.data.inference_data.InferenceData): Can be MAP solution from
        PyMC3. Can also be the posterior's trace itself
        (model.trace.posterior).  If the posterior is passed, bands showing the
        2-sigma uncertainty interval will be drawn.

        from_trace (bool): set to be True if using model.trace.posterior as
        your `soln`.

        ylimd (dict): dictionary the sets the ylimits of the plot, e.g.,
        `{"A": [-2.2,0.5], "B": [-0.1,0.1]}`.

        binsize_minutes (float): how many minutes per bin?

        map_estimate: if passed, this is used as the "best fit" line (useful
        when doing uncertainty bands from the full MCMC posterior). Otherwise,
        the nanmean is used.  This is most useful when drawing uncertainty
        bands with `soln` being `model.trace.posterior`.

        fullxlim (bool): if True, the xlimits of the plot will be the full
        orbit.  Otherwise, it'll be a ~20 hour window centered on the transit.

        BINMS (float): marker size for binned data points.

        show_rms_err (bool): if True, a representative error will be drawn
        for the binned points using the scatter of the out of transit points.

        hlines (list): if passed, horizontal lines will be drawn.
    """

    if not fullxlim:
        scale_x = lambda x: x*24
    else:
        scale_x = lambda x: x

    if mask is None:
        mask = np.ones(len(x), dtype=bool)

    plt.close("all")
    fig = plt.figure(figsize=(8,6))
    axd = fig.subplot_mosaic(
        """
        A
        B
        """,
        gridspec_kw={
            "height_ratios": [1,1]
        }
    )

    if from_trace==True:
        _t0 = np.nanmean(soln["t0"])
        _per = np.nanmean(soln["period"])

        if len(soln["gp_pred"].shape)==3:
            # ncores X nchains X time
            medfunc = doublemean
            pctfunc = doublepctile
        elif len(soln["gp_pred"].shape)==2:
            medfunc = lambda x: np.mean(x, axis=0)
            pctfunc = lambda x: np.percentile(x, [2.5,97.5], axis=0)
        else:
            raise NotImplementedError
        gp_mod = (
            medfunc(soln["gp_pred"]) +
            medfunc(soln["mean"])
        )
        lc_mod = (
            medfunc(np.sum(soln["transit_pred"], axis=-1))
        )
        lc_mod_band = (
            pctfunc(np.sum(soln["transit_pred"], axis=-1))
        )

        _yerr = (
            np.sqrt(yerr[mask] ** 2 +
                    np.exp(2 * medfunc(soln["log_jitter"])))
        )

        med_error = np.nanmedian(yerr[mask])
        med_jitter = np.nanmedian(np.exp(medfunc(soln["log_jitter"])))

        print(42*"-")
        print(f"WRN! Median σ_f = {med_error:.2e}. "
              f"Median jitter = {med_jitter:.2e}")
        print(42*"-")

    if (from_trace == False) or (map_estimate is not None):
        if map_estimate is not None:
            # If map_estimate is given, over-ride the mean/median estimate
            # above, take the MAP.
            print("WRN! Overriding mean/median estimate with MAP.")
            soln = deepcopy(map_estimate)
        _t0 = soln["t0"]
        _per = soln["period"]
        gp_mod = soln["gp_pred"] + soln["mean"]
        lc_mod = soln["transit_pred"][:, 0]
        _yerr = (
            np.sqrt(yerr[mask] ** 2 + np.exp(2 * soln["log_jitter"]))
        )

    x_fold = (x - _t0 + 0.5 * _per) % _per - 0.5 * _per

    #For plotting
    lc_modx = x_fold[mask]
    lc_mody = lc_mod[np.argsort(lc_modx)]
    if from_trace==True:
        lc_mod_lo = lc_mod_band[0][np.argsort(lc_modx)]
        lc_mod_hi = lc_mod_band[1][np.argsort(lc_modx)]
    lc_modx = np.sort(lc_modx)

    if len(x_fold[mask]) > int(2e4):
        # see https://github.com/matplotlib/matplotlib/issues/5907
        mpl.rcParams["agg.path.chunksize"] = 10000

    #
    # begin the plot!
    #
    ax = axd["A"]

    ax.errorbar(
        scale_x(x_fold[mask]), 1e3*(y[mask]-gp_mod), yerr=1e3*_yerr,
        color="darkgray", label="data-GP", fmt=".", elinewidth=0.2, capsize=0,
        markersize=1, rasterized=True, zorder=-1
    )

    binsize_days = (binsize_minutes / (60*24))
    orb_bd = phase_bin_magseries(
        x_fold[mask], y[mask]-gp_mod, binsize=binsize_days, minbinelems=3
    )
    ax.scatter(
        scale_x(orb_bd["binnedphases"]), 1e3*(orb_bd["binnedmags"]), color="k",
        s=BINMS,
        alpha=1, zorder=1002#, linewidths=0.2, edgecolors='white'
    )

    ax.plot(
        scale_x(lc_modx), 1e3*lc_mody, color="C4", label="transit model", lw=1,
        zorder=1001, alpha=1
    )

    if from_trace==True:
        art = ax.fill_between(
            scale_x(lc_modx), 1e3*lc_mod_lo, 1e3*lc_mod_hi, color="C4",
            alpha=0.5, zorder=1000
        )
        art.set_edgecolor("none")

    ax.set_xticklabels([])

    # residual axis
    ax = axd["B"]

    binsize_days = (binsize_minutes / (60*24))
    # data - GP - mean - transit
    orb_bd = phase_bin_magseries(
        x_fold[mask], y[mask]-gp_mod-lc_mod, binsize=binsize_days, minbinelems=3
    )
    ax.scatter(
        scale_x(orb_bd["binnedphases"]), 1e3*(orb_bd["binnedmags"]), color="k",
        s=BINMS, alpha=1, zorder=1002#, linewidths=0.2, edgecolors='white'
    )
    ax.axhline(0, color="C4", lw=1, ls="-", zorder=1000)

    if from_trace==True:
        sigma = 30
        print(f"WRN! Smoothing plotted by by sigma={sigma}")
        _g =  lambda a: gaussian_filter(a, sigma=sigma)
        art = ax.fill_between(
            scale_x(lc_modx), 1e3*_g(lc_mod_hi-lc_mody),
            1e3*_g(lc_mod_lo-lc_mody), color="C4", alpha=0.5, zorder=1000
        )
        art.set_edgecolor("none")

    ax.set_xlabel("Hours from mid-transit")
    if fullxlim:
        ax.set_xlabel("Days from mid-transit")

    fig.text(-0.01,0.5, "Relative flux [ppt]", va="center",
             rotation=90)

    for k,a in axd.items():
        if not fullxlim:
            a.set_xlim(-0.4*24,0.4*24)
        else:
            a.set_xlim(-_per/2,_per/2)
        if isinstance(ylimd, dict):
            a.set_ylim(ylimd[k])
        else:
            # sensible default guesses
            _y = 1e3*(y[mask]-gp_mod)
            axd["A"].set_ylim(get_ylimguess(_y))
            _y = 1e3*(y[mask] - gp_mod - lc_mod)
            axd["B"].set_ylim(get_ylimguess(_y))

    if show_rms_err:
        # Define "out of transit" to mean at least 3 hours from mid-transit
        # NOTE: this is somewhat dangerous, because if the errors are totally
        # wrong, you would not know it.
        sel = np.abs(orb_bd["binnedphases"]*24)>3
        binned_err = 1e3*np.nanstd((orb_bd["binnedmags"][sel]))
        print(f"WRN! Overriding binned unc as the residuals. "
              f"Binned_err = {binned_err:.4f} ppt")

    if hlines is not None:
        xlim = ax.get_xlim()
        axd["A"].hlines(
            hlines, xlim[0], xlim[1], ls="-", lw=0.5, colors="k"
        )

    _x,_y = 0.8*max(axd["A"].get_xlim()), 0.7*min(axd["A"].get_ylim())
    axd["A"].errorbar(
        _x, _y, yerr=binned_err,
        fmt="none", ecolor="black", alpha=1, elinewidth=0.5, capsize=2,
        markeredgewidth=0.5
    )
    _x,_y = 0.8*max(axd["B"].get_xlim()), 0.6*min(axd["B"].get_ylim())
    axd["B"].errorbar(
        _x, _y, yerr=binned_err,
        fmt="none", ecolor="black", alpha=1, elinewidth=0.5, capsize=2,
        markeredgewidth=0.5
    )

    fig.tight_layout()

plot_light_curve(x, y, map_estimate)
plt.savefig("temp3.png", bbox_inches="tight")

hlines = [-1e3*map_estimate["depth"],
          -1e3*(map_estimate["depth"]-map_estimate["mean"])]
plt.close("all")
plot_phased_light_curve(
    x, y, yerr, map_estimate, ylimd={"A": [-2.2,0.5], "B": [-0.1,0.1]},
    hlines=hlines
)
plt.savefig("temp4.png", bbox_inches="tight")

plot_phased_light_curve(
    x, y, yerr, map_estimate, ylimd={"A": [-2.2,0.5], "B": [-0.1,0.1]},
    hlines=hlines, fullxlim=True
)
plt.savefig("temp5.png", bbox_inches="tight")

plot_phased_light_curve(
    x, y, yerr, map_estimate, ylimd={"A": [-0.1,0.1], "B": [-0.1,0.1]},
    hlines=hlines, fullxlim=True
)
plt.savefig("temp6.png", bbox_inches="tight")


params = ('mean,logg_star,r_star,period,t0,log_depth,b,log_jitter,log_prot'
         ',log_Q0,log_dQ,r_star,rho_star,depth,ror,sigma_rot,prot,f,r_planet'
         .split(','))
for p in params:
    print(f"{p}: {map_estimate[p]:.4f}")

print(f"jitter is {np.exp(map_estimate['log_jitter'])*1e3:.2f} ppt")
print(f"mean(yerr) is {np.mean(yerr)*1e3:.2f} ppt")
print(f"MAP depth is {map_estimate['depth']*1e3:.2f} ppt")
print(f"mean is {map_estimate['mean']*1e3:.2f} ppt")

# We can of course go and sample. This takes a while.
##########################################
