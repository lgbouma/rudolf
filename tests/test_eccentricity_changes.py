import numpy as np
import matplotlib.pyplot as plt
from astropy import units as units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rudolf.plotting import multiline
from aesthetic.plot import savefig, format_ax, set_style

import exoplanet as xo

t = np.linspace(-0.1, 0.1, 1000)

def get_lc(ecc, omega):
	# The light curve calculation requires an orbit
	orbit = xo.orbits.KeplerianOrbit(
		period=7.20281, b=0.44, rho_star=2.00, r_star=0.881, ecc=ecc, omega=omega
	)

	# Compute a limb-darkened light curve using starry
	u = [0.294, 0.377]

	r_pl = (3.789*units.Rearth).to(units.Rsun).value

	t_exp = (29.5*units.minute).to(units.day).value

	# Note: the `eval` is needed because this is using Theano in
	# the background
	lc = (
		xo.LimbDarkLightCurve(*u)
		.get_light_curve(orbit=orbit, r=r_pl, t=t, texp=t_exp)
		.eval()
	)

	return lc.flatten()

def do_omega_plot():

    YSCALE=1e3

    ecc = 0.3
    lc0 = get_lc(0, 0)*YSCALE

    omegas = np.linspace(0,2*np.pi,20)
    light_curves = []
    resids = []
    times = []
    for omega in omegas:
        light_curves.append(get_lc(ecc, omega)*YSCALE)
        resids.append((get_lc(ecc, omega)-lc0/YSCALE)*YSCALE)
        times.append(t)

    #
    # make the plot!
    #
    set_style()
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(5,7), sharex=True)

    axs[0].plot(t, lc0, color="k", lw=2, zorder=10)

    linecollection0 = multiline(
        times, light_curves, omegas, cmap='twilight',
        ax=axs[0], lw=1
    )
    linecollection1 = multiline(
        times, resids, omegas, cmap='twilight',
        ax=axs[1], lw=1
    )

    axs[0].set_title(f"ecc = {ecc}")
    axs[0].set_ylabel("flux [ppt]")
    axs[1].set_ylabel("resid [ppt]")
    axs[1].set_xlabel("t [days]")

    axins1 = inset_axes(axs[0], width="5%", height="30%", loc='lower right',
                        borderpad=1.5)
    cb = fig.colorbar(linecollection0, cax=axins1, orientation="vertical")
    cb.ax.tick_params(labelsize='xx-small')
    cb.ax.set_title('$\omega$ [rad]', fontsize='xx-small')
    cb.ax.tick_params(size=0, which='both') # remove the ticks
    axins1.xaxis.set_ticks_position("bottom")

    for a in axs:
        a.set_xlim(t.min(), t.max())

    savefig(fig, 'test_omega_tuning.png', dpi=400)


def do_ecc_plot():
    t = np.linspace(-0.1, 0.1, 1000)

    YSCALE=1e3

    omega = np.pi/2
    ecc = 0.3
    lc0 = get_lc(0, 0)*YSCALE

    eccs = np.linspace(0,0.9,20)
    light_curves = []
    resids = []
    times = []
    for ecc in eccs:
        light_curves.append(get_lc(ecc, omega)*YSCALE)
        resids.append((get_lc(ecc, omega)-lc0/YSCALE)*YSCALE)
        times.append(t)

    #
    # make the plot!
    #
    set_style()
    fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(5,7), sharex=True)

    axs[0].plot(t, lc0, color="k", lw=2, zorder=10)

    linecollection0 = multiline(
        times, light_curves, eccs, cmap='viridis',
        ax=axs[0], lw=1
    )
    linecollection1 = multiline(
        times, resids, eccs, cmap='viridis',
        ax=axs[1], lw=1
    )

    axs[0].set_title(f"omega = {omega:.3f}")
    axs[0].set_ylabel("flux [ppt]")
    axs[1].set_ylabel("resid [ppt]")
    axs[1].set_xlabel("t [days]")

    axins1 = inset_axes(axs[0], width="5%", height="30%", loc='lower right',
                        borderpad=1.5)
    cb = fig.colorbar(linecollection0, cax=axins1, orientation="vertical")
    cb.ax.tick_params(labelsize='xx-small')
    cb.ax.set_title('ecc', fontsize='xx-small')
    cb.ax.tick_params(size=0, which='both') # remove the ticks
    axins1.xaxis.set_ticks_position("bottom")

    for a in axs:
        a.set_xlim(t.min(), t.max())

    savefig(fig, 'test_ecc_tuning.png', dpi=400)


if __name__ == "__main__":
    do_omega_plot()
    do_ecc_plot()
