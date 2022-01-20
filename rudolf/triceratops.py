import numpy as np, pandas as pd, matplotlib.pyplot as plt
import time
from lightkurve import KeplerLightCurve
import matplotlib.pyplot as plt
import triceratops.triceratops as tr
import sys, os
from rudolf.paths import DATADIR, RESULTSDIR

print(sys.version)

import warnings
warnings.filterwarnings('ignore')

def run_triceratops_demo():
    # Quarter 3, Kepler-10b (KIC 11904151)
    ID = 11904151
    sectors = np.array([3])
    target = tr.target(ID=ID, sectors=sectors, mission="Kepler")

    # select aperture
    ap = np.array([            [655, 247],
                   [654, 246], [655, 246], [656, 246],
                               [655, 245]             ])

    OUTDIR = os.path.join(
        RESULTSDIR, 'validation_fpp_analysis', 'triceratops',
        f'kepler{ID}'
    )
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    outpath = os.path.join(OUTDIR, f'kepler{ID}_plot_field')
    target.plot_field(sector=3, ap_pixels=ap, save=True, fname=outpath)

    print('Stars within 40 arcsec')
    print(target.stars)

    apertures = np.array([ap])
    target.calc_depths(tdepth=0.00019, all_ap_pixels=apertures)

    print('Stars within 40 arcsec, depths calculated')
    print(target.stars)

    outpath = os.path.join(OUTDIR, f'kepler{ID}_triceratops_target_stars_depths.csv')
    target.stars.to_csv(outpath, index=False)
    print(f'Wrote {outpath}')

    # Phase-folded and binned light curve.
    # time (times from phase-folded light curve in units of days from transit
    # center), flux_0 (normalized flux from phase-folded light curve), flux_err_0
    # (flux error values of the target's phase-folded light curve), and P_orb
    # (orbital period of the TOI in days).

    lcpath = os.path.join(DATADIR, 'validation', 'Kepler10b_lightcurve.csv')

    lc = pd.read_csv(lcpath, header=None)
    time, flux, flux_err = lc[0].values, lc[1].values, lc[2].values
    P_orb = 0.837

    mask = ~np.isnan(flux)
    lc = KeplerLightCurve(time=time[mask], flux=flux[mask],
                          flux_err=flux_err[mask])

    fig, ax = plt.subplots(figsize=(4,3))
    lc.plot(ax=ax)
    outpath = os.path.join(OUTDIR, f'kepler{ID}_plot_lc_phasefold.png')
    fig.savefig(outpath, bbox_inches='tight', dpi=400)
    print(f'Wrote {outpath}')

    # calculate the fpp
    print(42*'.')
    print('Beginning single FPP calculation')

    target.calc_probs(time=lc.time.value, flux_0=lc.flux.value,
                      flux_err_0=np.mean(lc.flux_err.value), P_orb=P_orb,
                      parallel=True)

    df_results = target.probs
    print("FPP =", target.FPP)
    print("NFPP =", target.NFPP)
    print(df_results)

    outpath = os.path.join(OUTDIR, f'kepler{ID}_triceratops_fpp_results.csv')
    df_results.to_csv(outpath, index=False)
    print(f'Wrote {outpath}')

    # but do it N~=20 times, because that's necessary to get a viable mean and
    # stdev on the FPP
    N = 20
    outpath = os.path.join(OUTDIR, f'kepler{ID}_triceratops_fpp_results_N{N}.csv')

    if not os.path.exists(outpath):
        print(f'Beginning bulk N={N} FPP calculations')
        FPPs, NFPPs = np.zeros(N), np.zeros(N)
        for i in range(N):
            print(f'{i}/{N}...')
            target.calc_probs(time=lc.time.value, flux_0=lc.flux.value,
                              flux_err_0=np.mean(lc.flux_err.value), P_orb=P_orb,
                              parallel=True, verbose=0)
            FPPs[i] = target.FPP
            NFPPs[i] = target.NFPP

        outdf = pd.DataFrame({'fpp': FPPs, 'nfpp': NFPPs})
        outdf.to_csv(outpath, index=False)
        print(f'Wrote {outpath}')

    else:
        _df = pd.read_csv(outpath)
        FPPs = np.array(_df.fpp)
        NFPPs = np.array(_df.nfpp)

    meanFPP = np.round(np.mean(FPPs), 4)
    stdvFPP = np.round(np.std(FPPs), 4)
    l0 = f"FPP = {meanFPP} +/- {stdvFPP}"
    print(l0)

    meanNFPP = np.round(np.mean(NFPPs), 4)
    stdvNFPP = np.round(np.std(NFPPs), 4)
    l1 = f"NFPP = {meanNFPP} +/- {stdvNFPP}"
    print(l1)

    outpath = os.path.join(OUTDIR, f'kepler{ID}_triceratops_fpp_results_N{N}_log.txt')
    with open(outpath, 'w') as f:
        f.writelines([l0+'\n',l1])
    print(f'Wrote {outpath}')

    outpath = os.path.join(OUTDIR, f'kepler{ID}_plot_fits')

    target.plot_fits(time=lc.time.value, flux_0=lc.flux.value,
                     flux_err_0=np.mean(lc.flux_err.value), save=True,
                     fname=outpath)
