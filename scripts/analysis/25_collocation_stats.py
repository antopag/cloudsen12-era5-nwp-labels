"""
Script 25: statistiche di collocazione per la revisione.

Risponde a tre richieste:
  * R2 punto 3 / editor: il bootstrap dei campioni singoli sottostima l'incertezza, perche'
    patch vicine condividono la stessa cella NWP. Qui si confronta il bootstrap per campione
    con quello RAGGRUPPATO per cella 0.25 deg e per regione di interesse.
  * editor: quante delle 48.380 collocazioni usano davvero l'ora piu' vicina e cosa usa il
    resto. Lo script 19 leggeva solo la prima ora di ogni file e si limitava a 1.077
    collocazioni non rappresentative.
  * R2 punto 8: terminologia e cifre coerenti; qui si producono i numeri definitivi.

Output: era5_data/collocation_stats_revision.csv e un riepilogo a schermo.
"""
import os
import glob
import numpy as np
import pandas as pd

CSV = os.path.join("era5_data", "era5_vs_cloudsen12.csv")
NC_DIR = os.path.join("era5_data", "netcdf")
OUT = os.path.join("era5_data", "collocation_stats_revision.csv")
B = 2000
RNG = np.random.default_rng(42)


def stats(p, o):
    d = p - o
    r = np.corrcoef(p, o)[0, 1] if p.std() > 0 and o.std() > 0 else np.nan
    from scipy.stats import spearmanr
    return dict(r=r, rho=spearmanr(p, o).statistic, MBE=d.mean(),
                Median=np.median(d), MAE=np.abs(d).mean(), RMSE=np.sqrt((d ** 2).mean()))


def boot(p, o, groups=None, B=B):
    """groups=None: ricampiona i singoli campioni. Altrimenti ricampiona i GRUPPI interi."""
    keys = ['r', 'rho', 'MBE', 'Median', 'MAE', 'RMSE']
    acc = {k: [] for k in keys}
    if groups is None:
        n = len(p)
        for _ in range(B):
            i = RNG.integers(0, n, n)
            s = stats(p[i], o[i])
            for k in keys:
                acc[k].append(s[k])
    else:
        uniq, inv = np.unique(groups, return_inverse=True)
        idx_by_g = [np.where(inv == g)[0] for g in range(len(uniq))]
        ng = len(uniq)
        for _ in range(B):
            gsel = RNG.integers(0, ng, ng)
            i = np.concatenate([idx_by_g[g] for g in gsel])
            s = stats(p[i], o[i])
            for k in keys:
                acc[k].append(s[k])
    return {k: np.nanpercentile(v, [2.5, 97.5]) for k, v in acc.items()}


def main():
    d = pd.read_csv(CSV)
    p = d.era5_tcc_pct.values.astype(float)
    o = d.cloudsen_total_cloud_pct.values.astype(float)
    pt = stats(p, o)

    # gruppi: cella ERA5 0.25 deg, e regione di interesse
    cell = (np.round(d.lat.values / 0.25).astype(int).astype(str) + '_' +
            np.round(d.lon.values / 0.25).astype(int).astype(str))
    roi = d.roi_id.values.astype(str)
    print("campioni %d | celle 0.25 deg distinte %d | ROI distinte %d"
          % (len(d), len(np.unique(cell)), len(np.unique(roi))))
    print("patch per cella: media %.1f, max %d" % (len(d) / len(np.unique(cell)),
                                                   pd.Series(cell).value_counts().max()))

    rows = []
    for name, g in [('per campione', None), ('per cella 0.25 deg', cell), ('per ROI', roi)]:
        ci = boot(p, o, g)
        print("\n--- bootstrap %s" % name)
        for k in ['r', 'rho', 'MBE', 'Median', 'MAE', 'RMSE']:
            w = ci[k][1] - ci[k][0]
            print("   %-7s %7.3f  IC [%7.3f, %7.3f]  ampiezza %.3f" % (k, pt[k], ci[k][0], ci[k][1], w))
            rows.append(dict(bootstrap=name, metric=k, value=pt[k], lo=ci[k][0], hi=ci[k][1], width=w))

    # confronto delle ampiezze
    df = pd.DataFrame(rows)
    piv = df.pivot(index='metric', columns='bootstrap', values='width')
    print("\n--- ampiezza dell'intervallo: quanto il raggruppamento allarga l'incertezza")
    for m in piv.index:
        a = piv.loc[m, 'per campione']
        print("   %-7s per campione %.3f | per cella %.3f (x%.1f) | per ROI %.3f (x%.1f)"
              % (m, a, piv.loc[m, 'per cella 0.25 deg'], piv.loc[m, 'per cella 0.25 deg'] / a,
                 piv.loc[m, 'per ROI'], piv.loc[m, 'per ROI'] / a))

    # ---- scarto temporale effettivo su TUTTE le collocazioni disponibili
    print("\n--- scarto temporale effettivo (editor)")
    d['date'] = d.s2_date.str[:10].str.replace('-', '')
    d['hour'] = pd.to_datetime(d.s2_date).dt.round('h').dt.hour
    files = {os.path.basename(f)[9:17]: f for f in glob.glob(os.path.join(NC_DIR, 'era5_tcc_*.nc'))}
    import xarray as xr
    offs, checked = [], 0
    for date, g in d.groupby('date'):
        f = files.get(date)
        if f is None:
            continue
        try:
            ds = xr.open_dataset(f)
            tc = 'valid_time' if 'valid_time' in ds.coords else 'time'
            hrs = np.atleast_1d(pd.to_datetime(ds[tc].values).hour)
            ds.close()
        except Exception:
            continue
        for h in g.hour.values:
            offs.append(int(np.min(np.abs(hrs - h))))
        checked += len(g)
    offs = np.array(offs)
    if len(offs):
        print("   collocazioni verificate: %d di %d (%.0f%%)" % (checked, len(d), 100 * checked / len(d)))
        for lim in (0, 1, 2):
            print("   scarto <= %d h: %.1f%%" % (lim, 100 * (offs <= lim).mean()))
        print("   scarto massimo: %d h" % offs.max())
        rows.append(dict(bootstrap='scarto temporale', metric='frac_ora_esatta',
                         value=float((offs == 0).mean()), lo=np.nan, hi=np.nan, width=np.nan))

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("\nsalvato", OUT)


if __name__ == '__main__':
    main()
