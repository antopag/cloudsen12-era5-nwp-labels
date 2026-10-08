"""
Script 18: intervalli di confidenza bootstrap (95%) per le statistiche di accordo
ERA5/GFS vs CloudSEN12+ riportate nel paper (risposta al reviewer: "report uncertainty").

Per ogni strato: r di Pearson, MBE, mediana di Delta, MAE, RMSE con IC 95% percentile
(B = 2000 ricampionamenti con reinserimento).

Strati: globale, label_type, anno, tipo di nube dominante, GFS (81) e ERA5 sugli stessi 81.
Output: era5_data/bootstrap_ci.csv + tabella LaTeX su stdout.
"""
import numpy as np
import pandas as pd

B = 2000
RNG = np.random.default_rng(42)


def stats(p, o):
    d = p - o
    r = np.corrcoef(p, o)[0, 1] if len(p) > 2 and p.std() > 0 and o.std() > 0 else np.nan
    return dict(r=r, MBE=d.mean(), Median=np.median(d), MAE=np.abs(d).mean(), RMSE=np.sqrt((d ** 2).mean()))


def boot(p, o, B=B):
    n = len(p)
    point = stats(p, o)
    idx = RNG.integers(0, n, size=(B, n))
    acc = {k: [] for k in point}
    for row in idx:
        s = stats(p[row], o[row])
        for k in acc:
            acc[k].append(s[k])
    out = {}
    for k in point:
        lo, hi = np.nanpercentile(acc[k], [2.5, 97.5])
        out[k] = (point[k], lo, hi)
    out['N'] = n
    return out


def dominant(df):
    cols = ['cloudsen_clear_pct', 'cloudsen_thick_cloud_pct', 'cloudsen_thin_cloud_pct', 'cloudsen_shadow_pct']
    names = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']
    return pd.Series(np.array(names)[df[cols].values.argmax(axis=1)], index=df.index)


def main():
    era = pd.read_csv('era5_data/era5_vs_cloudsen12.csv')
    gfs = pd.read_csv('gfs_data/gfs_vs_cloudsen12_full.csv')
    era['dominant'] = dominant(era)
    rows = []

    def add(name, p, o):
        r = boot(np.asarray(p, float), np.asarray(o, float))
        rows.append(dict(stratum=name, **{f'{k}_{t}': v for k in ['r', 'MBE', 'Median', 'MAE', 'RMSE']
                                          for t, v in zip(['pt', 'lo', 'hi'], r[k])}, N=r['N']))
        print(f"{name:<28} N={r['N']:>6} " + ' '.join(
            f"{k}={r[k][0]:6.3f} [{r[k][1]:6.3f},{r[k][2]:6.3f}]" if k == 'r' else
            f"{k}={r[k][0]:5.1f} [{r[k][1]:5.1f},{r[k][2]:5.1f}]" for k in ['r', 'MBE', 'Median', 'MAE', 'RMSE']))

    P, O = 'era5_tcc_pct', 'cloudsen_total_cloud_pct'
    print('=== ERA5 globale'); add('ERA5 all', era[P], era[O])
    print('=== per label type')
    for lt in ['high', 'scribble', 'nolabel']:
        s = era[era.label_type == lt]; add(f'ERA5 {lt}', s[P], s[O])
    print('=== per anno')
    for y in sorted(era.year.unique()):
        s = era[era.year == y]; add(f'ERA5 {y}', s[P], s[O])
    print('=== per tipo dominante')
    for c in ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']:
        s = era[era.dominant == c]; add(f'ERA5 dom {c}', s[P], s[O])
    print('=== GFS (81) e ERA5 sugli stessi campioni')
    add('GFS 81', gfs['gfs_tcdc_pct'], gfs[O])
    key = ['roi_id', 's2_date']
    m = gfs[key].merge(era, on=key, how='inner')
    print(f'  ERA5 matchati sui campioni GFS: {len(m)}/81')
    if len(m) > 10:
        add('ERA5 same 81', m[P], m[O])

    pd.DataFrame(rows).to_csv('era5_data/bootstrap_ci.csv', index=False)
    print('\nSalvato era5_data/bootstrap_ci.csv')


if __name__ == '__main__':
    main()
