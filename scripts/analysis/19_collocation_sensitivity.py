"""
Script 19: sensibilita' della collocazione spaziale ERA5 (risposta al reviewer).

Confronta, sugli stessi 48,380 campioni:
  - nearest : valore al nodo 0.25 deg piu' vicino al centroide (metodo del paper)
  - bilinear: interpolazione bilineare dei 4 nodi attorno al centroide
  - mean3x3 : media dei 9 nodi attorno al nodo piu' vicino (~0.75 deg, upper bound
              dell'effetto di "area averaging")
I file locali era5_data/netcdf/era5_tcc_YYYYMMDD.nc contengono UNA sola ora per data (quella del primo
campione scaricato), quindi l'analisi e' ristretta ai campioni la cui ora di acquisizione (arrotondata)
coincide con quella del campo salvato (~2,800 su 48,380); su questi il 'nearest' riproduce il CSV del paper.

Output: era5_data/collocation_sensitivity.csv (per campione) e statistiche + IC bootstrap a stdout.
"""
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

NC_DIR = 'era5_data/netcdf'
CSV = 'era5_data/era5_vs_cloudsen12.csv'
OUT = 'era5_data/collocation_sensitivity.csv'
B = 2000


def stats(p, o):
    d = p - o
    return dict(r=np.corrcoef(p, o)[0, 1], MBE=d.mean(), Median=np.median(d),
                MAE=np.abs(d).mean(), RMSE=np.sqrt((d ** 2).mean()))


def boot_ci(p, o, rng):
    n = len(p)
    acc = {k: [] for k in ['r', 'MBE', 'Median', 'MAE', 'RMSE']}
    for _ in range(B):
        i = rng.integers(0, n, n)
        s = stats(p[i], o[i])
        for k in acc:
            acc[k].append(s[k])
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in acc.items()}


def main():
    df = pd.read_csv(CSV)
    df['date'] = df.s2_date.str[:10].str.replace('-', '')
    df['hour'] = pd.to_datetime(df.s2_date).dt.round('h').dt.hour
    lon360 = df.lon.values % 360.0
    lat = df.lat.values
    near = np.full(len(df), np.nan)
    bil = np.full(len(df), np.nan)
    m3 = np.full(len(df), np.nan)

    files = {os.path.basename(f)[9:17]: f for f in glob.glob(os.path.join(NC_DIR, 'era5_tcc_*.nc'))}
    print(f'{len(files)} campi ERA5, {df.date.nunique()} date nel CSV')
    missing = 0
    for k, (date, g) in enumerate(df.groupby('date')):
        f = files.get(date)
        if f is None:
            missing += len(g)
            continue
        ds = xr.open_dataset(f)
        # il file contiene UNA sola ora: usa solo i campioni la cui ora arrotondata coincide
        h_nc = pd.Timestamp(ds.valid_time.values[0]).hour
        g = g[g.hour == h_nc]
        if len(g) == 0:
            ds.close(); continue
        tcc = ds['tcc'].isel(valid_time=0) * 100.0          # -> %
        lats = ds.latitude.values                            # 90 .. -90 (decrescente)
        lons = ds.longitude.values                           # 0 .. 359.75
        A = tcc.values
        idx = g.index.values
        la, lo = lat[idx], lon360[idx]

        # nearest
        ii = np.rint((90.0 - la) / 0.25).astype(int).clip(0, len(lats) - 1)
        jj = np.rint(lo / 0.25).astype(int) % len(lons)
        near[idx] = A[ii, jj]

        # bilinear (griglia regolare 0.25, lon periodica)
        fi = (90.0 - la) / 0.25
        fj = lo / 0.25
        i0 = np.floor(fi).astype(int).clip(0, len(lats) - 2)
        j0 = np.floor(fj).astype(int) % len(lons)
        i1 = i0 + 1
        j1 = (j0 + 1) % len(lons)
        wi = fi - i0
        wj = fj - j0
        bil[idx] = ((1 - wi) * (1 - wj) * A[i0, j0] + (1 - wi) * wj * A[i0, j1]
                    + wi * (1 - wj) * A[i1, j0] + wi * wj * A[i1, j1])

        # media 3x3 attorno al nearest
        acc = np.zeros(len(idx))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                acc += A[(ii + di).clip(0, len(lats) - 1), (jj + dj) % len(lons)]
        m3[idx] = acc / 9.0
        ds.close()
        if k % 100 == 0:
            print(f'  {k}/{df.date.nunique()} date')

    print(f'campioni senza campo ERA5 locale: {missing}; campioni con ora coincidente col campo salvato: {np.isfinite(near).sum()}')
    df['tcc_nearest'] = near
    df['tcc_bilinear'] = bil
    df['tcc_mean3x3'] = m3
    df[['roi_id', 's2_date', 'lon', 'lat', 'era5_tcc_pct', 'tcc_nearest', 'tcc_bilinear', 'tcc_mean3x3',
        'cloudsen_total_cloud_pct']].to_csv(OUT, index=False)

    ok = df.dropna(subset=['tcc_nearest'])
    o = ok.cloudsen_total_cloud_pct.values.astype(float)
    print(f'\nN = {len(ok)}')
    print(f'check: |nearest - era5_tcc_pct(csv)| max = {np.abs(ok.tcc_nearest - ok.era5_tcc_pct).max():.3f}')
    rng = np.random.default_rng(42)
    rows = []
    for name, col in [('nearest (paper)', 'tcc_nearest'), ('bilinear', 'tcc_bilinear'), ('mean 3x3', 'tcc_mean3x3')]:
        p = ok[col].values.astype(float)
        s = stats(p, o)
        ci = boot_ci(p, o, rng)
        rows.append(dict(method=name, **s, **{f'{k}_lo': v[0] for k, v in ci.items()},
                         **{f'{k}_hi': v[1] for k, v in ci.items()}))
        print(f"{name:<16} " + ' '.join(
            f"{k}={s[k]:.3f}[{ci[k][0]:.3f},{ci[k][1]:.3f}]" if k == 'r' else
            f"{k}={s[k]:.1f}[{ci[k][0]:.1f},{ci[k][1]:.1f}]" for k in s))
    d = ok.tcc_bilinear - ok.tcc_nearest
    print(f'\nbilinear - nearest: mean {d.mean():.2f}, sd {d.std():.2f}, |d|>10%: {(np.abs(d) > 10).mean():.1%}')
    pd.DataFrame(rows).to_csv('era5_data/collocation_sensitivity_stats.csv', index=False)


if __name__ == '__main__':
    main()
