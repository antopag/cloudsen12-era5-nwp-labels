"""
Script 29: distribuzione ESATTA dello scarto temporale della collocazione ERA5.

Lo script 25 misurava la differenza fra l'ora di acquisizione ARROTONDATA e le ore
disponibili nel file, cioe' un numero intero di ore. Da li' veniva l'affermazione
contraddittoria "100% entro 1 ora, ma il 7.5% arriva a 90 minuti": uno scarto arrotondato
di 1 ora corrisponde a uno scarto reale fra 30 e 90 minuti. Qui si calcola lo scarto reale
in minuti fra l'istante di acquisizione e l'istante del campo ERA5 usato.
"""
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

CSV = os.path.join("era5_data", "era5_vs_cloudsen12.csv")
NC_DIR = os.path.join("era5_data", "netcdf")
OUT = os.path.join("era5_data", "temporal_offset.csv")


def main():
    d = pd.read_csv(CSV)
    d['t'] = pd.to_datetime(d.s2_date)
    d['date'] = d.t.dt.strftime('%Y%m%d')
    files = {os.path.basename(f)[9:17]: f for f in glob.glob(os.path.join(NC_DIR, 'era5_tcc_*.nc'))}

    offs = np.full(len(d), np.nan)
    for date, g in d.groupby('date'):
        f = files.get(date)
        if f is None:
            continue
        try:
            ds = xr.open_dataset(f)
            tc = 'valid_time' if 'valid_time' in ds.coords else 'time'
            hrs = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(ds[tc].values)))
            ds.close()
        except Exception:
            continue
        # scarto reale in minuti, vettorizzato: |acquisizioni - ore disponibili| via broadcasting
        ta = g.t.values.astype('datetime64[s]').astype('int64')          # (n,)
        tb = hrs.values.astype('datetime64[s]').astype('int64')          # (m,)
        dt = np.abs(ta[:, None] - tb[None, :]) / 60.0                    # minuti
        offs[g.index.values] = dt.min(axis=1)

    ok = np.isfinite(offs)
    o = offs[ok]
    print("collocazioni con campo disponibile: %d di %d" % (ok.sum(), len(d)))
    print("\nscarto reale |acquisizione - campo ERA5|, in minuti:")
    print("   media %.1f   mediana %.1f   p90 %.1f   massimo %.1f" % (o.mean(), np.median(o), np.percentile(o, 90), o.max()))
    for lim in (15, 30, 45, 60, 90):
        print("   entro %2d min: %5.1f%%" % (lim, 100 * (o <= lim).mean()))
    # quante usano l'ora piu' vicina in assoluto (cioe' <= 30 min)
    print("\nuso dell'ora piu' vicina (scarto <= 30 min): %.1f%%" % (100 * (o <= 30).mean()))
    print("uso dell'ora adiacente (30 < scarto <= 90 min): %.1f%%" % (100 * ((o > 30) & (o <= 90)).mean()))
    print("oltre 90 min: %.2f%%" % (100 * (o > 90).mean()))

    pd.DataFrame({'roi_id': d.roi_id[ok], 's2_date': d.s2_date[ok], 'offset_min': o}).to_csv(OUT, index=False)
    print("\nsalvato", OUT)


if __name__ == '__main__':
    main()
