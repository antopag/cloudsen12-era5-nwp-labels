"""
Script 8: Download ERA5 cloud cover lungo la traiettoria SPB2
Metodo indipendente da GFS, da confrontare con il pipeline NCL e con CloudSEN12+.

Prerequisiti (una tantum):
  1) Account gratuito su https://cds.climate.copernicus.eu/
  2) Copiare API key in ~/.cdsapirc (su Windows: C:\\Users\\<utente>\\.cdsapirc)
     con il formato:
        url: https://cds.climate.copernicus.eu/api
        key: <UID>:<APIKEY>
  3) pip install cdsapi xarray netCDF4

Output:
  DATA/ERA5/era5_cloud_<tag>.nc    cube ERA5 scaricato (uno o due, se si attraversa il 180)
  DATA/ERA5/trajectory_cloud_era5.csv   cloud cover interpolata a ogni punto traiettoria
  DATA/ERA5/trajectory_cloud_era5.png   serie temporale lungo la traiettoria
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cdsapi

# ============================================================
# CONFIG
# ============================================================
TRAJ_FILE = r"G:\Shared drives\Clouds\Silvia\posizioni_SPB2-ballon-p1.dat_orig"
OUT_DIR   = r"G:\Shared drives\Clouds\DATA\ERA5"

LAT_BUFFER = 2.0    # gradi di margine attorno alla traiettoria
LON_BUFFER = 2.0

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1) LETTURA TRAIETTORIA
# ============================================================
# Formato: MM/DD/YY  HH:MM:SS  lon  lat  alt_ft  v1  v2
cols = ['date', 'time', 'lon', 'lat', 'alt_ft', 'v1', 'v2']
traj = pd.read_csv(TRAJ_FILE, sep=r'\s+', header=None, names=cols, engine='python')
traj['datetime'] = pd.to_datetime(traj['date'] + ' ' + traj['time'],
                                  format='%m/%d/%y %H:%M:%S', utc=True)
traj['alt_m'] = traj['alt_ft'] * 0.3048

print("=" * 60)
print(f"Traiettoria: {len(traj)} punti")
print(f"  t: {traj['datetime'].min()} -> {traj['datetime'].max()}")
print(f"  lat: [{traj['lat'].min():.2f}, {traj['lat'].max():.2f}]")
print(f"  lon: [{traj['lon'].min():.2f}, {traj['lon'].max():.2f}]")
print(f"  alt: [{traj['alt_m'].min():.0f}, {traj['alt_m'].max():.0f}] m")
print("=" * 60)

# ============================================================
# 2) DETERMINA PARAMETRI RICHIESTA CDS
# ============================================================
t_start = traj['datetime'].min().floor('1h')
t_end   = traj['datetime'].max().ceil('1h')
hours   = pd.date_range(t_start, t_end, freq='1h', tz='UTC')

years  = sorted({h.strftime('%Y') for h in hours})
months = sorted({h.strftime('%m') for h in hours})
days   = sorted({h.strftime('%d') for h in hours})
times  = sorted({h.strftime('%H:00') for h in hours})

lat_max = float(traj['lat'].max() + LAT_BUFFER)   # nord (piu alto)
lat_min = float(traj['lat'].min() - LAT_BUFFER)   # sud  (piu basso)

# La traiettoria attraversa il 180? lon in [-180,180], se il range > 180 si
# scavalca il meridiano di cambio data.
lon_min_raw = float(traj['lon'].min())
lon_max_raw = float(traj['lon'].max())
crosses_dateline = (lon_max_raw - lon_min_raw) > 180.0

if crosses_dateline:
    east_lons = traj.loc[traj['lon'] > 0, 'lon']
    west_lons = traj.loc[traj['lon'] < 0, 'lon']
    # Area ERA5: [North, West, South, East]
    area_east = [lat_max, float(east_lons.min() - LON_BUFFER), lat_min,  180.0]
    area_west = [lat_max, -180.0, lat_min, float(west_lons.max() + LON_BUFFER)]
    areas = [('east', area_east), ('west', area_west)]
    print("Traiettoria attraversa il 180E/W -> due richieste CDS")
else:
    area = [lat_max, lon_min_raw - LON_BUFFER, lat_min, lon_max_raw + LON_BUFFER]
    areas = [('all', area)]

# ============================================================
# 3) DOWNLOAD ERA5 (single levels)
# ============================================================
VARIABLES = [
    'total_cloud_cover',
    'high_cloud_cover',
    'medium_cloud_cover',
    'low_cloud_cover',
]

client = cdsapi.Client()
cubes = {}

for tag, area in areas:
    print(f"\n[{tag}] area N={area[0]:.2f} W={area[1]:.2f} S={area[2]:.2f} E={area[3]:.2f}")
    out_nc = os.path.join(OUT_DIR, f'era5_cloud_{tag}.nc')

    if not os.path.exists(out_nc):
        request = {
            'product_type': ['reanalysis'],
            'variable': VARIABLES,
            'year':  years,
            'month': months,
            'day':   days,
            'time':  times,
            'area':  area,                 # [N, W, S, E]
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        }
        print(f"  richiesta CDS in corso (puo richiedere qualche minuto)...")
        client.retrieve('reanalysis-era5-single-levels', request, out_nc)
        print(f"  salvato: {out_nc}")
    else:
        print(f"  gia presente: {out_nc} (skip download)")

    cubes[tag] = xr.open_dataset(out_nc)

# ============================================================
# 4) MERGE (se attraversa il 180) E NORMALIZZAZIONE COORDINATE
# ============================================================
if crosses_dateline:
    east = cubes['east']
    west = cubes['west'].assign_coords(longitude=(cubes['west']['longitude'] + 360.0))
    ds = xr.concat([east, west], dim='longitude').sortby('longitude')
    # rimuovi eventuali duplicati sul bordo 180
    _, unique_idx = np.unique(ds['longitude'].values, return_index=True)
    ds = ds.isel(longitude=np.sort(unique_idx))
else:
    ds = cubes['all']

# ERA5 nuovo CDS usa 'valid_time'; il vecchio 'time'
if 'valid_time' in ds.dims:
    ds = ds.rename({'valid_time': 'time'})

print(f"\nCubo ERA5 unito: dims={dict(ds.sizes)}")
print(f"Variabili: {list(ds.data_vars)}")

# ============================================================
# 5) INTERPOLAZIONE LUNGO TRAIETTORIA
# ============================================================
# Se siamo in convenzione 0..360 per effetto dello shift, adegua le lon della traiettoria
traj_lon = traj['lon'].to_numpy().astype(float)
if crosses_dateline:
    traj_lon = np.where(traj_lon < 0, traj_lon + 360.0, traj_lon)

t_da  = xr.DataArray(traj['datetime'].to_numpy().astype('datetime64[ns]'), dims='point')
la_da = xr.DataArray(traj['lat'].to_numpy(), dims='point')
lo_da = xr.DataArray(traj_lon, dims='point')

interp = ds.interp(time=t_da, latitude=la_da, longitude=lo_da, method='linear')

# ============================================================
# 6) SALVATAGGIO CSV + PLOT
# ============================================================
out = traj[['date', 'time', 'datetime', 'lon', 'lat', 'alt_ft', 'alt_m']].copy()

# ERA5 cloud cover e' in frazione [0..1] -> converti in %
name_map = {
    'tcc': 'total_cloud_pct',
    'hcc': 'high_cloud_pct',
    'mcc': 'medium_cloud_pct',
    'lcc': 'low_cloud_pct',
}
for short, col in name_map.items():
    if short in interp.data_vars:
        out[col] = interp[short].values * 100.0

out_csv = os.path.join(OUT_DIR, 'trajectory_cloud_era5.csv')
out.to_csv(out_csv, index=False)
print(f"\nSalvato: {out_csv}")
print(out.head())

fig, ax = plt.subplots(figsize=(11, 5))
if 'total_cloud_pct'  in out: ax.plot(out['datetime'], out['total_cloud_pct'],  label='Total',  color='black')
if 'high_cloud_pct'   in out: ax.plot(out['datetime'], out['high_cloud_pct'],   label='High',   color='tab:blue')
if 'medium_cloud_pct' in out: ax.plot(out['datetime'], out['medium_cloud_pct'], label='Medium', color='tab:orange')
if 'low_cloud_pct'    in out: ax.plot(out['datetime'], out['low_cloud_pct'],    label='Low',    color='tab:green')
ax.set_xlabel('UTC time')
ax.set_ylabel('Cloud cover (%)')
ax.set_ylim(0, 105)
ax.set_title('ERA5 cloud cover along SPB2 trajectory')
ax.legend(loc='best')
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()

out_png = os.path.join(OUT_DIR, 'trajectory_cloud_era5.png')
fig.savefig(out_png, dpi=150)
print(f"Salvato: {out_png}")
plt.show()

print("\n" + "=" * 60)
print("FATTO")
print("=" * 60)
