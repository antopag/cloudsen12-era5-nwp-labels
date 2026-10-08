"""
Script 8: Download dati GFS corrispondenti ai samples CloudSEN12+
Scarica Total Cloud Cover (TCDC) dal GFS su AWS per confronto

Prerequisiti:
    pip install boto3 cfgrib xarray eccodes
    oppure: conda install -c conda-forge cfgrib eccodes xarray boto3
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import urllib.request
import struct

# ==============================================================
# CONFIGURAZIONE
# ==============================================================

# Cartelle
DATA_DIR = "DATA"
OUTPUT_DIR = "gfs_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# URL base GFS su AWS (HTTPS, no boto3 richiesto)
# Formato: https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF
GFS_BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# ==============================================================
# FUNZIONI
# ==============================================================

def find_nearest_gfs_cycle(s2_datetime):
    """
    Trova il ciclo GFS piu vicino al passaggio Sentinel-2.
    GFS ha 4 cicli al giorno: 00, 06, 12, 18 UTC
    Usiamo f000 (analisi) del ciclo piu vicino.
    """
    dt = datetime.fromisoformat(s2_datetime.replace('Z', '+00:00'))
    hour = dt.hour

    # Trova ciclo piu vicino
    cycles = [0, 6, 12, 18]
    closest_cycle = min(cycles, key=lambda c: min(abs(hour - c), 24 - abs(hour - c)))

    # Se il ciclo e' nel futuro rispetto all'ora S2, usa il ciclo precedente
    # (i dati GFS sono disponibili ~3.5h dopo il ciclo)
    if closest_cycle > hour:
        idx = cycles.index(closest_cycle)
        closest_cycle = cycles[idx - 1]  # ciclo precedente

    gfs_date = dt.strftime('%Y%m%d')
    gfs_cycle = f"{closest_cycle:02d}"

    return gfs_date, gfs_cycle, dt


def build_gfs_url(gfs_date, gfs_cycle, forecast_hour=0):
    """Costruisce URL per scaricare il file GFS GRIB2"""
    fhh = f"{forecast_hour:03d}"
    url = f"{GFS_BASE_URL}/gfs.{gfs_date}/{gfs_cycle}/atmos/gfs.t{gfs_cycle}z.pgrb2.0p25.f{fhh}"
    return url


def build_gfs_idx_url(gfs_date, gfs_cycle, forecast_hour=0):
    """Costruisce URL per il file indice (.idx) del GRIB2"""
    return build_gfs_url(gfs_date, gfs_cycle, forecast_hour) + ".idx"


def parse_idx(idx_content):
    """
    Parsa il file .idx del GRIB2 per trovare offset dei campi.
    Restituisce lista di (offset, end_offset, field_info)
    """
    lines = idx_content.strip().split('\n')
    entries = []
    for line in lines:
        parts = line.split(':')
        if len(parts) >= 7:
            offset = int(parts[1])
            field = parts[3]
            level = parts[4]
            entries.append({
                'offset': offset,
                'field': field,
                'level': level,
                'line': line
            })

    # Calcola end offset
    for i in range(len(entries) - 1):
        entries[i]['end_offset'] = entries[i + 1]['offset']
    if entries:
        entries[-1]['end_offset'] = None  # ultimo campo

    return entries


def download_tcdc_field(gfs_date, gfs_cycle, forecast_hour=0):
    """
    Scarica solo il campo TCDC (Total Cloud Cover) dal file GFS.
    Usa range request per scaricare solo il campo necessario (pochi KB invece di ~300MB).
    """
    idx_url = build_gfs_idx_url(gfs_date, gfs_cycle, forecast_hour)
    grib_url = build_gfs_url(gfs_date, gfs_cycle, forecast_hour)

    # Scarica indice
    try:
        req = urllib.request.Request(idx_url)
        with urllib.request.urlopen(req, timeout=30) as response:
            idx_content = response.read().decode('utf-8')
    except Exception as e:
        print(f"    Errore download indice: {e}")
        return None

    # Cerca campo TCDC (entire atmosphere o cloud layer)
    entries = parse_idx(idx_content)
    tcdc_entries = [e for e in entries if 'TCDC' in e['field'] and 'entire atmosphere' in e['level']]

    if not tcdc_entries:
        # Prova con altri livelli TCDC
        tcdc_entries = [e for e in entries if 'TCDC' in e['field']]

    if not tcdc_entries:
        print(f"    Campo TCDC non trovato nell'indice")
        return None

    tcdc = tcdc_entries[0]
    start = tcdc['offset']
    end = tcdc['end_offset']

    # Scarica solo il campo TCDC con range request
    try:
        if end:
            range_header = f"bytes={start}-{end - 1}"
        else:
            range_header = f"bytes={start}-"

        req = urllib.request.Request(grib_url, headers={'Range': range_header})
        with urllib.request.urlopen(req, timeout=60) as response:
            grib_data = response.read()

        return grib_data

    except Exception as e:
        print(f"    Errore download TCDC: {e}")
        return None


def extract_tcdc_at_point(grib_data, target_lat, target_lon, gfs_date, gfs_cycle):
    """
    Estrae il valore TCDC alla coordinata specificata dal campo GRIB2.
    Prova prima con cfgrib, poi con fallback manuale.
    """
    # Salva temporaneamente il campo GRIB
    tmp_path = os.path.join(OUTPUT_DIR, "_tmp_tcdc.grib2")
    with open(tmp_path, 'wb') as f:
        f.write(grib_data)

    try:
        import xarray as xr
        ds = xr.open_dataset(tmp_path, engine='cfgrib')

        # Trova la variabile TCDC
        var_name = None
        for name in ds.data_vars:
            if 'tcc' in name.lower() or 'cloud' in name.lower() or name == 'tcc':
                var_name = name
                break
        if var_name is None:
            var_name = list(ds.data_vars)[0]

        data = ds[var_name]

        # GFS lon va da 0 a 360
        gfs_lon = target_lon if target_lon >= 0 else target_lon + 360

        # Nearest neighbor
        value = float(data.sel(latitude=target_lat, longitude=gfs_lon, method='nearest').values)

        ds.close()
        os.remove(tmp_path)
        return value

    except ImportError:
        print("    cfgrib non disponibile, uso fallback semplificato")
        os.remove(tmp_path)
        return extract_tcdc_fallback(grib_data, target_lat, target_lon)

    except Exception as e:
        print(f"    Errore lettura GRIB: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None


def extract_tcdc_fallback(grib_data, target_lat, target_lon):
    """
    Fallback: salva il campo GRIB e prova con pygrib.
    Se neanche pygrib funziona, restituisce None.
    """
    try:
        import pygrib
        tmp_path = os.path.join(OUTPUT_DIR, "_tmp_tcdc.grib2")
        with open(tmp_path, 'wb') as f:
            f.write(grib_data)

        grbs = pygrib.open(tmp_path)
        grb = grbs[1]  # primo messaggio

        # GFS lon va da 0 a 360
        gfs_lon = target_lon if target_lon >= 0 else target_lon + 360

        lats, lons = grb.latlons()
        data = grb.values

        # Trova punto piu vicino
        dist = (lats - target_lat)**2 + (lons - gfs_lon)**2
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        value = float(data[idx])

        grbs.close()
        os.remove(tmp_path)
        return value

    except ImportError:
        print("    ATTENZIONE: serve cfgrib o pygrib per leggere i dati GRIB2")
        print("    Installa: conda install -c conda-forge cfgrib eccodes")
        return None

    except Exception as e:
        print(f"    Errore fallback: {e}")
        return None


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("DOWNLOAD DATI GFS PER CONFRONTO CON CloudSEN12+")
    print("=" * 60)

    # Carica samples
    all_samples = []

    for year in [2021, 2022]:
        csv_path = os.path.join(DATA_DIR, f"samples_{year}_with_location.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(DATA_DIR, f"samples_{year}_global.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['year'] = year
            all_samples.append(df)
            print(f"  {year}: {len(df)} samples da {csv_path}")
        else:
            print(f"  {year}: nessun file trovato")

    if not all_samples:
        print("\nNessun sample trovato! Verifica la cartella DATA/")
        return

    df_all = pd.concat(all_samples, ignore_index=True)
    print(f"\nTotale samples: {len(df_all)}")

    # Cache per evitare download duplicati dello stesso ciclo GFS
    grib_cache = {}

    results = []
    errors = []

    print("\n" + "=" * 60)
    print("DOWNLOAD E ESTRAZIONE TCDC")
    print("=" * 60)

    for i, row in df_all.iterrows():
        s2_date = row['s2_date']
        lon = row['lon']
        lat = row['lat']
        roi_id = row['roi_id']

        gfs_date, gfs_cycle, s2_dt = find_nearest_gfs_cycle(s2_date)
        cache_key = f"{gfs_date}_{gfs_cycle}"

        print(f"\n[{i+1}/{len(df_all)}] {roi_id} | S2: {s2_date[:16]} | GFS: {gfs_date} {gfs_cycle}z")
        print(f"    Coords: ({lat:.2f}, {lon:.2f})")

        # Scarica campo TCDC (o usa cache)
        if cache_key not in grib_cache:
            grib_data = download_tcdc_field(gfs_date, gfs_cycle)
            grib_cache[cache_key] = grib_data
        else:
            grib_data = grib_cache[cache_key]
            print(f"    (cache hit: {cache_key})")

        if grib_data is None:
            print(f"    SKIP: dati GFS non disponibili")
            errors.append({
                'roi_id': roi_id,
                's2_date': s2_date,
                'error': f'GFS data not available for {gfs_date} {gfs_cycle}z'
            })
            continue

        # Estrai valore TCDC alla coordinata
        tcdc_value = extract_tcdc_at_point(grib_data, lat, lon, gfs_date, gfs_cycle)

        if tcdc_value is not None:
            print(f"    GFS TCDC: {tcdc_value:.1f}%")

            # Calcola cloud cover da CloudSEN12
            cloudsen_cloud_pct = row.get('thick_cloud_pct', 0) + row.get('thin_cloud_pct', 0)
            print(f"    CloudSEN12 cloud: {cloudsen_cloud_pct:.1f}%")
            print(f"    Differenza: {abs(tcdc_value - cloudsen_cloud_pct):.1f}%")

            result = {
                'roi_id': roi_id,
                's2_date': s2_date,
                'year': row.get('year', ''),
                'lon': lon,
                'lat': lat,
                'country': row.get('country', ''),
                'city': row.get('city', ''),
                'gfs_date': gfs_date,
                'gfs_cycle': gfs_cycle,
                'gfs_tcdc_pct': tcdc_value,
                'cloudsen_clear_pct': row.get('clear_pct', 0),
                'cloudsen_thick_cloud_pct': row.get('thick_cloud_pct', 0),
                'cloudsen_thin_cloud_pct': row.get('thin_cloud_pct', 0),
                'cloudsen_shadow_pct': row.get('shadow_pct', 0),
                'cloudsen_total_cloud_pct': cloudsen_cloud_pct,
                'difference': tcdc_value - cloudsen_cloud_pct,
            }
            results.append(result)
        else:
            errors.append({
                'roi_id': roi_id,
                's2_date': s2_date,
                'error': 'TCDC extraction failed'
            })

    # ==============================================================
    # SALVA RISULTATI
    # ==============================================================

    print("\n" + "=" * 60)
    print("RISULTATI")
    print("=" * 60)

    if results:
        df_results = pd.DataFrame(results)
        output_path = os.path.join(OUTPUT_DIR, "gfs_vs_cloudsen12.csv")
        df_results.to_csv(output_path, index=False)
        print(f"\nSalvato: {output_path} ({len(df_results)} confronti)")

        # Statistiche di confronto
        diff = df_results['difference']
        print(f"\n--- STATISTICHE DI CONFRONTO ---")
        print(f"  Samples analizzati:  {len(df_results)}")
        print(f"  Differenza media:    {diff.mean():.1f}%")
        print(f"  Differenza mediana:  {diff.median():.1f}%")
        print(f"  Deviazione standard: {diff.std():.1f}%")
        print(f"  RMSE:                {np.sqrt((diff**2).mean()):.1f}%")
        print(f"  MAE:                 {diff.abs().mean():.1f}%")
        print(f"  Min differenza:      {diff.min():.1f}%")
        print(f"  Max differenza:      {diff.max():.1f}%")

        # Correlazione
        corr = df_results['gfs_tcdc_pct'].corr(df_results['cloudsen_total_cloud_pct'])
        print(f"  Correlazione:        {corr:.3f}")

        # Distribuzione per range
        print(f"\n--- DISTRIBUZIONE DIFFERENZE ---")
        bins = [0, 10, 20, 30, 50, 100]
        for j in range(len(bins) - 1):
            mask = (diff.abs() >= bins[j]) & (diff.abs() < bins[j + 1])
            count = mask.sum()
            pct = 100 * count / len(diff)
            print(f"  |diff| {bins[j]:>3}-{bins[j+1]:>3}%: {count:>3} ({pct:>5.1f}%)")

    if errors:
        df_errors = pd.DataFrame(errors)
        error_path = os.path.join(OUTPUT_DIR, "gfs_errors.csv")
        df_errors.to_csv(error_path, index=False)
        print(f"\nErrori: {len(errors)} (salvati in {error_path})")

    print(f"\nCompletamenti: {len(results)}/{len(df_all)}")


if __name__ == '__main__':
    main()
