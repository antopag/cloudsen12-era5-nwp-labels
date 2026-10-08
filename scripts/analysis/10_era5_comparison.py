"""
Script 10: Confronto ERA5 vs CloudSEN12+ su larga scala
Scarica Total Cloud Cover (TCC) da ERA5 per TUTTI i samples

ERA5: reanalisi ECMWF, oraria, 0.25 gradi, 1979-oggi
Copre tutti i samples 2018-2022 (~49.000)

Prerequisiti:
    1. Account Copernicus CDS: https://cds.climate.copernicus.eu
    2. pip install cdsapi xarray netcdf4
    3. Configurare ~/.cdsapirc con la tua API key:
        url: https://cds.climate.copernicus.eu/api
        key: <your-uid>:<your-api-key>
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict

# ==============================================================
# CONFIGURAZIONE
# ==============================================================

OUTPUT_DIR = "era5_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "netcdf"), exist_ok=True)

SAMPLES_CACHE = os.path.join(OUTPUT_DIR, "all_samples.csv")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "era5_vs_cloudsen12.csv")

# ==============================================================
# FASE 1: Estrazione TUTTI i samples dal dataset
# ==============================================================

def extract_all_samples():
    """Estrae tutti i samples con coordinate dal dataset CloudSEN12+."""

    if os.path.exists(SAMPLES_CACHE):
        df = pd.read_csv(SAMPLES_CACHE)
        print(f"  Caricati {len(df)} samples da cache")
        return df

    print("\nCaricamento dataset da HuggingFace...")
    dataset = load_dataset("isp-uv-es/CloudSEN12Plus")

    def extract_coords(centroid_str):
        if centroid_str and "POINT" in str(centroid_str):
            match = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', str(centroid_str))
            if match:
                return float(match.group(1)), float(match.group(2))
        return None, None

    samples = []

    # Processa tutti gli split
    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            continue

        print(f"\nScansione split '{split_name}' ({len(dataset[split_name])} samples)...")

        for i, sample in enumerate(dataset[split_name]):
            if i % 5000 == 0:
                print(f"  {i}/{len(dataset[split_name])} processati...")

            s2_date = sample.get('s2_date', '')
            if not s2_date:
                continue

            centroid = sample.get('proj_centroid', '')
            lon, lat = extract_coords(centroid)
            if lon is None:
                continue

            year_match = re.match(r'(\d{4})', str(s2_date))
            year = int(year_match.group(1)) if year_match else 0

            samples.append({
                'split': split_name,
                'index': i,
                'roi_id': sample.get('roi_id', ''),
                's2_id_gee': sample.get('s2_id_gee', ''),
                's2_date': s2_date,
                'year': year,
                'lon': lon,
                'lat': lat,
                'label_type': sample.get('label_type', ''),
                'clear_pct': sample.get('clear_percentage', 0),
                'thick_cloud_pct': sample.get('thick_percentage', 0),
                'thin_cloud_pct': sample.get('thin_percentage', 0),
                'shadow_pct': sample.get('cloud_shadow_percentage', 0),
            })

    df = pd.DataFrame(samples)
    df.to_csv(SAMPLES_CACHE, index=False)

    print(f"\nTotale samples: {len(df)}")
    print(f"\n  Per anno:")
    print(df['year'].value_counts().sort_index().to_string())
    print(f"\n  Per split:")
    print(df['split'].value_counts().to_string())
    print(f"\n  Per label_type:")
    print(df['label_type'].value_counts().to_string())
    print(f"\nSalvato: {SAMPLES_CACHE}")

    return df


# ==============================================================
# FASE 2: Download ERA5 - per mese (efficiente)
# ==============================================================

def download_era5_month(year, month):
    """
    Scarica ERA5 TCC per un mese intero su griglia globale.
    Un file NetCDF per mese (~50-100MB) con dati orari.
    """
    import cdsapi

    nc_path = os.path.join(OUTPUT_DIR, "netcdf", f"era5_tcc_{year}{month:02d}.nc")

    if os.path.exists(nc_path):
        print(f"    {nc_path} gia scaricato")
        return nc_path

    print(f"    Download ERA5 TCC {year}-{month:02d}...")

    c = cdsapi.Client()

    # Calcola giorni nel mese
    if month == 12:
        n_days = 31
    else:
        from calendar import monthrange
        n_days = monthrange(year, month)[1]

    days = [f"{d:02d}" for d in range(1, n_days + 1)]

    # Scarica solo le ore necessarie (ogni 1h)
    # Per efficienza, scarichiamo tutte le ore
    hours = [f"{h:02d}:00" for h in range(24)]

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': 'total_cloud_cover',
            'year': str(year),
            'month': f"{month:02d}",
            'day': days,
            'time': hours,
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        },
        nc_path
    )

    print(f"    Salvato: {nc_path}")
    return nc_path


def download_era5_daily(year, month, day, hours_needed):
    """
    Alternativa: scarica ERA5 per un singolo giorno e ore specifiche.
    Piu leggero ma richiede piu richieste API.
    """
    import cdsapi

    nc_path = os.path.join(OUTPUT_DIR, "netcdf", f"era5_tcc_{year}{month:02d}{day:02d}.nc")

    if os.path.exists(nc_path):
        return nc_path

    print(f"    Download ERA5 TCC {year}-{month:02d}-{day:02d}...")

    c = cdsapi.Client()

    hours = sorted(set([f"{h:02d}:00" for h in hours_needed]))

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': 'total_cloud_cover',
            'year': str(year),
            'month': f"{month:02d}",
            'day': f"{day:02d}",
            'time': hours,
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        },
        nc_path
    )

    return nc_path


# ==============================================================
# FASE 3: Estrazione valori TCC
# ==============================================================

def extract_tcc_from_netcdf(nc_path, target_lat, target_lon, target_time):
    """Estrae TCC dal file NetCDF al punto e tempo piu vicino."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    # La variabile puo chiamarsi 'tcc' o 'total_cloud_cover'
    var_name = None
    for name in ['tcc', 'total_cloud_cover', 'TCC']:
        if name in ds.data_vars:
            var_name = name
            break
    if var_name is None:
        var_name = list(ds.data_vars)[0]

    data = ds[var_name]

    # ERA5 lon va da -180 a 180 o da 0 a 360 (dipende dal download)
    lons = ds['longitude'].values
    if lons.min() >= 0 and target_lon < 0:
        era5_lon = target_lon + 360
    elif lons.max() <= 180 and target_lon > 180:
        era5_lon = target_lon - 360
    else:
        era5_lon = target_lon

    # Nearest neighbor in spazio e tempo
    # La coordinata temporale puo chiamarsi 'time' o 'valid_time'
    time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'
    value = float(data.sel(
        latitude=target_lat,
        longitude=era5_lon,
        **{time_coord: target_time},
        method='nearest'
    ).values)

    ds.close()

    # ERA5 TCC e' in frazione (0-1), convertire in percentuale
    if value <= 1.0:
        value = value * 100.0

    return value


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("CONFRONTO ERA5 vs CloudSEN12+ - SCALA COMPLETA")
    print("=" * 60)

    # Verifica dipendenze
    try:
        import cdsapi
        print("  cdsapi: OK")
    except ImportError:
        print("  ERRORE: cdsapi non installato")
        print("  pip install cdsapi")
        print("\n  Poi configura ~/.cdsapirc:")
        print("    url: https://cds.climate.copernicus.eu/api")
        print("    key: <tuo-uid>:<tua-api-key>")
        return

    try:
        import xarray as xr
        import netCDF4
        print("  xarray + netCDF4: OK")
    except ImportError:
        print("  ERRORE: xarray o netCDF4 non installato")
        print("  conda install -c conda-forge xarray netcdf4")
        return

    # FASE 1: Estrai samples
    print("\n" + "=" * 60)
    print("FASE 1: ESTRAZIONE SAMPLES")
    print("=" * 60)

    df = extract_all_samples()

    # FASE 2: Organizza per mese (per download efficiente)
    print("\n" + "=" * 60)
    print("FASE 2: ORGANIZZAZIONE DOWNLOADS")
    print("=" * 60)

    # Raggruppa samples per anno-mese
    df['s2_datetime'] = pd.to_datetime(df['s2_date'])
    df['year_month'] = df['s2_datetime'].dt.to_period('M')

    months_needed = df['year_month'].unique()
    print(f"\n  Mesi da scaricare: {len(months_needed)}")
    for ym in sorted(months_needed):
        count = len(df[df['year_month'] == ym])
        print(f"    {ym}: {count} samples")

    # Scegli strategia download
    total_months = len(months_needed)
    print(f"\n  Strategia: download per giorno (piu leggero)")
    print(f"  Ogni file ~5-10 MB, totale stimato: ~{len(df) * 0.01:.0f} MB")

    # FASE 3: Download e confronto
    print("\n" + "=" * 60)
    print("FASE 3: DOWNLOAD ERA5 E CONFRONTO")
    print("=" * 60)

    # Carica risultati parziali
    already_done = set()
    if os.path.exists(RESULTS_PATH):
        df_partial = pd.read_csv(RESULTS_PATH)
        already_done = set(
            df_partial['split'].astype(str) + '_' + df_partial['sample_index'].astype(str)
        )
        print(f"  Trovati {len(already_done)} confronti gia completati")

    # Raggruppa per giorno per download efficiente
    df['date_only'] = df['s2_datetime'].dt.date
    df['hour'] = df['s2_datetime'].dt.hour

    days_grouped = df.groupby('date_only')

    results = []
    errors = []
    save_interval = 100
    total_processed = len(already_done)

    nc_cache = {}  # cache dei dataset NetCDF aperti

    for day, day_samples in days_grouped:
        year = day.year
        month = day.month
        day_num = day.day

        # Filtra samples non ancora processati
        pending = []
        for _, row in day_samples.iterrows():
            key = f"{row['split']}_{row['index']}"
            if key not in already_done:
                pending.append(row)

        if not pending:
            continue

        # Download ERA5 per questo giorno
        hours_needed = list(set([int(r['hour']) for r in pending]))

        nc_key = f"{year}{month:02d}{day_num:02d}"

        if nc_key not in nc_cache:
            try:
                nc_path = download_era5_daily(year, month, day_num, hours_needed)
                nc_cache[nc_key] = nc_path
            except Exception as e:
                print(f"    ERRORE download {day}: {e}")
                for r in pending:
                    errors.append({
                        'split': r['split'],
                        'sample_index': r['index'],
                        'roi_id': r['roi_id'],
                        's2_date': r['s2_date'],
                        'error': f"ERA5 download failed: {e}"
                    })
                continue
        else:
            nc_path = nc_cache[nc_key]

        # Estrai TCC per ogni sample del giorno
        for r in pending:
            try:
                # Rimuovi timezone per compatibilita con NetCDF
                target_time = r['s2_datetime'].tz_localize(None) if r['s2_datetime'].tzinfo else r['s2_datetime']
                tcc_value = extract_tcc_from_netcdf(
                    nc_path, r['lat'], r['lon'], target_time
                )

                cloudsen_cloud_pct = r.get('thick_cloud_pct', 0) + r.get('thin_cloud_pct', 0)

                results.append({
                    'split': r['split'],
                    'sample_index': r['index'],
                    'roi_id': r['roi_id'],
                    's2_date': r['s2_date'],
                    'year': r['year'],
                    'lon': r['lon'],
                    'lat': r['lat'],
                    'label_type': r['label_type'],
                    'era5_tcc_pct': tcc_value,
                    'cloudsen_clear_pct': r.get('clear_pct', 0),
                    'cloudsen_thick_cloud_pct': r.get('thick_cloud_pct', 0),
                    'cloudsen_thin_cloud_pct': r.get('thin_cloud_pct', 0),
                    'cloudsen_shadow_pct': r.get('shadow_pct', 0),
                    'cloudsen_total_cloud_pct': cloudsen_cloud_pct,
                    'difference': tcc_value - cloudsen_cloud_pct,
                })

                total_processed += 1

            except Exception as e:
                if len(errors) < 5:
                    print(f"    ERRORE estrazione: {e}")
                errors.append({
                    'split': r['split'],
                    'sample_index': r['index'],
                    'roi_id': r['roi_id'],
                    's2_date': r['s2_date'],
                    'error': str(e)
                })

        # Progress
        if total_processed % 50 == 0:
            print(f"  Processati: {total_processed}/{len(df)}")

        # Salva parziale
        if len(results) >= save_interval:
            _save_results(results, RESULTS_PATH, already_done)
            already_done.update(
                f"{r['split']}_{r['sample_index']}" for r in results
            )
            results = []

    # Salva finale
    if results:
        _save_results(results, RESULTS_PATH, already_done)

    # ==============================================================
    # STATISTICHE FINALI
    # ==============================================================

    print("\n\n" + "=" * 60)
    print("RISULTATI FINALI")
    print("=" * 60)

    if os.path.exists(RESULTS_PATH):
        df_results = pd.read_csv(RESULTS_PATH)
    else:
        print("Nessun risultato.")
        return

    print(f"\nTotale confronti: {len(df_results)}")
    print(f"Errori: {len(errors)}")

    diff = df_results['difference']
    print(f"\n--- STATISTICHE GLOBALI ---")
    print(f"  Samples analizzati:  {len(df_results)}")
    print(f"  Differenza media:    {diff.mean():.1f}%")
    print(f"  Differenza mediana:  {diff.median():.1f}%")
    print(f"  Deviazione standard: {diff.std():.1f}%")
    print(f"  RMSE:                {np.sqrt((diff**2).mean()):.1f}%")
    print(f"  MAE:                 {diff.abs().mean():.1f}%")

    corr = df_results['era5_tcc_pct'].corr(df_results['cloudsen_total_cloud_pct'])
    print(f"  Correlazione:        {corr:.3f}")

    # Per label type
    print(f"\n--- PER LABEL TYPE ---")
    for lt in sorted(df_results['label_type'].unique()):
        subset = df_results[df_results['label_type'] == lt]
        if len(subset) >= 5:
            d = subset['difference']
            c = subset['era5_tcc_pct'].corr(subset['cloudsen_total_cloud_pct'])
            print(f"  {lt:<12}: n={len(subset):>5}, MAE={d.abs().mean():.1f}%, RMSE={np.sqrt((d**2).mean()):.1f}%, corr={c:.3f}")

    # Per anno
    print(f"\n--- PER ANNO ---")
    for y in sorted(df_results['year'].unique()):
        subset = df_results[df_results['year'] == y]
        d = subset['difference']
        c = subset['era5_tcc_pct'].corr(subset['cloudsen_total_cloud_pct'])
        print(f"  {y}: n={len(subset):>5}, MAE={d.abs().mean():.1f}%, RMSE={np.sqrt((d**2).mean()):.1f}%, corr={c:.3f}")

    # Per tipo di nuvola dominante
    print(f"\n--- PER TIPO DI NUVOLA DOMINANTE ---")
    df_results['dominant_type'] = 'clear'
    df_results.loc[df_results['cloudsen_thick_cloud_pct'] > 50, 'dominant_type'] = 'thick_cloud'
    df_results.loc[df_results['cloudsen_thin_cloud_pct'] > 50, 'dominant_type'] = 'thin_cloud'
    df_results.loc[df_results['cloudsen_shadow_pct'] > 50, 'dominant_type'] = 'shadow'

    for ct in ['clear', 'thick_cloud', 'thin_cloud', 'shadow']:
        subset = df_results[df_results['dominant_type'] == ct]
        if len(subset) >= 5:
            d = subset['difference']
            c = subset['era5_tcc_pct'].corr(subset['cloudsen_total_cloud_pct'])
            print(f"  {ct:<15}: n={len(subset):>5}, MAE={d.abs().mean():.1f}%, corr={c:.3f}")

    # Distribuzione differenze
    print(f"\n--- DISTRIBUZIONE DIFFERENZE ---")
    bins = [0, 10, 20, 30, 50, 100]
    for j in range(len(bins) - 1):
        mask = (diff.abs() >= bins[j]) & (diff.abs() < bins[j + 1])
        count = mask.sum()
        pct = 100 * count / len(diff)
        bar = '#' * int(pct)
        print(f"  |diff| {bins[j]:>3}-{bins[j+1]:>3}%: {count:>5} ({pct:>5.1f}%) {bar}")

    # Salva errori
    if errors:
        df_errors = pd.DataFrame(errors)
        df_errors.to_csv(os.path.join(OUTPUT_DIR, "era5_errors.csv"), index=False)

    print(f"\nFile output: {RESULTS_PATH}")


def _save_results(new_results, output_path, already_done):
    """Salva risultati parziali."""
    df_new = pd.DataFrame(new_results)

    if os.path.exists(output_path) and already_done:
        df_old = pd.read_csv(output_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.drop_duplicates(
            subset=['split', 'sample_index'], keep='last', inplace=True
        )
    else:
        df_combined = df_new

    df_combined.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
