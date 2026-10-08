"""
Script 9: Confronto GFS vs CloudSEN12+ su larga scala
Scarica TCDC per TUTTI i samples 2021-2022 (qualsiasi label_type e regione)

Prerequisiti:
    conda install -c conda-forge cfgrib eccodes xarray
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
import os
import re
import urllib.request
from datetime import datetime

# ==============================================================
# CONFIGURAZIONE
# ==============================================================

OUTPUT_DIR = "gfs_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GFS_AWS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# Anni di interesse (GFS su AWS disponibile dal 2021)
TARGET_YEARS = [2021, 2022]

# ==============================================================
# FASE 1: Estrazione samples dal dataset
# ==============================================================

def extract_all_samples():
    """Estrae TUTTI i samples 2021-2022 dal dataset CloudSEN12+."""

    cache_path = os.path.join(OUTPUT_DIR, "all_samples_2021_2022.csv")

    # Se gia estratti, ricarica
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        print(f"  Caricati {len(df)} samples da cache ({cache_path})")
        return df

    print("\nCaricamento dataset da HuggingFace...")
    dataset = load_dataset("isp-uv-es/CloudSEN12Plus")
    print(f"Dataset caricato: {len(dataset['train'])} samples nel train")

    def extract_coords(centroid_str):
        if centroid_str and "POINT" in str(centroid_str):
            match = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', str(centroid_str))
            if match:
                return float(match.group(1)), float(match.group(2))
        return None, None

    samples = []
    print("\nScansione dataset...")

    for i, sample in enumerate(dataset['train']):
        if i % 5000 == 0:
            print(f"  {i}/{len(dataset['train'])} processati... ({len(samples)} trovati)")

        s2_date = sample.get('s2_date', '')
        if not s2_date:
            continue

        # Filtra per anno
        year_match = re.match(r'(\d{4})', str(s2_date))
        if not year_match:
            continue
        year = int(year_match.group(1))
        if year not in TARGET_YEARS:
            continue

        # Estrai coordinate
        centroid = sample.get('proj_centroid', '')
        lon, lat = extract_coords(centroid)
        if lon is None:
            continue

        samples.append({
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

    # Salva cache
    df.to_csv(cache_path, index=False)
    print(f"\nTrovati {len(df)} samples totali per 2021-2022")
    print(f"  2021: {len(df[df['year']==2021])}")
    print(f"  2022: {len(df[df['year']==2022])}")
    print(f"\n  Per label_type:")
    print(df['label_type'].value_counts().to_string())
    print(f"\nSalvato cache: {cache_path}")

    return df


# ==============================================================
# FASE 2: Download GFS TCDC
# ==============================================================

def find_nearest_gfs_cycle(s2_datetime):
    """Trova ciclo GFS piu vicino."""
    dt = datetime.fromisoformat(s2_datetime.replace('Z', '+00:00'))
    hour = dt.hour
    cycles = [0, 6, 12, 18]
    valid_cycles = [c for c in cycles if c <= hour]
    closest_cycle = max(valid_cycles) if valid_cycles else 18
    return dt.strftime('%Y%m%d'), f"{closest_cycle:02d}"


def download_and_extract_tcdc(gfs_date, gfs_cycle, target_lat, target_lon, grib_cache):
    """Scarica campo TCDC e estrae valore al punto."""

    cache_key = f"{gfs_date}_{gfs_cycle}"

    # Scarica campo GRIB se non in cache
    if cache_key not in grib_cache:
        idx_url = f"{GFS_AWS_BASE}/gfs.{gfs_date}/{gfs_cycle}/atmos/gfs.t{gfs_cycle}z.pgrb2.0p25.f000.idx"
        grib_url = f"{GFS_AWS_BASE}/gfs.{gfs_date}/{gfs_cycle}/atmos/gfs.t{gfs_cycle}z.pgrb2.0p25.f000"

        try:
            # Scarica indice
            req = urllib.request.Request(idx_url)
            with urllib.request.urlopen(req, timeout=30) as response:
                idx_content = response.read().decode('utf-8')

            # Trova TCDC entire atmosphere
            lines = idx_content.strip().split('\n')
            tcdc_start = None
            tcdc_end = None

            for j, line in enumerate(lines):
                if 'TCDC' in line and 'entire atmosphere' in line:
                    parts = line.split(':')
                    tcdc_start = int(parts[1])
                    if j + 1 < len(lines):
                        next_parts = lines[j + 1].split(':')
                        tcdc_end = int(next_parts[1]) - 1
                    break

            if tcdc_start is None:
                grib_cache[cache_key] = None
                return None, "TCDC not found in IDX"

            # Scarica campo
            range_header = f"bytes={tcdc_start}-{tcdc_end}" if tcdc_end else f"bytes={tcdc_start}-"
            req = urllib.request.Request(grib_url, headers={'Range': range_header})
            with urllib.request.urlopen(req, timeout=60) as response:
                grib_data = response.read()

            # Decodifica con cfgrib e salva griglia in cache
            tmp_path = os.path.join(OUTPUT_DIR, "_tmp_tcdc.grib2")
            with open(tmp_path, 'wb') as f:
                f.write(grib_data)

            import xarray as xr
            ds = xr.open_dataset(tmp_path, engine='cfgrib')
            var_name = list(ds.data_vars)[0]
            data = ds[var_name].values
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            ds.close()
            os.remove(tmp_path)

            grib_cache[cache_key] = {'data': data, 'lats': lats, 'lons': lons}

        except Exception as e:
            grib_cache[cache_key] = None
            return None, str(e)

    # Estrai valore dal cache
    cached = grib_cache[cache_key]
    if cached is None:
        return None, "GFS data not available (cached)"

    data = cached['data']
    lats = cached['lats']
    lons = cached['lons']

    gfs_lon = target_lon if target_lon >= 0 else target_lon + 360

    # Nearest neighbor
    lat_idx = np.argmin(np.abs(lats - target_lat))
    lon_idx = np.argmin(np.abs(lons - gfs_lon))

    value = float(data[lat_idx, lon_idx])
    return value, None


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("CONFRONTO GFS vs CloudSEN12+ - SCALA COMPLETA")
    print("=" * 60)

    # Verifica cfgrib
    try:
        import cfgrib
        print("  cfgrib: OK")
    except ImportError:
        print("  ERRORE: cfgrib necessario")
        print("  conda install -c conda-forge cfgrib eccodes")
        return

    # FASE 1: Estrai samples
    print("\n" + "=" * 60)
    print("FASE 1: ESTRAZIONE SAMPLES 2021-2022")
    print("=" * 60)

    df = extract_all_samples()

    # FASE 2: Download GFS
    print("\n" + "=" * 60)
    print("FASE 2: DOWNLOAD GFS TCDC")
    print("=" * 60)

    # Controlla se esiste gia un risultato parziale
    partial_path = os.path.join(OUTPUT_DIR, "gfs_vs_cloudsen12_full.csv")
    already_done = set()
    if os.path.exists(partial_path):
        df_partial = pd.read_csv(partial_path)
        already_done = set(df_partial['sample_index'].astype(str))
        print(f"  Trovati {len(already_done)} confronti gia completati, riprendo...")

    # Conta cicli GFS unici (per stimare download)
    cycles = set()
    for _, row in df.iterrows():
        gfs_date, gfs_cycle = find_nearest_gfs_cycle(row['s2_date'])
        cycles.add(f"{gfs_date}_{gfs_cycle}")

    print(f"\n  Samples da processare: {len(df)}")
    print(f"  Cicli GFS unici da scaricare: {len(cycles)}")
    print(f"  Download stimato: ~{len(cycles) * 50 / 1024:.1f} MB totali")

    grib_cache = {}
    results = []
    errors = []
    save_interval = 50  # salva ogni N samples

    for i, (idx, row) in enumerate(df.iterrows()):
        sample_key = str(row['index'])

        # Salta se gia fatto
        if sample_key in already_done:
            continue

        s2_date = row['s2_date']
        lon = row['lon']
        lat = row['lat']
        roi_id = row['roi_id']

        gfs_date, gfs_cycle = find_nearest_gfs_cycle(s2_date)
        cache_key = f"{gfs_date}_{gfs_cycle}"
        is_cached = cache_key in grib_cache

        if (i % 100 == 0) or (not is_cached):
            n_done = len(results) + len(already_done)
            print(f"\n[{n_done}/{len(df)}] {roi_id} | {s2_date[:10]} | GFS {gfs_date}/{gfs_cycle}z | ({lat:.1f}, {lon:.1f})", end="")
            if is_cached:
                print(" [cache]", end="")

        value, error = download_and_extract_tcdc(gfs_date, gfs_cycle, lat, lon, grib_cache)

        if value is not None:
            cloudsen_cloud_pct = row.get('thick_cloud_pct', 0) + row.get('thin_cloud_pct', 0)

            results.append({
                'sample_index': row['index'],
                'roi_id': roi_id,
                's2_date': s2_date,
                'year': row['year'],
                'lon': lon,
                'lat': lat,
                'label_type': row['label_type'],
                'gfs_date': gfs_date,
                'gfs_cycle': gfs_cycle,
                'gfs_tcdc_pct': value,
                'cloudsen_clear_pct': row.get('clear_pct', 0),
                'cloudsen_thick_cloud_pct': row.get('thick_cloud_pct', 0),
                'cloudsen_thin_cloud_pct': row.get('thin_cloud_pct', 0),
                'cloudsen_shadow_pct': row.get('shadow_pct', 0),
                'cloudsen_total_cloud_pct': cloudsen_cloud_pct,
                'difference': value - cloudsen_cloud_pct,
            })
        else:
            errors.append({
                'sample_index': row['index'],
                'roi_id': roi_id,
                's2_date': s2_date,
                'error': error
            })

        # Salva risultati parziali
        if len(results) > 0 and len(results) % save_interval == 0:
            _save_partial(results, already_done, partial_path)
            print(f"\n  [Salvataggio parziale: {len(results) + len(already_done)} totali]")

    # Salvataggio finale
    _save_partial(results, already_done, partial_path)

    # ==============================================================
    # STATISTICHE FINALI
    # ==============================================================

    print("\n\n" + "=" * 60)
    print("RISULTATI FINALI")
    print("=" * 60)

    if os.path.exists(partial_path):
        df_results = pd.read_csv(partial_path)
    elif results:
        df_results = pd.DataFrame(results)
    else:
        print("Nessun risultato.")
        return

    print(f"\nTotale confronti: {len(df_results)}")
    print(f"Errori: {len(errors)}")

    diff = df_results['difference']
    print(f"\n--- STATISTICHE DI CONFRONTO ---")
    print(f"  Samples analizzati:  {len(df_results)}")
    print(f"  Differenza media:    {diff.mean():.1f}%")
    print(f"  Differenza mediana:  {diff.median():.1f}%")
    print(f"  Deviazione standard: {diff.std():.1f}%")
    print(f"  RMSE:                {np.sqrt((diff**2).mean()):.1f}%")
    print(f"  MAE:                 {diff.abs().mean():.1f}%")

    corr = df_results['gfs_tcdc_pct'].corr(df_results['cloudsen_total_cloud_pct'])
    print(f"  Correlazione:        {corr:.3f}")

    # Per label type
    print(f"\n--- PER LABEL TYPE ---")
    for lt in df_results['label_type'].unique():
        subset = df_results[df_results['label_type'] == lt]
        if len(subset) >= 5:
            d = subset['difference']
            c = subset['gfs_tcdc_pct'].corr(subset['cloudsen_total_cloud_pct'])
            print(f"  {lt:<12}: n={len(subset):>5}, MAE={d.abs().mean():.1f}%, corr={c:.3f}")

    # Per anno
    print(f"\n--- PER ANNO ---")
    for y in sorted(df_results['year'].unique()):
        subset = df_results[df_results['year'] == y]
        d = subset['difference']
        c = subset['gfs_tcdc_pct'].corr(subset['cloudsen_total_cloud_pct'])
        print(f"  {y}: n={len(subset):>5}, MAE={d.abs().mean():.1f}%, corr={c:.3f}")

    # Distribuzione
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
        error_path = os.path.join(OUTPUT_DIR, "gfs_errors_full.csv")
        df_errors.to_csv(error_path, index=False)

    print(f"\nFile output: {partial_path}")


def _save_partial(new_results, already_done, output_path):
    """Salva risultati parziali, unendo con quelli precedenti."""
    if not new_results:
        return

    df_new = pd.DataFrame(new_results)

    if os.path.exists(output_path) and already_done:
        df_old = pd.read_csv(output_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset='sample_index', keep='last', inplace=True)
    else:
        df_combined = df_new

    df_combined.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
