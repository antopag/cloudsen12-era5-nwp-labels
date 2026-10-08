"""
Script 5: Download immagini per anno (2021 e 2022)
Obiettivo: Scaricare dati filtrati per anno in cartelle separate
Modalità: GLOBALE (tutte le regioni) con label high quality
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
import os
import json
import re

print("=" * 60)
print("DOWNLOAD DATI PER ANNO (2021 e 2022) - GLOBALE")
print("=" * 60)

# Funzione per estrarre coordinate dal proj_centroid
def extract_coords(centroid_str):
    """Estrae lon, lat da stringa POINT (lon lat)"""
    if centroid_str and "POINT" in str(centroid_str):
        match = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', str(centroid_str))
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None

def extract_year(date_str):
    """Estrae l'anno dalla data ISO"""
    if date_str:
        match = re.match(r'(\d{4})', str(date_str))
        if match:
            return int(match.group(1))
    return None

# Carica il dataset
print("\nCaricamento dataset da HuggingFace...")
dataset = load_dataset("isp-uv-es/CloudSEN12Plus")

print(f"Dataset caricato!")
print(f"- Train: {len(dataset['train'])} samples")

# Analizza distribuzione anni
print("\n" + "=" * 60)
print("ANALISI DISTRIBUZIONE ANNI (GLOBALE)")
print("=" * 60)

# Raccolta samples per anno
samples_by_year = {2021: [], 2022: []}
year_counts = {}
year_high_counts = {}

print("\nScansione del dataset train...")
for i, sample in enumerate(dataset['train']):
    if i % 5000 == 0:
        print(f"  Processati {i}/{len(dataset['train'])} samples...")

    # Estrai anno
    s2_date = sample.get('s2_date', '')
    year = extract_year(s2_date)

    if year:
        year_counts[year] = year_counts.get(year, 0) + 1

        # Conta anche quelli high quality
        if sample.get('label_type', '') == 'high':
            year_high_counts[year] = year_high_counts.get(year, 0) + 1

    # Filtra solo 2021 e 2022
    if year not in [2021, 2022]:
        continue

    # Estrai coordinate
    centroid = sample.get('proj_centroid', '')
    lon, lat = extract_coords(centroid)

    # Filtra per alta qualità
    label_type = sample.get('label_type', '')
    if label_type != 'high':
        continue

    # Salva sample (GLOBALE - tutte le regioni)
    sample_data = {
        'index': i,
        'roi_id': sample.get('roi_id', ''),
        's2_id': sample.get('s2_id', ''),
        's2_id_gee': sample.get('s2_id_gee', ''),
        's2_date': s2_date,
        'year': year,
        'lon': lon if lon else 0,
        'lat': lat if lat else 0,
        'proj_epsg': sample.get('proj_epsg', ''),
        'label_type': label_type,
        'clear_pct': sample.get('clear_percentage', 0),
        'thick_cloud_pct': sample.get('thick_percentage', 0),
        'thin_cloud_pct': sample.get('thin_percentage', 0),
        'shadow_pct': sample.get('cloud_shadow_percentage', 0),
        'url': sample.get('url', ''),
        'datapoint_id': sample.get('datapoint_id', ''),
    }

    samples_by_year[year].append(sample_data)

# Mostra distribuzione anni globale
print("\n" + "=" * 60)
print("DISTRIBUZIONE ANNI NEL DATASET COMPLETO")
print("=" * 60)
print(f"{'Anno':<8} {'Totali':<12} {'High Quality':<12}")
print("-" * 32)
for year in sorted(year_counts.keys()):
    total = year_counts[year]
    high = year_high_counts.get(year, 0)
    print(f"{year:<8} {total:<12} {high:<12}")

# Mostra risultati per anno
print("\n" + "=" * 60)
print("SAMPLES HIGH QUALITY PER ANNO (GLOBALE)")
print("=" * 60)
for year in [2021, 2022]:
    count = len(samples_by_year[year])
    print(f"  {year}: {count} samples")

# Verifica se ci sono dati
total_samples = len(samples_by_year[2021]) + len(samples_by_year[2022])
if total_samples == 0:
    print("\n" + "=" * 60)
    print("ATTENZIONE: Nessun sample HIGH QUALITY trovato per 2021/2022!")
    print("=" * 60)
    print("\nIl dataset CloudSEN12+ potrebbe non contenere dati per questi anni.")
    print("Anni disponibili nel dataset:", sorted(year_counts.keys()))
    exit(0)

# Salva CSV per anno
for year in [2021, 2022]:
    if samples_by_year[year]:
        df = pd.DataFrame(samples_by_year[year])
        csv_path = f"samples_{year}_global.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSalvato: {csv_path} ({len(df)} samples)")

        # Mostra distribuzione geografica
        print(f"  Coordinate: lon [{df['lon'].min():.1f}, {df['lon'].max():.1f}], lat [{df['lat'].min():.1f}, {df['lat'].max():.1f}]")

# Chiedi se procedere con download
print("\n" + "=" * 60)
print("DOWNLOAD IMMAGINI")
print("=" * 60)

proceed = input("\nVuoi procedere con il download delle immagini? (s/n): ")
if proceed.lower() != 's':
    print("Download annullato.")
    exit(0)

# Import per download
try:
    import tacoreader.v1 as tacoreader
    import rasterio as rio
    print("\ntacoreader e rasterio importati correttamente")
except ImportError as e:
    print(f"ERRORE: {e}")
    print("\nInstalla le dipendenze:")
    print("  conda install rasterio")
    print("  pip install tacoreader")
    exit(1)

# Carica dataset tacoreader
print("\nCaricamento dataset CloudSEN12 L1C...")
taco_dataset = tacoreader.load("tacofoundation:cloudsen12-l1c")
print(f"Dataset caricato: {len(taco_dataset)} samples totali")

# Crea mapping datapoint_id -> indice
print("\nCostruzione indice dataset...")
datapoint_to_idx = {}
for i, sample in enumerate(dataset['train']):
    datapoint_to_idx[sample['datapoint_id']] = i
print(f"Mapping creato: {len(datapoint_to_idx)} entries")

# Download per ogni anno
for year in [2021, 2022]:
    if not samples_by_year[year]:
        print(f"\nNessun sample per {year}, skip...")
        continue

    print(f"\n" + "=" * 60)
    print(f"DOWNLOAD {year}")
    print("=" * 60)

    # Crea cartelle
    year_dir = str(year)
    os.makedirs(year_dir, exist_ok=True)
    os.makedirs(os.path.join(year_dir, "s2_images"), exist_ok=True)
    os.makedirs(os.path.join(year_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(year_dir, "metadata"), exist_ok=True)

    df = pd.DataFrame(samples_by_year[year])
    n_samples = len(df)

    downloaded = []
    errors = []

    for i, row in df.iterrows():
        roi_id = row['roi_id']
        s2_id_gee = row['s2_id_gee']
        datapoint_id = row.get('datapoint_id', f"{roi_id}__{s2_id_gee}")

        print(f"\n[{len(downloaded)+len(errors)+1}/{n_samples}] {datapoint_id}")

        try:
            # Trova indice nel dataset
            idx = None
            if datapoint_id in datapoint_to_idx:
                idx = datapoint_to_idx[datapoint_id]
            else:
                # Prova a cercare per roi_id e s2_id_gee
                for dp_id, dp_idx in datapoint_to_idx.items():
                    if roi_id in dp_id and s2_id_gee in dp_id:
                        idx = dp_idx
                        break

            if idx is None:
                print(f"  SKIP: datapoint_id non trovato")
                errors.append({'datapoint_id': datapoint_id, 'error': 'not found'})
                continue

            # Leggi dal dataset tacoreader
            sample_data = taco_dataset.read(idx)

            # Salva S2 image
            s2_path = sample_data.read(0)
            with rio.open(s2_path) as src:
                img = src.read()
                profile = src.profile

            out_s2_path = os.path.join(year_dir, "s2_images", f"{roi_id}_{s2_id_gee}.tif")
            profile.update(count=img.shape[0])
            with rio.open(out_s2_path, 'w', **profile) as dst:
                dst.write(img)

            # Salva label
            label_path = sample_data.read(1)
            with rio.open(label_path) as src:
                label = src.read()
                label_profile = src.profile

            out_label_path = os.path.join(year_dir, "labels", f"{roi_id}_{s2_id_gee}_label.tif")
            with rio.open(out_label_path, 'w', **label_profile) as dst:
                dst.write(label)

            # Salva metadata
            metadata = row.to_dict()
            metadata['hf_index'] = idx
            out_meta_path = os.path.join(year_dir, "metadata", f"{roi_id}_{s2_id_gee}.json")
            with open(out_meta_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            downloaded.append({
                'datapoint_id': datapoint_id,
                's2_path': out_s2_path,
                'label_path': out_label_path,
                'meta_path': out_meta_path
            })

            print(f"  OK: salvato")

        except Exception as e:
            print(f"  ERRORE: {e}")
            errors.append({'datapoint_id': datapoint_id, 'error': str(e)})

    # Salva indice per anno
    print(f"\n--- Riepilogo {year} ---")
    print(f"Scaricati: {len(downloaded)}")
    print(f"Errori: {len(errors)}")

    if downloaded:
        df_downloaded = pd.DataFrame(downloaded)
        df_downloaded.to_csv(os.path.join(year_dir, "index.csv"), index=False)
        print(f"Indice salvato: {year_dir}/index.csv")

    if errors:
        df_errors = pd.DataFrame(errors)
        df_errors.to_csv(os.path.join(year_dir, "errors.csv"), index=False)
        print(f"Errori salvati: {year_dir}/errors.csv")

print("\n" + "=" * 60)
print("COMPLETATO!")
print("=" * 60)
print("\nCartelle create:")
for year in [2021, 2022]:
    if samples_by_year[year]:
        print(f"  - {year}/ (s2_images, labels, metadata)")
