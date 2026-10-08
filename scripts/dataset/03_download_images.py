"""
Script 3: Download immagini GeoTIFF usando tacoreader
Prerequisito:
  - aver eseguito 02_download_samples.py
  - conda install rasterio matplotlib
  - pip install tacoreader
"""

import pandas as pd
import numpy as np
import os
import json

print("=" * 60)
print("FASE 1c: Download immagini con tacoreader")
print("=" * 60)

# Verifica dipendenze
try:
    # Usa import legacy per formato v1
    import tacoreader.v1 as tacoreader
    import rasterio as rio
    print("tacoreader (v1 legacy) e rasterio importati correttamente")
except ImportError as e:
    print(f"ERRORE: {e}")
    print("\nInstalla le dipendenze:")
    print("  conda install rasterio")
    print("  pip install tacoreader")
    exit(1)

# Carica lista samples
if not os.path.exists("samples_high_quality.csv"):
    print("ERRORE: Esegui prima 02_download_samples.py")
    exit(1)

df = pd.read_csv("samples_high_quality.csv")
print(f"\nCaricati {len(df)} samples da scaricare")

# Carica dataset con tacoreader
print("\nCaricamento dataset CloudSEN12 L1C...")
dataset = tacoreader.load("tacofoundation:cloudsen12-l1c")

print(f"Dataset caricato: {len(dataset)} samples totali")

# Crea cartelle output
output_dir = "downloaded_images"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "s2_images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)

# Mappa tra datapoint_id e indice nel dataset tacoreader
print("\nCostruzione indice dataset...")
# Il dataset tacoreader potrebbe avere un modo diverso di indicizzare
# Proviamo prima a capire la struttura

# Prova a leggere un sample di test
print("\nTest lettura sample...")
try:
    test_sample = dataset.read(0)
    print(f"  Sample 0 letto correttamente")
    print(f"  Numero di assets: {len(test_sample)}")

    # Leggi i primi bytes per verificare
    s2_asset = test_sample.read(0)  # S2 L1C
    label_asset = test_sample.read(1)  # Label
    print(f"  S2 asset: {s2_asset}")
    print(f"  Label asset: {label_asset}")
except Exception as e:
    print(f"  Errore: {e}")

# Chiedi quanti samples scaricare
print("\n" + "=" * 60)
n_samples = len(df)
if n_samples > 50:
    print(f"Hai {n_samples} samples da scaricare.")
    print("Consiglio: inizia con un subset per test.")
    try:
        user_input = input(f"Quanti samples vuoi scaricare? (default: 10, max: {n_samples}): ")
        if user_input.strip():
            n_download = min(int(user_input), n_samples)
        else:
            n_download = 10
    except:
        n_download = 10
else:
    n_download = n_samples

print(f"\nScaricamento di {n_download} samples...")

# Per ora, usiamo l'indice dal dataset HuggingFace
# e mappiamo con il datapoint_id
from datasets import load_dataset

print("\nCaricamento dataset HuggingFace per mapping...")
hf_dataset = load_dataset("isp-uv-es/CloudSEN12Plus")

# Crea mapping datapoint_id -> indice
datapoint_to_idx = {}
for i, sample in enumerate(hf_dataset['train']):
    datapoint_to_idx[sample['datapoint_id']] = i

print(f"Mapping creato: {len(datapoint_to_idx)} entries")

# Download samples
downloaded = []
errors = []

for i, row in df.head(n_download).iterrows():
    roi_id = row['roi_id']
    s2_id = row['s2_id']
    datapoint_id = f"{roi_id}__{row['s2_id_gee']}"

    print(f"\n[{i+1}/{n_download}] {datapoint_id}")

    try:
        # Trova indice nel dataset
        if datapoint_id in datapoint_to_idx:
            idx = datapoint_to_idx[datapoint_id]
        else:
            # Prova a cercare per roi_id e s2_id_gee separatamente
            found = False
            for dp_id, dp_idx in datapoint_to_idx.items():
                if roi_id in dp_id and row['s2_id_gee'] in dp_id:
                    idx = dp_idx
                    found = True
                    break
            if not found:
                print(f"  SKIP: datapoint_id non trovato")
                errors.append({'datapoint_id': datapoint_id, 'error': 'not found'})
                continue

        # Leggi dal dataset tacoreader usando l'indice
        sample_data = dataset.read(idx)

        # Salva S2 image (bande RGB: B4, B3, B2 -> indici 3, 2, 1)
        s2_path = sample_data.read(0)
        with rio.open(s2_path) as src:
            # Leggi tutte le 12 bande
            img = src.read()
            profile = src.profile

        # Salva come GeoTIFF
        out_s2_path = os.path.join(output_dir, "s2_images", f"{roi_id}_{row['s2_id_gee']}.tif")
        profile.update(count=img.shape[0])
        with rio.open(out_s2_path, 'w', **profile) as dst:
            dst.write(img)

        # Salva label
        label_path = sample_data.read(1)
        with rio.open(label_path) as src:
            label = src.read()
            label_profile = src.profile

        out_label_path = os.path.join(output_dir, "labels", f"{roi_id}_{row['s2_id_gee']}_label.tif")
        with rio.open(out_label_path, 'w', **label_profile) as dst:
            dst.write(label)

        # Salva metadata
        metadata = row.to_dict()
        metadata['hf_index'] = idx
        out_meta_path = os.path.join(output_dir, "metadata", f"{roi_id}_{row['s2_id_gee']}.json")
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

# Riepilogo
print("\n" + "=" * 60)
print("RIEPILOGO DOWNLOAD")
print("=" * 60)
print(f"Scaricati: {len(downloaded)}")
print(f"Errori: {len(errors)}")

# Salva indice
if downloaded:
    df_downloaded = pd.DataFrame(downloaded)
    df_downloaded.to_csv(os.path.join(output_dir, "index.csv"), index=False)
    print(f"\nIndice salvato: {output_dir}/index.csv")

if errors:
    df_errors = pd.DataFrame(errors)
    df_errors.to_csv(os.path.join(output_dir, "errors.csv"), index=False)
    print(f"Errori salvati: {output_dir}/errors.csv")

print(f"\nImmagini salvate in: {output_dir}/")
