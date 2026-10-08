"""
Script 1: Esplorazione del dataset CloudSEN12+
Obiettivo: Capire la struttura e identificare i dati della Nuova Zelanda/Australia
"""

from datasets import load_dataset
import pandas as pd
import re

print("=" * 60)
print("FASE 1: Caricamento dataset CloudSEN12+")
print("=" * 60)

# Carica il dataset (solo metadati, non scarica le immagini)
print("\nCaricamento dataset da HuggingFace...")
print("(Questo potrebbe richiedere alcuni minuti la prima volta)\n")

dataset = load_dataset("isp-uv-es/CloudSEN12Plus")

print(f"Dataset caricato!")
print(f"- Train: {len(dataset['train'])} samples")
print(f"- Validation: {len(dataset['validation'])} samples")
print(f"- Test: {len(dataset['test'])} samples")

# Esplora le colonne disponibili
print("\n" + "=" * 60)
print("COLONNE DISPONIBILI")
print("=" * 60)
columns = dataset['train'].column_names
for col in columns:
    print(f"  - {col}")

# Prendi un campione per capire i valori
print("\n" + "=" * 60)
print("ESEMPIO DI UN RECORD")
print("=" * 60)
sample = dataset['train'][0]
for key, value in sample.items():
    if not isinstance(value, bytes) and not str(value).startswith("{'bytes'"):
        print(f"  {key}: {value}")

# Funzione per estrarre coordinate dal proj_centroid
def extract_coords(centroid_str):
    """Estrae lon, lat da stringa POINT (lon lat)"""
    if centroid_str and "POINT" in str(centroid_str):
        match = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', str(centroid_str))
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None

# Analizza la distribuzione geografica
print("\n" + "=" * 60)
print("ANALISI GEOGRAFICA - Ricerca Nuova Zelanda/Australia")
print("=" * 60)

# Nuova Zelanda: lon 166-179, lat -34 a -47
# Australia: lon 113-154, lat -10 a -44

nz_samples = []
au_samples = []

print("\nScansione del dataset train...")
for i, sample in enumerate(dataset['train']):
    if i % 5000 == 0:
        print(f"  Processati {i}/{len(dataset['train'])} samples...")

    centroid = sample.get('proj_centroid', '')
    lon, lat = extract_coords(centroid)

    if lon is not None and lat is not None:
        # Nuova Zelanda
        if 166 <= lon <= 179 and -47 <= lat <= -34:
            nz_samples.append({
                'index': i,
                'roi_id': sample.get('roi_id', ''),
                's2_id': sample.get('s2_id', ''),
                's2_id_gee': sample.get('s2_id_gee', ''),
                's2_date': sample.get('s2_date', ''),
                'lon': lon,
                'lat': lat,
                'proj_epsg': sample.get('proj_epsg', ''),
                'label_type': sample.get('label_type', ''),
                'clear_pct': sample.get('clear_percentage', 0),
                'thick_cloud_pct': sample.get('thick_percentage', 0),
                'thin_cloud_pct': sample.get('thin_percentage', 0),
                'shadow_pct': sample.get('cloud_shadow_percentage', 0),
                'url': sample.get('url', ''),
            })
        # Australia
        elif 113 <= lon <= 154 and -44 <= lat <= -10:
            au_samples.append({
                'index': i,
                'roi_id': sample.get('roi_id', ''),
                's2_id': sample.get('s2_id', ''),
                's2_id_gee': sample.get('s2_id_gee', ''),
                's2_date': sample.get('s2_date', ''),
                'lon': lon,
                'lat': lat,
                'proj_epsg': sample.get('proj_epsg', ''),
                'label_type': sample.get('label_type', ''),
                'clear_pct': sample.get('clear_percentage', 0),
                'thick_cloud_pct': sample.get('thick_percentage', 0),
                'thin_cloud_pct': sample.get('thin_percentage', 0),
                'shadow_pct': sample.get('cloud_shadow_percentage', 0),
                'url': sample.get('url', ''),
            })

print(f"\n  Trovati {len(nz_samples)} samples in Nuova Zelanda")
print(f"  Trovati {len(au_samples)} samples in Australia")

# Salva i risultati
if nz_samples:
    df_nz = pd.DataFrame(nz_samples)
    df_nz.to_csv("nz_samples.csv", index=False)
    print(f"\n  Salvato: nz_samples.csv")
    print(f"\n  Prime 10 righe Nuova Zelanda:")
    print(df_nz.head(10).to_string())

if au_samples:
    df_au = pd.DataFrame(au_samples)
    df_au.to_csv("au_samples.csv", index=False)
    print(f"\n  Salvato: au_samples.csv")
    print(f"\n  Prime 10 righe Australia:")
    print(df_au.head(10).to_string())

# Riepilogo
print("\n" + "=" * 60)
print("RIEPILOGO")
print("=" * 60)
print(f"Totale samples Nuova Zelanda: {len(nz_samples)}")
print(f"Totale samples Australia: {len(au_samples)}")
print(f"Totale samples area di interesse: {len(nz_samples) + len(au_samples)}")

# Statistiche label_type
if nz_samples:
    df_nz = pd.DataFrame(nz_samples)
    print(f"\nNuova Zelanda - label_type:")
    print(df_nz['label_type'].value_counts().to_string())
    nz_high = len(df_nz[df_nz['label_type'] == 'high'])
    print(f"  -> Con label 'high': {nz_high}")

if au_samples:
    df_au = pd.DataFrame(au_samples)
    print(f"\nAustralia - label_type:")
    print(df_au['label_type'].value_counts().to_string())
    au_high = len(df_au[df_au['label_type'] == 'high'])
    print(f"  -> Con label 'high': {au_high}")

print("\nFile generati:")
print("  - nz_samples.csv (Nuova Zelanda)")
print("  - au_samples.csv (Australia)")
print("\nProssimo step: esegui 02_download_samples.py per scaricare i dati")
