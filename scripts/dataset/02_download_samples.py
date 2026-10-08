"""
Script 2: Filtra samples 'high' e scarica le immagini
Prerequisito: aver eseguito 01_explore_dataset.py
"""

import pandas as pd
import os

print("=" * 60)
print("FASE 1b: Filtraggio samples con label 'high'")
print("=" * 60)

# Carica i CSV
df_nz = pd.read_csv("nz_samples.csv")
df_au = pd.read_csv("au_samples.csv")

print(f"Caricati {len(df_nz)} samples Nuova Zelanda")
print(f"Caricati {len(df_au)} samples Australia")

# Filtra solo 'high'
df_nz_high = df_nz[df_nz['label_type'] == 'high'].copy()
df_au_high = df_au[df_au['label_type'] == 'high'].copy()

print(f"\nFiltrati per label_type='high':")
print(f"  Nuova Zelanda: {len(df_nz_high)} samples")
print(f"  Australia: {len(df_au_high)} samples")

# Combina
df_high = pd.concat([df_nz_high, df_au_high], ignore_index=True)
print(f"\nTotale samples 'high': {len(df_high)}")

# Salva CSV filtrato
df_high.to_csv("samples_high_quality.csv", index=False)
print(f"\nSalvato: samples_high_quality.csv")

# Mostra distribuzione geografica
print("\n" + "=" * 60)
print("DISTRIBUZIONE SAMPLES 'HIGH'")
print("=" * 60)

print("\nPer ROI (Nuova Zelanda):")
if len(df_nz_high) > 0:
    print(df_nz_high.groupby('roi_id').size().to_string())

print("\nPer ROI (Australia) - primi 20:")
if len(df_au_high) > 0:
    roi_counts = df_au_high.groupby('roi_id').size().sort_values(ascending=False)
    print(roi_counts.head(20).to_string())

# Statistiche copertura nuvolosa
print("\n" + "=" * 60)
print("STATISTICHE COPERTURA NUVOLOSA")
print("=" * 60)

print("\nDistribuzione clear_pct:")
bins = [0, 25, 50, 75, 100]
labels = ['0-25%', '25-50%', '50-75%', '75-100%']
df_high['clear_category'] = pd.cut(df_high['clear_pct'], bins=bins, labels=labels, include_lowest=True)
print(df_high['clear_category'].value_counts().sort_index().to_string())

print("\nDistribuzione thick_cloud_pct:")
df_high['cloud_category'] = pd.cut(df_high['thick_cloud_pct'], bins=bins, labels=labels, include_lowest=True)
print(df_high['cloud_category'].value_counts().sort_index().to_string())

print("\n" + "=" * 60)
print("PROSSIMO STEP")
print("=" * 60)
print("Esegui: 03_download_images.py per scaricare le immagini GeoTIFF")
print("(richiede l'aggiornamento dell'environment con tacoreader)")
