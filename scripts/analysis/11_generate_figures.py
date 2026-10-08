"""
Script 11: Genera figure per il paper
ERA5 vs CloudSEN12+ e GFS vs CloudSEN12+
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import os

# ==============================================================
# CONFIGURAZIONE
# ==============================================================

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Stile per paper
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Carica dati
print("Caricamento dati...")
df_era5 = pd.read_csv("era5_data/era5_vs_cloudsen12.csv")
print(f"  ERA5: {len(df_era5)} samples")

try:
    df_gfs = pd.read_csv("gfs_data/gfs_vs_cloudsen12_full.csv")
    print(f"  GFS: {len(df_gfs)} samples")
    HAS_GFS = True
except:
    print("  GFS: non disponibile")
    HAS_GFS = False

# Aggiungi colonne utili
df_era5['cloudsen_total_cloud_pct'] = df_era5['cloudsen_thick_cloud_pct'] + df_era5['cloudsen_thin_cloud_pct']
df_era5['abs_diff'] = df_era5['difference'].abs()

# Tipo dominante
df_era5['dominant_type'] = 'Clear'
df_era5.loc[df_era5['cloudsen_thick_cloud_pct'] > 50, 'dominant_type'] = 'Thick cloud'
df_era5.loc[df_era5['cloudsen_thin_cloud_pct'] > 50, 'dominant_type'] = 'Thin cloud'
df_era5.loc[df_era5['cloudsen_shadow_pct'] > 50, 'dominant_type'] = 'Shadow'

# Banda di latitudine
df_era5['lat_band'] = 'Mid-latitude'
df_era5.loc[df_era5['lat'].abs() < 23.5, 'lat_band'] = 'Tropical'
df_era5.loc[df_era5['lat'].abs() > 60, 'lat_band'] = 'Polar'

# Stagione (basata su emisfero)
df_era5['s2_datetime'] = pd.to_datetime(df_era5['s2_date'])
df_era5['month'] = df_era5['s2_datetime'].dt.month

def get_season(row):
    m = row['month']
    south = row['lat'] < 0
    if m in [12, 1, 2]:
        return 'Summer' if south else 'Winter'
    elif m in [3, 4, 5]:
        return 'Autumn' if south else 'Spring'
    elif m in [6, 7, 8]:
        return 'Winter' if south else 'Summer'
    else:
        return 'Spring' if south else 'Autumn'

df_era5['season'] = df_era5.apply(get_season, axis=1)


# ==============================================================
# FIGURA 1: Scatter plot ERA5 vs CloudSEN12+ (density)
# ==============================================================

def fig1_scatter_density():
    print("\nFigura 1: Scatter plot density...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ERA5
    ax = axes[0]
    x = df_era5['cloudsen_total_cloud_pct']
    y = df_era5['era5_tcc_pct']
    h = ax.hist2d(x, y, bins=100, cmap='YlOrRd', norm=LogNorm(), range=[[0, 100], [0, 100]])
    fig.colorbar(h[3], ax=ax, label='Count')
    ax.plot([0, 100], [0, 100], 'k--', lw=1, alpha=0.5, label='1:1 line')
    ax.set_xlabel('CloudSEN12+ Cloud Cover (%)')
    ax.set_ylabel('ERA5 TCC (%)')
    ax.set_title(f'ERA5 vs CloudSEN12+ (n={len(df_era5):,})')
    r = np.corrcoef(x, y)[0, 1]
    mae = (y - x).abs().mean()
    ax.text(5, 92, f'r = {r:.3f}\nMAE = {mae:.1f}%', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')

    # GFS
    ax = axes[1]
    if HAS_GFS:
        x2 = df_gfs['cloudsen_total_cloud_pct']
        y2 = df_gfs['gfs_tcdc_pct']
        ax.scatter(x2, y2, alpha=0.5, s=20, c='steelblue', edgecolor='none')
        ax.plot([0, 100], [0, 100], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('CloudSEN12+ Cloud Cover (%)')
        ax.set_ylabel('GFS TCDC (%)')
        ax.set_title(f'GFS vs CloudSEN12+ (n={len(df_gfs)})')
        r2 = np.corrcoef(x2, y2)[0, 1]
        mae2 = (y2 - x2).abs().mean()
        ax.text(5, 92, f'r = {r2:.3f}\nMAE = {mae2:.1f}%', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'GFS data not available', transform=ax.transAxes, ha='center')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_scatter_era5_gfs.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_scatter_era5_gfs.pdf'))
    plt.close()
    print("  Salvato: fig1_scatter_era5_gfs")


# ==============================================================
# FIGURA 2: Distribuzione differenze
# ==============================================================

def fig2_difference_distribution():
    print("\nFigura 2: Distribuzione differenze...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Istogramma differenze
    ax = axes[0]
    ax.hist(df_era5['difference'], bins=80, range=(-100, 100),
            color='steelblue', alpha=0.7, edgecolor='none')
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.axvline(df_era5['difference'].mean(), color='red', lw=1.5, ls='-',
               label=f"Mean = {df_era5['difference'].mean():.1f}%")
    ax.axvline(df_era5['difference'].median(), color='orange', lw=1.5, ls='-',
               label=f"Median = {df_era5['difference'].median():.1f}%")
    ax.set_xlabel('ERA5 TCC - CloudSEN12+ Cloud Cover (%)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Differences')
    ax.legend()

    # Distribuzione per tipo dominante
    ax = axes[1]
    types = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']
    colors = ['#2ecc71', '#95a5a6', '#bdc3c7', '#34495e']

    for ct, color in zip(types, colors):
        subset = df_era5[df_era5['dominant_type'] == ct]
        if len(subset) > 10:
            ax.hist(subset['difference'], bins=60, range=(-100, 100),
                    alpha=0.5, label=f"{ct} (n={len(subset):,})", color=color,
                    density=True, edgecolor='none')
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_xlabel('ERA5 TCC - CloudSEN12+ Cloud Cover (%)')
    ax.set_ylabel('Density')
    ax.set_title('Differences by Dominant Cloud Type')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_difference_distribution.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_difference_distribution.pdf'))
    plt.close()
    print("  Salvato: fig2_difference_distribution")


# ==============================================================
# FIGURA 3: Scatter per cloud type
# ==============================================================

def fig3_scatter_by_cloud_type():
    print("\nFigura 3: Scatter per cloud type...")

    types = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']
    colors = ['#2ecc71', '#95a5a6', '#bdc3c7', '#34495e']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, (ct, color) in enumerate(zip(types, colors)):
        ax = axes[i]
        subset = df_era5[df_era5['dominant_type'] == ct]

        if len(subset) > 100:
            h = ax.hist2d(subset['cloudsen_total_cloud_pct'], subset['era5_tcc_pct'],
                          bins=50, cmap='YlOrRd', norm=LogNorm(), range=[[0, 100], [0, 100]])
        else:
            ax.scatter(subset['cloudsen_total_cloud_pct'], subset['era5_tcc_pct'],
                       alpha=0.5, s=15, c=color, edgecolor='none')

        ax.plot([0, 100], [0, 100], 'k--', lw=0.8, alpha=0.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')

        r = subset['era5_tcc_pct'].corr(subset['cloudsen_total_cloud_pct'])
        mae = subset['difference'].abs().mean()
        ax.set_title(f'{ct}\n(n={len(subset):,}, r={r:.2f})', fontsize=10)
        ax.set_xlabel('CloudSEN12+ (%)' if i == 1 else '')
        if i == 0:
            ax.set_ylabel('ERA5 TCC (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_scatter_by_cloudtype.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_scatter_by_cloudtype.pdf'))
    plt.close()
    print("  Salvato: fig3_scatter_by_cloudtype")


# ==============================================================
# FIGURA 4: Mappa globale MAE
# ==============================================================

def fig4_global_map():
    print("\nFigura 4: Mappa globale...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Mappa 1: distribuzione samples con colore = differenza
    ax = axes[0]
    sc = ax.scatter(df_era5['lon'], df_era5['lat'],
                    c=df_era5['abs_diff'], cmap='RdYlGn_r',
                    s=2, alpha=0.3, vmin=0, vmax=80, edgecolor='none')
    fig.colorbar(sc, ax=ax, label='|ERA5 - CloudSEN12+| (%)', shrink=0.8)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Global Distribution of Absolute Difference')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axhline(23.5, color='gray', lw=0.5, ls=':')
    ax.axhline(-23.5, color='gray', lw=0.5, ls=':')
    ax.axhline(60, color='gray', lw=0.5, ls=':')
    ax.axhline(-60, color='gray', lw=0.5, ls=':')

    # Mappa 2: distribuzione samples con colore = ERA5 TCC
    ax = axes[1]
    sc2 = ax.scatter(df_era5['lon'], df_era5['lat'],
                     c=df_era5['era5_tcc_pct'], cmap='Blues',
                     s=2, alpha=0.3, vmin=0, vmax=100, edgecolor='none')
    fig.colorbar(sc2, ax=ax, label='ERA5 TCC (%)', shrink=0.8)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('ERA5 Total Cloud Cover at Sample Locations')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.axhline(23.5, color='gray', lw=0.5, ls=':')
    ax.axhline(-23.5, color='gray', lw=0.5, ls=':')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_global_map.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_global_map.pdf'))
    plt.close()
    print("  Salvato: fig4_global_map")


# ==============================================================
# FIGURA 5: Analisi per latitudine e stagione
# ==============================================================

def fig5_latitude_season():
    print("\nFigura 5: Latitudine e stagione...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Boxplot per banda di latitudine
    ax = axes[0]
    bands = ['Tropical', 'Mid-latitude', 'Polar']
    data_lat = [df_era5[df_era5['lat_band'] == b]['difference'] for b in bands]
    bp = ax.boxplot(data_lat, labels=bands, patch_artist=True, showfliers=False,
                    medianprops=dict(color='red', lw=2))
    colors_box = ['#f39c12', '#3498db', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_ylabel('ERA5 - CloudSEN12+ (%)')
    ax.set_title('Difference by Latitude Band')

    # Statistiche per banda
    for i, b in enumerate(bands):
        subset = df_era5[df_era5['lat_band'] == b]
        r = subset['era5_tcc_pct'].corr(subset['cloudsen_total_cloud_pct'])
        n = len(subset)
        ax.text(i + 1, ax.get_ylim()[1] * 0.85, f'n={n:,}\nr={r:.2f}',
                ha='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

    # Boxplot per stagione
    ax = axes[1]
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    data_season = [df_era5[df_era5['season'] == s]['difference'] for s in seasons]
    bp2 = ax.boxplot(data_season, labels=seasons, patch_artist=True, showfliers=False,
                     medianprops=dict(color='red', lw=2))
    colors_season = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db']
    for patch, color in zip(bp2['boxes'], colors_season):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_ylabel('ERA5 - CloudSEN12+ (%)')
    ax.set_title('Difference by Season')

    # MAE per banda latitudinale (profilo)
    ax = axes[2]
    lat_bins = np.arange(-90, 91, 10)
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    mae_by_lat = []
    corr_by_lat = []
    count_by_lat = []

    for j in range(len(lat_bins) - 1):
        mask = (df_era5['lat'] >= lat_bins[j]) & (df_era5['lat'] < lat_bins[j + 1])
        subset = df_era5[mask]
        if len(subset) >= 10:
            mae_by_lat.append(subset['abs_diff'].mean())
            corr_by_lat.append(subset['era5_tcc_pct'].corr(subset['cloudsen_total_cloud_pct']))
            count_by_lat.append(len(subset))
        else:
            mae_by_lat.append(np.nan)
            corr_by_lat.append(np.nan)
            count_by_lat.append(0)

    ax2 = ax.twinx()
    ax.bar(lat_centers, mae_by_lat, width=8, alpha=0.5, color='steelblue', label='MAE')
    ax2.plot(lat_centers, corr_by_lat, 'r-o', markersize=4, label='Correlation')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('MAE (%)', color='steelblue')
    ax2.set_ylabel('Correlation', color='red')
    ax.set_title('MAE and Correlation by Latitude')
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_latitude_season.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_latitude_season.pdf'))
    plt.close()
    print("  Salvato: fig5_latitude_season")


# ==============================================================
# FIGURA 6: Analisi per label type
# ==============================================================

def fig6_label_type():
    print("\nFigura 6: Label type comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    label_types = ['high', 'scribble', 'nolabel']
    colors_lt = ['#e74c3c', '#f39c12', '#95a5a6']

    for i, lt in enumerate(label_types):
        ax = axes[i]
        subset = df_era5[df_era5['label_type'] == lt]

        if len(subset) > 100:
            h = ax.hist2d(subset['cloudsen_total_cloud_pct'], subset['era5_tcc_pct'],
                          bins=50, cmap='YlOrRd', norm=LogNorm(), range=[[0, 100], [0, 100]])
        else:
            ax.scatter(subset['cloudsen_total_cloud_pct'], subset['era5_tcc_pct'],
                       alpha=0.5, s=10, edgecolor='none')

        ax.plot([0, 100], [0, 100], 'k--', lw=0.8, alpha=0.5)
        r = subset['era5_tcc_pct'].corr(subset['cloudsen_total_cloud_pct'])
        mae = subset['difference'].abs().mean()
        ax.set_title(f'{lt.capitalize()}\n(n={len(subset):,}, r={r:.3f}, MAE={mae:.1f}%)', fontsize=10)
        ax.set_xlabel('CloudSEN12+ Cloud (%)')
        if i == 0:
            ax.set_ylabel('ERA5 TCC (%)')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_label_type.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_label_type.pdf'))
    plt.close()
    print("  Salvato: fig6_label_type")


# ==============================================================
# FIGURA 7: Heatmap riassuntiva
# ==============================================================

def fig7_summary_heatmap():
    print("\nFigura 7: Heatmap riassuntiva...")

    # Crea matrice: righe = cloud type, colonne = metrica
    categories = {
        'All': df_era5,
        'Clear': df_era5[df_era5['dominant_type'] == 'Clear'],
        'Thick': df_era5[df_era5['dominant_type'] == 'Thick cloud'],
        'Thin': df_era5[df_era5['dominant_type'] == 'Thin cloud'],
        'Tropical': df_era5[df_era5['lat_band'] == 'Tropical'],
        'Mid-lat': df_era5[df_era5['lat_band'] == 'Mid-latitude'],
        'Polar': df_era5[df_era5['lat_band'] == 'Polar'],
        'High': df_era5[df_era5['label_type'] == 'high'],
        'Scribble': df_era5[df_era5['label_type'] == 'scribble'],
    }

    metrics = ['Correlation', 'MAE (%)', 'RMSE (%)', 'Bias (%)', 'N']
    data_matrix = []

    for name, subset in categories.items():
        if len(subset) < 10:
            data_matrix.append([np.nan] * 5)
            continue
        r = subset['era5_tcc_pct'].corr(subset['cloudsen_total_cloud_pct'])
        mae = subset['abs_diff'].mean()
        rmse = np.sqrt((subset['difference'] ** 2).mean())
        bias = subset['difference'].mean()
        n = len(subset)
        data_matrix.append([r, mae, rmse, bias, n])

    df_matrix = pd.DataFrame(data_matrix, index=categories.keys(), columns=metrics)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalizza per heatmap (escludi N)
    display_cols = ['Correlation', 'MAE (%)', 'RMSE (%)', 'Bias (%)']
    display_data = df_matrix[display_cols].values

    im = ax.imshow(display_data, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(display_cols)))
    ax.set_xticklabels(display_cols)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories.keys())

    # Annota celle
    for i in range(len(categories)):
        for j in range(len(display_cols)):
            val = display_data[i, j]
            if not np.isnan(val):
                text = f'{val:.2f}' if j == 0 else f'{val:.1f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=9,
                        color='white' if abs(val) > 30 else 'black')

    # Aggiungi N come testo a destra
    for i, (name, subset) in enumerate(categories.items()):
        n = len(subset)
        ax.text(len(display_cols) - 0.3, i, f'  n={n:,}', va='center', fontsize=8)

    ax.set_title('Summary Statistics: ERA5 vs CloudSEN12+')
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_summary_heatmap.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_summary_heatmap.pdf'))
    plt.close()
    print("  Salvato: fig7_summary_heatmap")


# ==============================================================
# FIGURA 8: Esempio visuale (immagini S2 + label + ERA5)
# ==============================================================

def fig8_visual_examples():
    print("\nFigura 8: Esempi visuali...")

    # Cerca compositi esistenti
    composite_dirs = []
    for d in ['DATA/visualizations/composite', '2021/visualizations/composite', '2022/visualizations/composite']:
        if os.path.exists(d):
            files = [f for f in os.listdir(d) if f.endswith('.png')]
            if files:
                composite_dirs.append((d, files))

    if not composite_dirs:
        print("  Nessun composito trovato, skip...")
        return

    # Prendi fino a 6 esempi
    examples = []
    for d, files in composite_dirs:
        for f in files[:3]:
            examples.append(os.path.join(d, f))
        if len(examples) >= 6:
            break

    n = min(len(examples), 6)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(n):
        img = plt.imread(examples[i])
        axes[i].imshow(img)
        name = os.path.basename(examples[i]).replace('_composite.png', '')[:30]
        axes[i].set_title(name, fontsize=8)
        axes[i].axis('off')

    for i in range(n, 6):
        axes[i].axis('off')

    plt.suptitle('Example CloudSEN12+ Composite Images (RGB | Label | Overlay)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig8_visual_examples.png'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig8_visual_examples.pdf'))
    plt.close()
    print("  Salvato: fig8_visual_examples")


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("GENERAZIONE FIGURE PER IL PAPER")
    print("=" * 60)

    fig1_scatter_density()
    fig2_difference_distribution()
    fig3_scatter_by_cloud_type()
    fig4_global_map()
    fig5_latitude_season()
    fig6_label_type()
    fig7_summary_heatmap()
    fig8_visual_examples()

    print("\n" + "=" * 60)
    print("COMPLETATO")
    print("=" * 60)
    print(f"\nTutte le figure salvate in: {FIGURES_DIR}/")
    print("\nFigure generate:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        size = os.path.getsize(os.path.join(FIGURES_DIR, f)) / 1024
        print(f"  {f} ({size:.0f} KB)")


if __name__ == '__main__':
    main()
