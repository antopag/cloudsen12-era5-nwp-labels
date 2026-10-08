"""
Script 11b: rigenera fig2 (distribuzione differenze) e fig3 (scatter per tipo dominante)
con colori distinguibili e SENZA la classe shadow (ERA5 non modella le ombre; n=85).
Commento di Anna 30/09/2026. Definizione di tipo dominante identica allo script 11 (>50%).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

FIG = 'figures'
df = pd.read_csv('era5_data/era5_vs_cloudsen12.csv')
df['dominant_type'] = 'Clear'
df.loc[df['cloudsen_thick_cloud_pct'] > 50, 'dominant_type'] = 'Thick cloud'
df.loc[df['cloudsen_thin_cloud_pct'] > 50, 'dominant_type'] = 'Thin cloud'
df.loc[df['cloudsen_shadow_pct'] > 50, 'dominant_type'] = 'Shadow'
TYPES = ['Clear', 'Thick cloud', 'Thin cloud']
COLORS = ['#2ca02c', '#1f4e79', '#ff7f0e']       # verde, blu scuro, arancione

# ---------------- fig2
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
ax.hist(df['difference'], bins=100, range=(-100, 100), color='steelblue', alpha=0.7, edgecolor='none')
ax.axvline(0, color='black', lw=1, ls='--')
ax.axvline(df['difference'].mean(), color='red', lw=1.5, label=f"Mean = {df['difference'].mean():.1f}%")
ax.axvline(df['difference'].median(), color='orange', lw=1.5, label=f"Median = {df['difference'].median():.1f}%")
ax.set_xlabel('ERA5 TCC - CloudSEN12+ Cloud Cover (%)'); ax.set_ylabel('Count')
ax.set_title('(a) Distribution of differences'); ax.legend()
ax = axes[1]
bins = np.linspace(-100, 100, 51)
for ct, col in zip(TYPES, COLORS):
    sub = df[df['dominant_type'] == ct]
    ax.hist(sub['difference'], bins=bins, histtype='step', lw=2, color=col, density=True,
            label=f"{ct} (n={len(sub):,})")
ax.axvline(0, color='black', lw=1, ls='--')
ax.set_xlabel('ERA5 TCC - CloudSEN12+ Cloud Cover (%)'); ax.set_ylabel('Density')
ax.set_title('(b) Differences by dominant cloud type'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig2_difference_distribution.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG, 'fig2_difference_distribution.pdf'), bbox_inches='tight')
plt.close()

# ---------------- fig3
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, ct in zip(axes, TYPES):
    sub = df[df['dominant_type'] == ct]
    r = np.corrcoef(sub['era5_tcc_pct'], sub['cloudsen_total_cloud_pct'])[0, 1]
    h = ax.hist2d(sub['cloudsen_total_cloud_pct'], sub['era5_tcc_pct'], bins=50,
                  cmap='YlOrRd', norm=LogNorm(), range=[[0, 100], [0, 100]])
    ax.plot([0, 100], [0, 100], 'k--', lw=0.8)
    ax.set_title(f"{ct}\n(n={len(sub):,}, r={r:.2f})")
    ax.set_xlabel('CloudSEN12+ cloud fraction (%)')
axes[0].set_ylabel('ERA5 TCC (%)')
fig.colorbar(h[3], ax=axes, label='Count', shrink=0.8, pad=0.02)
plt.savefig(os.path.join(FIG, 'fig3_scatter_by_cloudtype.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG, 'fig3_scatter_by_cloudtype.pdf'), bbox_inches='tight')
print('fig2, fig3 rigenerate (senza shadow)')
