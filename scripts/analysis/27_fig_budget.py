"""Script 27: figura del budget di etichette esperte, da zero vs fine-tuning da ERA5 (fig15)."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = "unet_revision"
FIG = "figures"
df = pd.read_csv(os.path.join(OUT, 'all_runs.csv'))
base = json.load(open(os.path.join(OUT, 'baselines.json')))

sc = df[(df.file.str.startswith('expert_b') | (df.file == 'expert_full')) & (df.bands == 13)].copy()
ft = df[df.file.str.startswith('expert_ft') & (df.bands == 13)].copy()
weak = df[(df.file == 'weak_scale')]


def agg(d):
    return d.groupby('n_train').agg(m=('biou', 'mean'), s=('biou', 'std'),
                                    mi=('miou', 'mean'), si=('miou', 'std')).reset_index()


a, b = agg(sc), agg(ft)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, (km, ks), lab in [(axes[0], ('m', 's'), 'Binary cloud IoU'),
                          (axes[1], ('mi', 'si'), 'Mean IoU (4 classes)')]:
    ax.errorbar(a.n_train, a[km], yerr=a[ks], marker='o', lw=2, capsize=4,
                color='#1f4e79', label='From scratch')
    ax.errorbar(b.n_train, b[km], yerr=b[ks], marker='s', lw=2, capsize=4,
                color='#d94801', label='Fine-tuned from ERA5 pretraining')
    v = weak.biou.mean() if km == 'm' else weak.miou.mean()
    ax.axhline(v, color='#7f7f7f', ls='--', lw=1.5, label='ERA5 weak labels only (20,000 patches)')
    ax.set_xscale('log')
    ax.set_xticks(a.n_train)
    ax.set_xticklabels([f'{int(n):,}' for n in a.n_train])
    ax.set_xlabel('Expert-labelled training patches')
    ax.set_ylabel(lab)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.9)
axes[0].legend(fontsize=8, loc='lower right', framealpha=0.95)
fig.suptitle('Pretraining on ERA5 labels is worse than random initialisation at every budget')
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(FIG, 'fig15_budget.' + ext), dpi=300, bbox_inches='tight')
print('fig15 salvata')
print('da zero  :', ' '.join('%d:%.3f' % (n, v) for n, v in zip(a.n_train, a.m)))
print('fine-tune:', ' '.join('%d:%.3f' % (n, v) for n, v in zip(b.n_train, b.m)))
