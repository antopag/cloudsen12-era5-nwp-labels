"""Script 26: figura della curva di scala delle etichette deboli (fig14)."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = "unet_revision"
FIG = "figures"
df = pd.read_csv(os.path.join(OUT, 'all_runs.csv'))
base = json.load(open(os.path.join(OUT, 'baselines.json')))

w = df[(df['mode'] == 'weak') & (df.bands == 13) & (df.threshold == 50)]
g = w.groupby('n_train').agg(biou_m=('biou', 'mean'), biou_s=('biou', 'std'),
                             miou_m=('miou', 'mean'), miou_s=('miou', 'std')).reset_index()
e = df[(df.file == 'expert_full')]
exp_b, exp_m = e.biou.mean(), e.miou.mean()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, key, lab, expv, basev in [
        (axes[0], 'biou', 'Binary cloud IoU', exp_b, base['era5_as_mask']['Binary']['IoU cloud']),
        (axes[1], 'miou', 'Mean IoU (4 classes)', exp_m, base['era5_as_mask']['Overall']['Mean IoU'])]:
    ax.errorbar(g.n_train, g[key + '_m'], yerr=g[key + '_s'], marker='o', lw=2,
                color='#d94801', capsize=4, label='Trained on ERA5 weak labels')
    ax.axhline(expv, color='#1f4e79', ls='-', lw=2, label='Trained on 8,490 expert labels')
    ax.axhline(basev, color='#7f7f7f', ls='--', lw=1.5, label='ERA5 mask, no training')
    ax.set_xscale('log')
    ax.set_xticks(g.n_train)
    ax.set_xticklabels([f'{int(n):,}' for n in g.n_train])
    ax.set_xlabel('Weakly labelled training patches')
    ax.set_ylabel(lab)
    ax.set_ylim(0, max(expv * 1.15, 0.9))
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, loc='lower right', framealpha=0.95)
fig.suptitle('More weak supervision does not help: the curve is flat over two orders of magnitude')
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(FIG, 'fig14_scale.' + ext), dpi=300, bbox_inches='tight')
print('fig14 salvata |', ' '.join('%d:%.3f' % (n, v) for n, v in zip(g.n_train, g.biou_m)))
