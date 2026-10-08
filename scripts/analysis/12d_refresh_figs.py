"""
Script 12d: rigenera fig9 (curve di training) e fig10 (IoU per classe) con il Modello B
CORRETTO (matching ERA5 per ROI+data, soglia 50%, media 3 seed da 12c) al posto del run
originale dello script 12 (matching per solo ROI).
Model A resta quello dello script 12 (unet_results/unet_results.json).
"""
import json
import numpy as np
import matplotlib.pyplot as plt

A = json.load(open('unet_results/unet_results.json'))
W = json.load(open('unet_results/weak_variants_results.json'))
B_runs = [r for r in W['runs'] if r['kind'] == 'weak' and r.get('thr') == 50]
classes = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']

# ---------- fig9: loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.plot(A['history_expert']['train_loss'], 'b-', alpha=0.5, label='Expert (train)')
ax.plot(A['history_expert']['val_loss'], 'b-', lw=2, label='Expert (val)')
tl = np.mean([r['history']['train_loss'] for r in B_runs], axis=0)
vl = np.mean([r['history']['val_loss'] for r in B_runs], axis=0)
ax.plot(tl, 'r-', alpha=0.5, label='ERA5 weak (train, mean of 3 seeds)')
ax.plot(vl, 'r-', lw=2, label='ERA5 weak (val, mean of 3 seeds)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy loss'); ax.set_title('(a) Training loss'); ax.legend(fontsize=8)
ax = axes[1]
ax.plot(A['history_expert']['val_acc'], 'b-', lw=2, label='Expert')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('(b) Validation accuracy (expert labels)'); ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('figures/fig9_unet_training.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig9_unet_training.pdf', bbox_inches='tight')
plt.close()

# ---------- fig10: per-class IoU bars with seed std for B
iou_a = [A['expert'][c]['IoU'] for c in classes]
iou_b = [np.mean([r['results'][c]['IoU'] for r in B_runs]) for c in classes]
sd_b = [np.std([r['results'][c]['IoU'] for r in B_runs]) for c in classes]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(classes)); w = 0.35
b1 = ax.bar(x - w/2, iou_a, w, label='Model A: expert labels', color='steelblue')
b2 = ax.bar(x + w/2, iou_b, w, yerr=sd_b, capsize=4, label='Model B: ERA5 weak labels (mean $\\pm$ sd, 3 seeds)', color='indianred')
ax.set_ylabel('IoU'); ax.set_title('U-Net: expert vs. ERA5 weak labels, per-class IoU on the expert test set')
ax.set_xticks(x); ax.set_xticklabels(classes); ax.legend(); ax.set_ylim(0, 1)
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f'{h:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig10_unet_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig10_unet_comparison.pdf', bbox_inches='tight')
print('fig9/fig10 rigenerate; B IoU =', np.round(iou_b, 3))
