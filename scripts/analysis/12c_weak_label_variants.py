"""
Script 12c: varianti di weak label ERA5 per la U-Net (risposta al reviewer, punti 6 e 8).

Rispetto allo script 12:
  * CORREZIONE del matching ERA5: per ROI **e data** (lo script 12 usava solo il roi_id come
    sottostringa, quindi immagini dello stesso ROI a date diverse ricevevano lo stesso TCC).
    Le patch senza riga ERA5 esatta (~16/306) sono escluse SOLO dal training con weak label.
  * Varianti di weak label (tutte con testa a 4 classi, come Model B):
      hard-T : label binaria patch-costante, cloudy se TCC >= T, T in {30,40,50,60,70}
      soft   : target per pixel p = [1-TCC, TCC, 0, 0] (probabilita'), cross-entropy su distribuzioni
  * Fine-tune dal modello soft su 10% e 25% di label esperte (stesso protocollo di 12b), per
    verificare se un pretraining piu' informativo recupera thin cloud / shadow.
  * 3 seed per configurazione; stesso split (seed 42) e stesso test set esperto di 12 e 12b.

Output: unet_results/weak_variants_runs.json (incrementale), weak_variants_results.json,
        figures/fig12_weak_variants.png
Uso:   python 12c_weak_label_variants.py [--quick]
"""
import os
import json
import argparse
import importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import rasterio as rio
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("unet12", os.path.join(HERE, "12_unet_experiment.py"))
u12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u12)
spec_b = importlib.util.spec_from_file_location("unet12b", os.path.join(HERE, "12b_unet_finetune.py"))
u12b = importlib.util.module_from_spec(spec_b)
spec_b.loader.exec_module(u12b)

OUTPUT_DIR = u12.OUTPUT_DIR
FIGURES_DIR = u12.FIGURES_DIR
THRESHOLDS = [30, 40, 50, 60, 70]
SEEDS = [0, 1, 2]
FT_FRACTIONS = [0.10, 0.25]
CLASSES = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']


# ------------------------------------------------------------------ ERA5 matching corretto
def era5_dict_exact(pairs):
    df = pd.read_csv('era5_data/era5_vs_cloudsen12.csv')
    df['key'] = df.roi_id + '_' + df.s2_date.str.replace('-', '').str[:8]
    lut = dict(zip(df.key, df.era5_tcc_pct))
    out = {}
    for p in pairs:
        parts = p['name'].split('_')            # ROI_0061_20190130T013719_...
        key = f"{parts[0]}_{parts[1]}_{parts[2][:8]}"
        if key in lut:
            out[p['name']] = float(lut[key])
    return out


# ------------------------------------------------------------------ dataset con weak label
class WeakDataset(u12.CloudGeoTIFFDataset):
    """mode: 'hard' (soglia thr) o 'soft' (target probabilistico 4xHxW)."""

    def __init__(self, s2_paths, era5_dict, mode='hard', thr=50.0):
        super().__init__(s2_paths, [None] * len(s2_paths), u12.IMG_SIZE,
                         use_era5_labels=True, era5_dict=era5_dict, era5_threshold=thr)
        self.mode = mode

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)            # img gia' 4xSxS
        tcc = self.era5_dict[os.path.basename(self.s2_paths[idx])[:-4]] / 100.0
        S = self.img_size
        if self.mode == 'hard':
            cls = 1 if tcc >= self.era5_threshold / 100.0 else 0
            return img, torch.full((S, S), cls, dtype=torch.long)
        tgt = torch.zeros(4, S, S)
        tgt[0] = 1.0 - tcc
        tgt[1] = tcc
        return img, tgt


def make_weak_loader(pairs, era5, mode, thr, shuffle):
    pairs = [p for p in pairs if p['name'] in era5]
    ds = WeakDataset([p['s2_path'] for p in pairs], era5, mode, thr)
    return torch.utils.data.DataLoader(ds, batch_size=u12.BATCH_SIZE, shuffle=shuffle,
                                       num_workers=u12.NUM_WORKERS), len(pairs)


def train_weak(model, train_loader, val_loader, epochs, name):
    """Come u12.train_model ma con CE che accetta anche target probabilistici."""
    model = model.to(u12.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    best = float('inf')
    hist = {'train_loss': [], 'val_loss': []}
    for ep in range(epochs):
        model.train()
        tl, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(u12.DEVICE), y.to(u12.DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item(); nb += 1
        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(u12.DEVICE), y.to(u12.DEVICE)
                vl += crit(model(x), y).item(); nv += 1
        tl /= max(nb, 1); vl /= max(nv, 1)
        sched.step(vl)
        hist['train_loss'].append(tl); hist['val_loss'].append(vl)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1:>3}/{epochs} | TrLoss {tl:.4f} | VaLoss {vl:.4f}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{name}_best.pth'))
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'{name}_best.pth'), map_location=u12.DEVICE))
    return model, hist


def rec(tag, kind, res, **kw):
    return dict(tag=tag, kind=kind, results=res, **kw)


def summarize(runs, kind, key):
    sel = [r for r in runs if r['kind'] == kind and r.get(key[0]) == key[1]]
    out = {}
    for m in ['Accuracy', 'Mean IoU']:
        v = [r['results']['Overall'][m] for r in sel]
        out[m] = (float(np.mean(v)), float(np.std(v)), len(v))
    for c in CLASSES:
        v = [r['results'][c]['IoU'] for r in sel]
        out[f'IoU {c}'] = (float(np.mean(v)), float(np.std(v)), len(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--bands', type=int, default=4, choices=[4, 13], help='bande in ingresso (4 o 13)')
    ap.add_argument('--suffix', default='', help='suffisso per i file di output (es. b13)')
    args = ap.parse_args()
    seeds = SEEDS[:1] if args.quick else SEEDS
    sfx = f'_{args.suffix}' if args.suffix else ''
    if args.bands == 13:
        for m in (u12, u12b.u12):
            m.BAND_INDICES = list(range(13))
            m.IN_CHANNELS = 13
    print(f'  bande in ingresso: {u12.IN_CHANNELS}; suffix {sfx!r}')
    epochs = 3 if args.quick else u12.NUM_EPOCHS

    pairs = u12b.find_pairs()
    tr, va, te = u12b.split_like_script12(pairs)
    era5 = era5_dict_exact(pairs)
    print(f"Coppie {len(pairs)}; ERA5 esatti {len(era5)}/{len(pairs)}; "
          f"train con ERA5: {sum(p['name'] in era5 for p in tr)}/{len(tr)}")
    tcc_tr = np.array([era5[p['name']] for p in tr if p['name'] in era5])
    print('TCC ERA5 sul train: quantili 10/25/50/75/90 =', np.percentile(tcc_tr, [10, 25, 50, 75, 90]).round(1))
    test_loader = u12b.make_loader(te)
    val_expert = u12b.make_loader(va)

    runs = []
    runs_path = os.path.join(OUTPUT_DIR, f'weak_variants_runs{sfx}.json')

    def save():
        with open(runs_path, 'w') as f:
            json.dump(runs, f, indent=1, default=float)

    # ---------- 1. soglie hard + soft (Model B corretto)
    configs = [('hard', t) for t in THRESHOLDS] + [('soft', None)]
    for mode, thr in configs:
        for seed in seeds:
            tag = f"weak_{mode}{'' if thr is None else int(thr)}_s{seed}{sfx}"
            print('\n' + '-' * 60 + f'\nRUN {tag}')
            u12b.set_seed(seed)
            trl, ntr = make_weak_loader(tr, era5, mode, thr or 50, True)
            val, _ = make_weak_loader(va, era5, mode, thr or 50, False)
            model = u12.build_unet(u12.IN_CHANNELS, u12.NUM_CLASSES)
            model, hist = train_weak(model, trl, val, epochs, tag)
            res, _ = u12.evaluate_model(model, test_loader)
            u12.print_results(res, tag)
            runs.append(rec(tag, 'weak', res, mode=mode, thr=thr, seed=seed, n_train=ntr, history=hist))
            save()

    # ---------- 2. fine-tune dal modello soft (seed 0 come pretrain) su 10% / 25%
    for frac in FT_FRACTIONS:
        n_sub = max(u12.BATCH_SIZE, int(round(frac * len(tr))))
        for seed in seeds:
            tag = f"ftsoft_f{int(frac*100):03d}_s{seed}{sfx}"
            print('\n' + '-' * 60 + f'\nRUN {tag}: {n_sub} patch esperte')
            rng = np.random.RandomState(100 + seed)                 # stessi subset di 12b
            subset = [tr[i] for i in rng.permutation(len(tr))[:n_sub]]
            u12b.set_seed(seed)
            model = u12.build_unet(u12.IN_CHANNELS, u12.NUM_CLASSES)
            model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'weak_soft_s{seed}{sfx}_best.pth'),
                                             map_location=u12.DEVICE))
            u12.LEARNING_RATE = 5e-5
            model, hist = u12.train_model(model, u12b.make_loader(subset, True), val_expert, epochs, tag)
            model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'{tag}_best.pth'), map_location=u12.DEVICE))
            res, _ = u12.evaluate_model(model, test_loader)
            u12.print_results(res, tag)
            runs.append(rec(tag, 'ftsoft', res, fraction=frac, seed=seed, n_train=n_sub, history=hist))
            save()

    # ---------- riepilogo
    summary = {'hard': {}, 'soft': None, 'ftsoft': {}}
    print('\n' + '=' * 70 + '\nRIEPILOGO (test esperto, media +- std su seed)\n' + '=' * 70)
    print(f"{'config':<14}{'Acc':>14}{'mIoU':>14}{'IoU clear':>11}{'thick':>8}{'thin':>8}{'shadow':>8}")

    def line(name, sm):
        print(f"{name:<14}{sm['Accuracy'][0]:.3f}+-{sm['Accuracy'][1]:.3f} "
              f"{sm['Mean IoU'][0]:.3f}+-{sm['Mean IoU'][1]:.3f}   "
              + ' '.join(f"{sm[f'IoU {c}'][0]:7.3f}" for c in CLASSES))
    for t in THRESHOLDS:
        sm = summarize(runs, 'weak', ('thr', t)); summary['hard'][t] = sm; line(f'hard {t}%', sm)
    sm = summarize(runs, 'weak', ('mode', 'soft')); summary['soft'] = sm; line('soft', sm)
    for f in FT_FRACTIONS:
        sm = summarize(runs, 'ftsoft', ('fraction', f)); summary['ftsoft'][f] = sm; line(f'soft->ft {f:.0%}', sm)

    with open(os.path.join(OUTPUT_DIR, f'weak_variants_results{sfx}.json'), 'w') as fh:
        json.dump({'config': {'thresholds': THRESHOLDS, 'seeds': seeds, 'epochs': epochs,
                              'n_era5_matched': len(era5)}, 'summary': summary, 'runs': runs},
                  fh, indent=1, default=float)

    # ---------- figura: acc e mIoU vs soglia, con soft come linea
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, m in zip(axes, ['Accuracy', 'Mean IoU']):
        mu = [summary['hard'][t][m][0] for t in THRESHOLDS]
        sd = [summary['hard'][t][m][1] for t in THRESHOLDS]
        ax.errorbar(THRESHOLDS, mu, yerr=sd, marker='o', color='indianred', capsize=4, lw=2,
                    label='Hard binary weak label (threshold)')
        ax.axhline(summary['soft'][m][0], color='darkorange', ls='--', lw=2, label='Soft target (ERA5 fraction)')
        ax.axhspan(summary['soft'][m][0] - summary['soft'][m][1], summary['soft'][m][0] + summary['soft'][m][1],
                   color='darkorange', alpha=0.15)
        ax.set_xlabel('ERA5 TCC threshold for "cloudy" (%)')
        ax.set_ylabel(m if m == 'Accuracy' else 'Mean IoU')
        ax.set_xticks(THRESHOLDS); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('U-Net trained on ERA5 weak labels only: sensitivity to label formulation')
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f'fig12_weak_variants{sfx}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print('Salvato', out)


if __name__ == '__main__':
    main()
