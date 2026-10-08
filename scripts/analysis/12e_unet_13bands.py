"""
Script 12e: U-Net con tutte le 13 bande Sentinel-2 L1C (richiesta di Anna, 05/10/2026).

Ripete Model A (label esperte) e Model B (weak label ERA5, soglia 50%, matching ROI+data
corretto) con input a 13 bande invece di 4, 3 seed ciascuno, stesso split (seed 42) e stesso
test set di 12/12b/12c. Confronto diretto con i risultati a 4 bande.

I GeoTIFF contengono le 13 bande nell'ordine B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
(le bande a 20/60 m sono gia' riportate a 10 m nel download).

Output: unet_results/bands13_runs.json, bands13_results.json, figures/fig13_bands.png
Uso:    python 12e_unet_13bands.py [--quick]
"""
import os
import json
import argparse
import importlib.util
import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


u12 = load('unet12', '12_unet_experiment.py')
u12b = load('unet12b', '12b_unet_finetune.py')
u12c = load('unet12c', '12c_weak_label_variants.py')

OUTPUT_DIR = u12.OUTPUT_DIR
FIGURES_DIR = u12.FIGURES_DIR
SEEDS = [0, 1, 2]
CLASSES = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']
BAND_NAMES = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']


def set_bands(n):
    """Riconfigura il modulo 12 (e quindi 12c che lo importa) per 4 o 13 bande."""
    if n == 13:
        u12.BAND_INDICES = list(range(13))
        u12.IN_CHANNELS = 13
    else:
        u12.BAND_INDICES = [1, 2, 3, 7]
        u12.IN_CHANNELS = 4
    # ogni modulo importato ha la PROPRIA copia dello script 12: allineale tutte
    for m in (u12b.u12, u12c.u12, u12c.u12b.u12):
        m.BAND_INDICES = u12.BAND_INDICES
        m.IN_CHANNELS = u12.IN_CHANNELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    seeds = SEEDS[:1] if args.quick else SEEDS
    epochs = 3 if args.quick else u12.NUM_EPOCHS

    set_bands(13)
    pairs = u12b.find_pairs()
    tr, va, te = u12b.split_like_script12(pairs)
    era5 = u12c.era5_dict_exact(pairs)
    test_loader = u12b.make_loader(te)
    val_expert = u12b.make_loader(va)
    print(f"13 bande; coppie {len(pairs)}; train {len(tr)} val {len(va)} test {len(te)}; ERA5 esatti {len(era5)}")

    runs = []
    path = os.path.join(OUTPUT_DIR, 'bands13_runs.json')

    def save():
        with open(path, 'w') as f:
            json.dump(runs, f, indent=1, default=float)

    for seed in seeds:
        # ---- Model A, 13 bande
        tag = f'b13_expert_s{seed}'
        print('\n' + '-' * 60 + f'\nRUN {tag}')
        u12b.set_seed(seed)
        u12.LEARNING_RATE = 1e-4
        model = u12.build_unet(13, u12.NUM_CLASSES)
        model, hist = u12.train_model(model, u12b.make_loader(tr, True), val_expert, epochs, tag)
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f'{tag}_best.pth'), map_location=u12.DEVICE))
        res, _ = u12.evaluate_model(model, test_loader)
        u12.print_results(res, tag)
        runs.append(dict(tag=tag, model='expert', seed=seed, results=res, history=hist)); save()

        # ---- Model B, 13 bande (hard 50%, matching corretto)
        tag = f'b13_era5_s{seed}'
        print('\n' + '-' * 60 + f'\nRUN {tag}')
        u12b.set_seed(seed)
        trl, ntr = u12c.make_weak_loader(tr, era5, 'hard', 50, True)
        val, _ = u12c.make_weak_loader(va, era5, 'hard', 50, False)
        model = u12.build_unet(13, u12.NUM_CLASSES)
        model, hist = u12c.train_weak(model, trl, val, epochs, tag)
        res, _ = u12.evaluate_model(model, test_loader)
        u12.print_results(res, tag)
        runs.append(dict(tag=tag, model='era5', seed=seed, results=res, history=hist, n_train=ntr)); save()

    # ---- riepilogo e confronto con 4 bande
    def summ(sel):
        out = {}
        for m in ['Accuracy', 'Mean IoU']:
            v = [r['results']['Overall'][m] for r in sel]; out[m] = (float(np.mean(v)), float(np.std(v)))
        for c in CLASSES:
            v = [r['results'][c]['IoU'] for r in sel]; out[f'IoU {c}'] = (float(np.mean(v)), float(np.std(v)))
            v = [r['results'][c]['F1'] for r in sel]; out[f'F1 {c}'] = (float(np.mean(v)), float(np.std(v)))
        return out

    s13 = {'expert': summ([r for r in runs if r['model'] == 'expert']),
           'era5': summ([r for r in runs if r['model'] == 'era5'])}
    # 4 bande: expert = scratch 100% di 12b (3 seed), era5 = hard50 di 12c (3 seed)
    ft = json.load(open(os.path.join(OUTPUT_DIR, 'finetune_runs.json')))
    wv = json.load(open(os.path.join(OUTPUT_DIR, 'weak_variants_runs.json')))
    s4 = {'expert': summ([r for r in ft if r['mode'] == 'scratch' and r['fraction'] == 1.0]),
          'era5': summ([r for r in wv if r['kind'] == 'weak' and r.get('thr') == 50])}

    print('\n' + '=' * 78 + '\nRIEPILOGO 4 vs 13 bande (test esperto, media +- std su 3 seed)\n' + '=' * 78)
    print(f"{'config':<18}{'Acc':>14}{'mIoU':>14}{'clear':>8}{'thick':>8}{'thin':>8}{'shadow':>8}")
    for name, S in [('expert 4b', s4['expert']), ('expert 13b', s13['expert']),
                    ('era5 4b', s4['era5']), ('era5 13b', s13['era5'])]:
        print(f"{name:<18}{S['Accuracy'][0]:.3f}+-{S['Accuracy'][1]:.3f} {S['Mean IoU'][0]:.3f}+-{S['Mean IoU'][1]:.3f}  "
              + ' '.join(f"{S[f'IoU {c}'][0]:7.3f}" for c in CLASSES))

    with open(os.path.join(OUTPUT_DIR, 'bands13_results.json'), 'w') as f:
        json.dump({'config': {'seeds': seeds, 'epochs': epochs, 'bands': BAND_NAMES},
                   'bands13': s13, 'bands4': s4, 'runs': runs}, f, indent=1, default=float)

    # ---- figura: IoU per classe, 4 gruppi
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(CLASSES)); w = 0.2
    for k, (name, S, col) in enumerate([('Expert, 4 bands', s4['expert'], '#9ecae1'),
                                        ('Expert, 13 bands', s13['expert'], '#1f4e79'),
                                        ('ERA5 weak, 4 bands', s4['era5'], '#fdae6b'),
                                        ('ERA5 weak, 13 bands', s13['era5'], '#d94801')]):
        mu = [S[f'IoU {c}'][0] for c in CLASSES]; sd = [S[f'IoU {c}'][1] for c in CLASSES]
        ax.bar(x + (k - 1.5) * w, mu, w, yerr=sd, capsize=3, color=col, label=name)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES); ax.set_ylabel('IoU'); ax.set_ylim(0, 1)
    ax.set_title('Per-class IoU on the expert test set: 4-band vs. 13-band input (mean $\\pm$ sd, 3 seeds)')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig13_bands.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print('Salvato', out)


if __name__ == '__main__':
    main()
