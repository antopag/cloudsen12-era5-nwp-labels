"""
Script 12b: U-Net — ERA5 weak-label pretraining + fine-tuning con label esperte
(risposta al commento di Anna, 15/09/2026: "training con ERA5 = transfer learning?
 allora retraining con dati dello stesso tipo del test").

Confronta, a parita' di budget di label esperte (frazione f del training set):
    SCRATCH  : U-Net da zero, addestrata solo sul sottoinsieme esperto (f)
    FINETUNE : U-Net inizializzata dai pesi ERA5-weak (unet_results/era5weak_best.pth),
               poi rifinita sullo stesso sottoinsieme esperto (f)
Baseline di riferimento (da script 12): expert-only (f=1) e ERA5-only (f=0).
Il test set e' IDENTICO a quello dello script 12 (stesso seed 42, stesso split).

Riusa dataset / modello / train / eval dello script 12 (importato come modulo).

Output:
    unet_results/finetune_runs.json     (salvataggio incrementale, un record per run)
    unet_results/finetune_results.json  (riepilogo mean +/- std)
    unet_results/ft_*.pth               (pesi migliori per ogni run)
    figures/fig11_unet_finetune.png/.pdf

Uso:
    python 12b_unet_finetune.py            # run completo (4 frazioni x 3 seed x 2 modi = 24 training)
    python 12b_unet_finetune.py --quick    # 1 seed, 5 epoche (test del codice)
    python 12b_unet_finetune.py --ft-lr 1e-4 --only-finetune --suffix lr1e4
        # ripete SOLO i run fine-tune con lr 1e-4 (uguale allo scratch); i run scratch
        # vengono riletti da unet_results/finetune_runs.json; output con suffisso _lr1e4
"""

import os
import json
import argparse
import importlib.util
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- import script 12
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("unet12", os.path.join(HERE, "12_unet_experiment.py"))
u12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u12)

# ---------------------------------------------------------------- configurazione
DATA_DIR = "CloudSEN12+_data"             # lo script 12 usava "DATA", che non esiste piu'
OUTPUT_DIR = u12.OUTPUT_DIR
FIGURES_DIR = u12.FIGURES_DIR
PRETRAINED = os.path.join(OUTPUT_DIR, "era5weak_best.pth")

FRACTIONS = [0.10, 0.25, 0.50, 1.00]      # frazione del training set esperto usata
SEEDS = [0, 1, 2]                          # ripetizioni (subset + init diversi)
SCRATCH_LR = 1e-4                          # come script 12
FT_LR = 5e-5                               # lr per fine-tuning (meta' di quello da zero)
CLASSES = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']


# ---------------------------------------------------------------- utilita'
def find_pairs():
    s2_dir = os.path.join(DATA_DIR, "downloaded_images", "s2_images")
    label_dir = os.path.join(DATA_DIR, "downloaded_images", "labels")
    pairs = []
    for f in sorted(os.listdir(s2_dir)):
        if not f.endswith('.tif'):
            continue
        name = f[:-4]
        lp = os.path.join(label_dir, name + '_label.tif')
        if os.path.exists(lp):
            pairs.append({'name': name, 's2_path': os.path.join(s2_dir, f), 'label_path': lp})
    return pairs


def split_like_script12(pairs):
    """Stesso split (seed 42, 70/15/15) dello script 12 -> stesso test set."""
    np.random.seed(42)
    idx = np.random.permutation(len(pairs))
    n = len(pairs)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    return ([pairs[i] for i in idx[:n_tr]],
            [pairs[i] for i in idx[n_tr:n_tr + n_va]],
            [pairs[i] for i in idx[n_tr + n_va:]])


def make_loader(pair_list, shuffle=False):
    ds = u12.CloudGeoTIFFDataset([p['s2_path'] for p in pair_list],
                                 [p['label_path'] for p in pair_list],
                                 u12.IMG_SIZE, use_era5_labels=False)
    return torch.utils.data.DataLoader(ds, batch_size=u12.BATCH_SIZE,
                                       shuffle=shuffle, num_workers=u12.NUM_WORKERS)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_pretrained(model):
    state = torch.load(PRETRAINED, map_location=u12.DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  ATTENZIONE: chiavi mancanti={len(missing)} inattese={len(unexpected)}")
    return model


def run_one(train_subset, val_loader, test_loader, mode, tag, epochs):
    """mode: 'scratch' | 'finetune'. Ritorna (results, history)."""
    model = u12.build_unet(u12.IN_CHANNELS, u12.NUM_CLASSES)
    if mode == 'finetune':
        model = load_pretrained(model)
        u12.LEARNING_RATE = FT_LR       # train_model legge la globale del modulo 12
    else:
        u12.LEARNING_RATE = SCRATCH_LR
    train_loader = make_loader(train_subset, shuffle=True)
    model, hist = u12.train_model(model, train_loader, val_loader, epochs, f"ft_{tag}")
    # valuta il checkpoint migliore (val loss) per coerenza tra run
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"ft_{tag}_best.pth"),
                                     map_location=u12.DEVICE))
    res, _ = u12.evaluate_model(model, test_loader)
    return res, hist


# ---------------------------------------------------------------- figura
def plot_finetune(summary, baseline, out_png):
    fr = [s['fraction'] for s in summary]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, key, lab in zip(axes, ['Accuracy', 'Mean IoU'], ['Overall accuracy', 'Mean IoU']):
        for mode, col, mk, name in [('scratch', 'steelblue', 'o', 'Expert only (scratch)'),
                                    ('finetune', 'darkorange', 's', 'ERA5 pretrain + expert fine-tune')]:
            m = [s[mode][key]['mean'] for s in summary]
            sd = [s[mode][key]['std'] for s in summary]
            ax.errorbar(fr, m, yerr=sd, marker=mk, color=col, lw=2, capsize=4, label=name)
        if baseline:
            ax.axhline(baseline['era5_weak'][key], color='indianred', ls='--',
                       label='ERA5 weak only (script 12)')
            ax.axhline(baseline['expert'][key], color='steelblue', ls=':',
                       label='Expert 100% (script 12)')
        ax.set_xlabel('Fraction of expert-labelled training patches')
        ax.set_ylabel(lab)
        ax.set_xscale('log')
        ax.set_xticks(fr)
        ax.set_xticklabels([f'{f:.0%}' for f in fr])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('U-Net: value of ERA5 weak-label pretraining vs. expert-label budget')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {out_png}")


# ---------------------------------------------------------------- main
def main():
    global FT_LR, PRETRAINED
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='1 seed, 5 epoche')
    ap.add_argument('--epochs', type=int, default=u12.NUM_EPOCHS)
    ap.add_argument('--ft-lr', type=float, default=FT_LR, help='learning rate fine-tune')
    ap.add_argument('--only-finetune', action='store_true',
                    help='salta i run scratch e li rilegge da finetune_runs.json')
    ap.add_argument('--suffix', default='', help='suffisso per i file di output (es. lr1e4)')
    ap.add_argument('--pretrained', default=None, help='file .pth di pretraining (default era5weak_best.pth)')
    ap.add_argument('--bands', type=int, default=4, choices=[4, 13], help='bande in ingresso (4 o 13)')
    args = ap.parse_args()
    seeds = SEEDS[:1] if args.quick else SEEDS
    FT_LR = args.ft_lr
    if args.bands == 13:
        u12.BAND_INDICES = list(range(13))
        u12.IN_CHANNELS = 13
    print(f'  bande in ingresso: {u12.IN_CHANNELS}')
    if args.pretrained:
        PRETRAINED = args.pretrained
    print(f'  pretraining: {PRETRAINED}')
    sfx = f'_{args.suffix}' if args.suffix else ''
    modes = ['finetune'] if args.only_finetune else ['scratch', 'finetune']
    print(f'  fine-tune lr = {FT_LR:g}; modes = {modes}; suffix = {sfx!r}')
    epochs = 5 if args.quick else args.epochs

    print("=" * 60)
    print("ESPERIMENTO U-NET 12b: ERA5 PRETRAIN + EXPERT FINE-TUNE")
    print("=" * 60)
    if not os.path.exists(PRETRAINED):
        print(f"ERRORE: pesi ERA5 non trovati: {PRETRAINED} (lancia prima lo script 12)")
        return

    pairs = find_pairs()
    train_pairs, val_pairs, test_pairs = split_like_script12(pairs)
    print(f"  Coppie: {len(pairs)}  split train={len(train_pairs)} "
          f"val={len(val_pairs)} test={len(test_pairs)}")
    val_loader = make_loader(val_pairs)
    test_loader = make_loader(test_pairs)

    baseline = None
    bl_path = os.path.join(OUTPUT_DIR, 'unet_results.json')
    if os.path.exists(bl_path):
        with open(bl_path) as f:
            b = json.load(f)
        baseline = {k: b[k]['Overall'] for k in ['expert', 'era5_weak']}
        print(f"  Baseline script 12: {baseline}")

    runs = []   # un record per run
    for frac in FRACTIONS:
        n_sub = max(u12.BATCH_SIZE, int(round(frac * len(train_pairs))))
        for seed in seeds:
            rng = np.random.RandomState(100 + seed)
            subset = [train_pairs[i] for i in rng.permutation(len(train_pairs))[:n_sub]]
            for mode in modes:
                tag = f"{mode}_f{int(frac * 100):03d}_s{seed}{sfx}"
                print("\n" + "-" * 60)
                print(f"RUN {tag}: {n_sub} patch esperte, {epochs} epoche")
                set_seed(seed)
                res, hist = run_one(subset, val_loader, test_loader, mode, tag, epochs)
                u12.print_results(res, tag)
                runs.append({'tag': tag, 'mode': mode, 'fraction': frac, 'n_train': n_sub,
                             'seed': seed, 'results': res, 'history': hist})
                # salvataggio incrementale (i run sono lunghi)
                with open(os.path.join(OUTPUT_DIR, f'finetune_runs{sfx}.json'), 'w') as f:
                    json.dump(runs, f, indent=1, default=float)

    # in modalita' only-finetune i run scratch vengono riletti ORA dal run precedente
    # (cosi' i due run possono girare in parallelo: il file deve essere completo solo qui)
    if args.only_finetune:
        prev = os.path.join(OUTPUT_DIR, 'finetune_runs.json')
        with open(prev) as f:
            scratch_runs = [r for r in json.load(f) if r['mode'] == 'scratch']
        print(f'  Run scratch riletti da {prev}: {len(scratch_runs)}')
        runs = scratch_runs + runs

    # riepilogo mean +/- std per (fraction, mode)
    summary = []
    for frac in FRACTIONS:
        row = {'fraction': frac}
        for mode in ['scratch', 'finetune']:
            sel = [r for r in runs if r['fraction'] == frac and r['mode'] == mode]
            row['n_train'] = sel[0]['n_train']
            row[mode] = {}
            for key in ['Accuracy', 'Mean IoU']:
                v = [r['results']['Overall'][key] for r in sel]
                row[mode][key] = {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)}
            for c in CLASSES:
                v = [r['results'][c]['IoU'] for r in sel]
                row[mode][f'IoU {c}'] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}
        summary.append(row)

    print("\n" + "=" * 60)
    print("RIEPILOGO (test set esperto, media +/- std sui seed)")
    print("=" * 60)
    print(f"  {'frac':>6} {'n_tr':>5} | {'scratch Acc':>13} {'scratch mIoU':>13} | "
          f"{'finetune Acc':>13} {'finetune mIoU':>13} | {'dAcc':>7} {'dmIoU':>7}")
    for s in summary:
        sa, sm = s['scratch']['Accuracy'], s['scratch']['Mean IoU']
        fa, fm = s['finetune']['Accuracy'], s['finetune']['Mean IoU']
        print(f"  {s['fraction']:>6.0%} {s['n_train']:>5} | "
              f"{sa['mean']:.3f}+-{sa['std']:.3f}   {sm['mean']:.3f}+-{sm['std']:.3f} | "
              f"{fa['mean']:.3f}+-{fa['std']:.3f}   {fm['mean']:.3f}+-{fm['std']:.3f} | "
              f"{fa['mean'] - sa['mean']:>+7.3f} {fm['mean'] - sm['mean']:>+7.3f}")
    if baseline:
        print(f"  ERA5-only (script 12): Acc {baseline['era5_weak']['Accuracy']:.3f} "
              f"mIoU {baseline['era5_weak']['Mean IoU']:.3f}")

    out = {'config': {'fractions': FRACTIONS, 'seeds': seeds, 'epochs': epochs,
                      'ft_lr': FT_LR, 'scratch_lr': SCRATCH_LR, 'pretrained': PRETRAINED,
                      'n_train_full': len(train_pairs), 'n_test': len(test_pairs)},
           'baseline_script12': baseline, 'summary': summary, 'runs': runs}
    with open(os.path.join(OUTPUT_DIR, f'finetune_results{sfx}.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)

    plot_finetune(summary, baseline, os.path.join(FIGURES_DIR, f'fig11_unet_finetune{sfx}.png'))
    print(f"\nRisultati in: {OUTPUT_DIR}/finetune_results{sfx}.json")


if __name__ == '__main__':
    main()
