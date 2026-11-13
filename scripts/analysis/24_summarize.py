"""
Script 24: aggrega i risultati della revisione in tabelle pronte per il manoscritto.
Legge unet_revision/runs_*.json e baselines.json, stampa media +/- deviazione standard
sui seed e salva un CSV per ogni tabella.
"""
import os
import json
import glob
import numpy as np
import pandas as pd

OUT = "unet_revision"
CLASSES = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']


def load_runs():
    rows = []
    for f in glob.glob(os.path.join(OUT, 'runs_*.json')):
        tag = os.path.basename(f)[5:-5]
        if tag == 'expert_ssdtest':
            continue
        for r in json.load(open(f)):
            m = r['results']
            row = dict(file=tag, mode=r['mode'], n_train=r['n_train'], bands=r['bands'],
                       seed=r['seed'], threshold=r.get('threshold', 50), minutes=r.get('minutes'),
                       acc=m['Overall']['Accuracy'], miou=m['Overall']['Mean IoU'],
                       biou=m['Binary']['IoU cloud'], bacc=m['Binary']['Accuracy'])
            for c in CLASSES:
                row['iou_' + c.split()[0].lower()] = m[c]['IoU']
            s = m.get('Scene', {})
            row['scene_acc'] = s.get('Accuracy')
            row['scene_bacc'] = s.get('Balanced accuracy')
            row['scene_mae'] = s.get('MAE cloud fraction')
            row['scene_auc'] = s.get('AUC')
            rows.append(row)
    return pd.DataFrame(rows)


def agg(df, by):
    g = df.groupby(by)
    out = g.agg(n_seeds=('seed', 'size'), **{
        k + '_m': (k, 'mean') for k in ['acc', 'miou', 'biou', 'bacc', 'scene_acc',
                                        'iou_clear', 'iou_thick', 'iou_thin', 'iou_shadow']})
    for k in ['acc', 'miou', 'biou', 'bacc', 'scene_acc']:
        out[k + '_s'] = g[k].std()
    return out.reset_index()


def fmt(m, s, d=3):
    return ("%.*f" % (d, m)) + (u" \u00b1 %.*f" % (d, s) if np.isfinite(s) else "")


def main():
    df = load_runs()
    df.to_csv(os.path.join(OUT, 'all_runs.csv'), index=False)
    print("run totali: %d\n" % len(df))

    base = json.load(open(os.path.join(OUT, 'baselines.json')))
    print("=" * 96)
    print("BASELINE SENZA ADDESTRAMENTO")
    print("=" * 96)
    print("%-34s %6s %6s %7s %7s" % ("", "acc", "mIoU", "IoU nub", "acc bin"))
    for k, v in base.items():
        print("%-34s %6.3f %6.3f %7.3f %7.3f" % (k, v['Overall']['Accuracy'], v['Overall']['Mean IoU'],
                                                 v['Binary']['IoU cloud'], v['Binary']['Accuracy']))

    # ---- tabella 1: confronto principale a 13 bande
    main13 = df[(df.bands == 13) & (df.file.isin(['expert_full', 'weak_scale']))]
    t1 = agg(main13, ['mode', 'n_train'])
    print("\n" + "=" * 96)
    print("TABELLA 1 - CONFRONTO PRINCIPALE (13 bande, split ufficiale per ROI)")
    print("=" * 96)
    print("%-26s %5s %14s %14s %14s %14s" % ("modello", "n", "accuratezza", "mIoU", "IoU nuvola", "acc. scena"))
    for _, r in t1.iterrows():
        nome = "esperto" if r['mode'] == 'expert' else "ERA5 (debole)"
        print("%-26s %5d %14s %14s %14s %14s" % (
            nome, r['n_train'], fmt(r['acc_m'], r['acc_s']), fmt(r['miou_m'], r['miou_s']),
            fmt(r['biou_m'], r['biou_s']), fmt(r['scene_acc_m'], r['scene_acc_s'])))

    # ---- tabella 2: scala delle etichette deboli
    scale = df[(df['mode'] == 'weak') & (df.bands == 13) & (df.threshold == 50) &
               (df.file.str.startswith(('weak_n', 'weak_scale')))]
    t2 = agg(scale, ['n_train'])
    print("\n" + "=" * 96)
    print("TABELLA 2 - SCALA DELLE ETICHETTE DEBOLI (la quantita' non aiuta)")
    print("=" * 96)
    print("%-10s %14s %14s %14s %14s" % ("n patch", "accuratezza", "mIoU", "IoU nuvola", "acc. scena"))
    for _, r in t2.iterrows():
        print("%-10d %14s %14s %14s %14s" % (
            r['n_train'], fmt(r['acc_m'], r['acc_s']), fmt(r['miou_m'], r['miou_s']),
            fmt(r['biou_m'], r['biou_s']), fmt(r['scene_acc_m'], r['scene_acc_s'])))

    # ---- tabella 3: budget esperto, da zero vs fine-tuning
    sc = df[df.file.str.startswith('expert_b') & (df.bands == 13)].copy(); sc['kind'] = 'da zero'
    ft = df[df.file.str.startswith('expert_ft') & (df.bands == 13)].copy(); ft['kind'] = 'fine-tune'
    full = df[(df.file == 'expert_full')].copy(); full['kind'] = 'da zero'
    bud = pd.concat([sc, ft, full])
    t3 = agg(bud, ['kind', 'n_train'])
    print("\n" + "=" * 96)
    print("TABELLA 3 - BUDGET DI ETICHETTE ESPERTE: DA ZERO vs FINE-TUNING DA ERA5")
    print("=" * 96)
    print("%-6s %12s %14s %14s %14s" % ("n", "modo", "accuratezza", "mIoU", "IoU nuvola"))
    for n in sorted(t3.n_train.unique()):
        for kind in ['da zero', 'fine-tune']:
            r = t3[(t3.n_train == n) & (t3.kind == kind)]
            if len(r):
                r = r.iloc[0]
                print("%-6d %12s %14s %14s %14s" % (n, kind, fmt(r['acc_m'], r['acc_s']),
                                                    fmt(r['miou_m'], r['miou_s']), fmt(r['biou_m'], r['biou_s'])))

    # ---- tabella 4: soglia di binarizzazione
    thr = df[(df['mode'] == 'weak') & (df.n_train == 5000) & (df.bands == 13)]
    t4 = agg(thr, ['threshold'])
    print("\n" + "=" * 96)
    print("TABELLA 4 - SOGLIA DI BINARIZZAZIONE (5.000 patch)")
    print("=" * 96)
    print("%-8s %14s %14s %14s" % ("soglia", "accuratezza", "mIoU", "IoU nuvola"))
    for _, r in t4.iterrows():
        print("%-8.0f %14s %14s %14s" % (r['threshold'], fmt(r['acc_m'], r['acc_s']),
                                         fmt(r['miou_m'], r['miou_s']), fmt(r['biou_m'], r['biou_s'])))

    # ---- tabella 5: 4 vs 13 bande
    ab = df[df.file.str.endswith('_b4') | df.file.isin(['expert_full', 'weak_scale'])]
    t5 = agg(ab, ['mode', 'bands'])
    print("\n" + "=" * 96)
    print("TABELLA 5 - ABLAZIONE 4 vs 13 BANDE")
    print("=" * 96)
    print("%-16s %6s %14s %14s %14s" % ("modello", "bande", "accuratezza", "mIoU", "IoU nuvola"))
    for _, r in t5.iterrows():
        nome = "esperto" if r['mode'] == 'expert' else "ERA5 (debole)"
        print("%-16s %6d %14s %14s %14s" % (nome, r['bands'], fmt(r['acc_m'], r['acc_s']),
                                            fmt(r['miou_m'], r['miou_s']), fmt(r['biou_m'], r['biou_s'])))

    # ---- per classe, confronto principale
    print("\n" + "=" * 96)
    print("PER CLASSE (13 bande, media sui seed)")
    print("=" * 96)
    pc = agg(main13, ['mode'])
    print("%-16s %10s %10s %10s %10s" % ("modello", "clear", "thick", "thin", "shadow"))
    for _, r in pc.iterrows():
        nome = "esperto" if r['mode'] == 'expert' else "ERA5 (debole)"
        print("%-16s %10.3f %10.3f %10.3f %10.3f" % (nome, r['iou_clear_m'], r['iou_thick_m'],
                                                     r['iou_thin_m'], r['iou_shadow_m']))

    for name, t in [('tab1_principale', t1), ('tab2_scala', t2), ('tab3_budget', t3),
                    ('tab4_soglia', t4), ('tab5_bande', t5)]:
        t.to_csv(os.path.join(OUT, name + '.csv'), index=False)
    print("\nCSV salvati in", OUT)


if __name__ == '__main__':
    main()
