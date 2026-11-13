"""
Script 21: esperimenti U-Net per la revisione. Sostituisce 12/12b/12c/12e.

Correzioni rispetto alla versione sottomessa:
  1. SPLIT UFFICIALE CloudSEN12+ (disgiunto per roi_id). Prima lo split era casuale per
     patch: il 96% delle patch di test proveniva da ROI presenti anche nel training.
  2. I pixel NON ANNOTATI (valore 99) sono ignorati in addestramento (ignore_index) e
     esclusi dalla valutazione. Prima venivano troncati a 3 e quindi contati come ombra.
  3. Niente ridimensionamento delle maschere: in addestramento si ritaglia a 256x256,
     la valutazione e' a piena risoluzione 512x512 (R1 punto 5).
  4. Metriche primarie BINARIE e di SCENA (editor, R1 punto 2, R2 punto 5), accanto a
     quelle per classe: la mIoU a 4 classi penalizza il modello debole su due classi che
     non puo' predire per costruzione.
  5. Baseline senza addestramento (editor): classe maggioritaria e ERA5 usata direttamente
     come maschera costante.
  6. Scala: il modello debole si addestra su decine di migliaia di patch, non su 204.

Dati attesi in DATA_DIR (prodotti da 20_download_scaleup.py).

Uso:
    python 21_unet_revision.py --mode baselines
    python 21_unet_revision.py --mode expert --n-train 8490 --seeds 0 1 2 3 4
    python 21_unet_revision.py --mode weak   --n-train 20000 --seeds 0 1 2
    python 21_unet_revision.py --mode finetune --n-train 500 --pretrained <file.pth>
"""
import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import rasterio as rio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rasterio.windows import Window

DATA_DIR = os.environ.get("CS12_DATA", "cloudsen12_fast")  # SSD: 17x piu' veloce di D: (HDD)
OUT_DIR = "unet_revision"
NODATA = 99
NUM_CLASSES = 4
CROP = 256
BATCH = 8
EPOCHS = 12          # con migliaia di patch per epoca 30 sono eccessive
LR = 1e-4
CLASSES = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']
NWORKERS = 4   # su Windows >4 worker con GDAL porta a deadlock dei DataLoader
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------- dati
def load_index():
    idx = pd.read_csv(os.path.join(DATA_DIR, 'index.csv'))
    idx = idx[idx.status.isin(['ok', 'cached'])].copy()
    idx['s2'] = [os.path.join(DATA_DIR, s, 's2', i + '.tif') for s, i in zip(idx.split, idx.id)]
    idx['label'] = [os.path.join(DATA_DIR, s, 'label', i + '.tif') for s, i in zip(idx.split, idx.id)]
    return idx


class PatchDataset(Dataset):
    """weak=False: maschera esperta dal file. weak=True: etichetta costante da ERA5."""

    def __init__(self, df, weak=False, threshold=50.0, train=True, bands=None):
        self.rows = df.to_dict('records')
        self.weak = weak
        self.thr = threshold
        self.train = train
        self.bands = bands or list(range(1, 14))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        r = self.rows[k]
        # in addestramento si legge SOLO la finestra che serve: i file sono scritti a
        # tile di 256, quindi la lettura a finestra evita di decomprimere l'intera patch.
        win = None
        if self.train:
            i = np.random.randint(0, 512 - CROP + 1)
            j = np.random.randint(0, 512 - CROP + 1)
            win = Window(j, i, CROP, CROP)
        with rio.open(r['s2']) as src:
            img = src.read(self.bands, window=win).astype(np.float32)
        img /= 10000.0
        np.clip(img, 0, 1, out=img)
        if self.weak:
            cls = 1 if float(r['era5_tcc_pct']) >= self.thr else 0
            lab = np.full(img.shape[1:], cls, dtype=np.int64)
        else:
            with rio.open(r['label']) as src:
                lab = src.read(1, window=win).astype(np.int64)
            lab[(lab > 3) & (lab != NODATA)] = NODATA      # valori 4,5,6: non documentati
        return torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(lab))


def build_unet(in_ch):
    import segmentation_models_pytorch as smp
    return smp.Unet('resnet34', encoder_weights=None, in_channels=in_ch, classes=NUM_CLASSES)


# ----------------------------------------------------------------- metriche
def metrics_from_counts(cm):
    """cm: matrice 4x4 di conteggi (righe = vero, colonne = predetto), senza i no-data."""
    out = {}
    tot = cm.sum()
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        out[CLASSES[c]] = {
            'IoU': float(tp / max(tp + fp + fn, 1)),
            'F1': float(2 * tp / max(2 * tp + fp + fn, 1)),
            'support': int(cm[c, :].sum()),
        }
    out['Overall'] = {
        'Accuracy': float(np.trace(cm) / max(tot, 1)),
        'Mean IoU': float(np.mean([out[c]['IoU'] for c in CLASSES])),
    }
    # binario: nuvola = thick + thin (le ombre non sono nuvola; ERA5 non le modella)
    cloud = np.array([0, 1, 1, 0])
    b = np.zeros((2, 2))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            b[cloud[i], cloud[j]] += cm[i, j]
    tp, fp, fn = b[1, 1], b[0, 1], b[1, 0]
    out['Binary'] = {
        'IoU cloud': float(tp / max(tp + fp + fn, 1)),
        'IoU clear': float(b[0, 0] / max(b[0, 0] + fn + fp, 1)),
        'Accuracy': float((b[0, 0] + b[1, 1]) / max(b.sum(), 1)),
    }
    return out


@torch.no_grad()
def evaluate(model, loader, scene_thr=50.0):
    """Valutazione a piena risoluzione, escludendo i pixel non annotati."""
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    scene_true, scene_pred = [], []
    for img, lab in loader:
        img = img.to(DEVICE)
        pred = model(img).argmax(1).cpu().numpy()
        lab = lab.numpy()
        valid = lab != NODATA
        for p, l, v in zip(pred, lab, valid):
            if not v.any():
                continue
            np.add.at(cm, (l[v], p[v]), 1)
            scene_true.append(100 * np.isin(l[v], (1, 2)).mean())
            scene_pred.append(100 * np.isin(p[v], (1, 2)).mean())
    m = metrics_from_counts(cm)
    st, sp = np.array(scene_true), np.array(scene_pred)
    if len(st):
        yt, yp = st >= scene_thr, sp >= scene_thr
        m['Scene'] = {
            'Accuracy': float((yt == yp).mean()),
            'Balanced accuracy': float(0.5 * ((yp[yt]).mean() if yt.any() else 0)
                                       + 0.5 * ((~yp[~yt]).mean() if (~yt).any() else 0)),
            'MAE cloud fraction': float(np.abs(st - sp).mean()),
            'n scenes': int(len(st)),
        }
        try:
            from sklearn.metrics import roc_auc_score
            if yt.any() and not yt.all():
                m['Scene']['AUC'] = float(roc_auc_score(yt, sp))
        except Exception:
            pass
    m['_cm'] = cm.tolist()
    return m


# ----------------------------------------------------------------- training
def train(model, tr_loader, va_loader, epochs, tag):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(ignore_index=NODATA)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)
    best, hist = float('inf'), []
    ckpt = os.path.join(OUT_DIR, tag + '.pth')
    for ep in range(epochs):
        model.train()
        tl, n = 0.0, 0
        t0 = time.time()
        for img, lab in tr_loader:
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(img), lab)
            loss.backward()
            opt.step()
            tl += loss.item(); n += 1
        model.eval()
        vl, m = 0.0, 0
        with torch.no_grad():
            for img, lab in va_loader:
                img, lab = img.to(DEVICE), lab.to(DEVICE)
                vl += crit(model(img), lab).item(); m += 1
        tl, vl = tl / max(n, 1), vl / max(m, 1)
        sched.step(vl)
        hist.append({'epoch': ep + 1, 'train_loss': tl, 'val_loss': vl})
        if vl < best:
            best = vl
            torch.save(model.state_dict(), ckpt)
        if (ep + 1) % 5 == 0 or ep == 0:
            print("    ep %2d/%d  train %.4f  val %.4f  (%.0fs)" % (ep + 1, epochs, tl, vl, time.time() - t0), flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model, hist


# ----------------------------------------------------------------- baseline
def run_baselines(idx, args):
    te = idx[idx.split == 'test']
    loader = DataLoader(PatchDataset(te, train=False), batch_size=4, num_workers=NWORKERS)
    res = {}
    cm_major = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_era5 = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    rows = te.to_dict('records')
    for k, (_, lab) in enumerate(loader):
        lab = lab.numpy()
        for b in range(lab.shape[0]):
            l = lab[b]
            v = l != NODATA
            if not v.any():
                continue
            np.add.at(cm_major, (l[v], np.zeros(v.sum(), dtype=int)), 1)   # tutto clear
            r = rows[k * 4 + b]
            cls = 1 if float(r['era5_tcc_pct'] or 0) >= 50 else 0          # ERA5 come maschera
            np.add.at(cm_era5, (l[v], np.full(v.sum(), cls, dtype=int)), 1)
    res['majority_clear'] = metrics_from_counts(cm_major)
    res['era5_as_mask'] = metrics_from_counts(cm_era5)
    with open(os.path.join(OUT_DIR, 'baselines.json'), 'w') as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        print("%-16s acc %.3f  mIoU %.3f  binary IoU cloud %.3f  binary acc %.3f"
              % (k, v['Overall']['Accuracy'], v['Overall']['Mean IoU'],
                 v['Binary']['IoU cloud'], v['Binary']['Accuracy']))
    return res


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['baselines', 'expert', 'weak', 'finetune'])
    ap.add_argument('--n-train', type=int, default=0, help='0 = tutte le patch disponibili')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--threshold', type=float, default=50.0)
    ap.add_argument('--pretrained', default=None)
    ap.add_argument('--bands', type=int, default=13, choices=[4, 13])
    ap.add_argument('--tag', default='')
    ap.add_argument('--data-dir', default=None)
    args = ap.parse_args()

    global DATA_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir
    idx = load_index()
    print("patch valide:", idx.groupby('split').size().to_dict())
    bands = [2, 3, 4, 8] if args.bands == 4 else list(range(1, 14))

    if args.mode == 'baselines':
        run_baselines(idx, args)
        return

    te = idx[idx.split == 'test']
    va = idx[idx.split == 'val']
    pool = idx[idx.split == ('train_weak' if args.mode == 'weak' else 'train_expert')]
    if args.mode == 'weak':
        pool = pool[pool.era5_tcc_pct.notna()]
    weak = args.mode == 'weak'

    test_loader = DataLoader(PatchDataset(te, train=False, bands=bands), batch_size=4, num_workers=NWORKERS)
    val_loader = DataLoader(PatchDataset(va, weak=weak, threshold=args.threshold,
                                         train=True, bands=bands), batch_size=BATCH, num_workers=NWORKERS)

    runs = []
    out_path = os.path.join(OUT_DIR, 'runs_%s%s.json' % (args.mode, args.tag))
    for seed in args.seeds:
        n = args.n_train or len(pool)
        n = min(n, len(pool))
        sub = pool.sample(n, random_state=100 + seed) if n < len(pool) else pool
        tag = "%s%s_n%d_b%d_s%d" % (args.mode, args.tag, n, args.bands, seed)
        print("\n--- %s  (%d patch di addestramento)" % (tag, n), flush=True)
        torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
        tr_loader = DataLoader(PatchDataset(sub, weak=weak, threshold=args.threshold,
                                            train=True, bands=bands),
                               batch_size=BATCH, shuffle=True, num_workers=NWORKERS, drop_last=True)
        model = build_unet(len(bands))
        if args.pretrained:
            model.load_state_dict(torch.load(args.pretrained, map_location=DEVICE), strict=False)
        t0 = time.time()
        model, hist = train(model, tr_loader, val_loader, args.epochs, tag)
        m = evaluate(model, test_loader)
        m.pop('_cm', None)
        print("    test: acc %.3f  mIoU %.3f | binary IoU cloud %.3f acc %.3f | scene acc %.3f"
              % (m['Overall']['Accuracy'], m['Overall']['Mean IoU'], m['Binary']['IoU cloud'],
                 m['Binary']['Accuracy'], m.get('Scene', {}).get('Accuracy', float('nan'))), flush=True)
        runs.append({'tag': tag, 'mode': args.mode, 'n_train': n, 'bands': args.bands,
                     'seed': seed, 'threshold': args.threshold, 'minutes': (time.time() - t0) / 60,
                     'results': m, 'history': hist})
        with open(out_path, 'w') as f:
            json.dump(runs, f, indent=1)
    print("\nsalvato", out_path)


if __name__ == '__main__':
    main()
