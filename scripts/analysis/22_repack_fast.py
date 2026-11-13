"""
Script 22: riscrive le patch scaricate in un formato a lettura rapida.

I file prodotti dal download sono GeoTIFF compressi DEFLATE: ottimi per lo spazio, ma la
decompressione domina il tempo di addestramento (11.8 min/epoca su 8,490 patch, di cui
oltre il 90% in lettura). Qui si riscrivono come GeoTIFF NON compressi e tiled a 256,
cosi' la lettura di una finestra e' un seek diretto.

Costo in spazio: ~6.8 MB per patch (13 bande uint16) invece di ~1.4 MB.
Uso:
    python 22_repack_fast.py --splits test val train_expert
    python 22_repack_fast.py --splits train_weak
"""
import os
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio as rio

SRC = "cloudsen12_scaleup"
DST = "cloudsen12_fast"


def repack_one(args):
    split, pid, need_label = args
    s_s2 = os.path.join(SRC, split, 's2', pid + '.tif')
    d_s2 = os.path.join(DST, split, 's2', pid + '.tif')
    try:
        if not os.path.exists(d_s2):
            with rio.open(s_s2) as src:
                img = src.read()
            with rio.open(d_s2, 'w', driver='GTiff', height=img.shape[1], width=img.shape[2],
                          count=img.shape[0], dtype='uint16', compress=None,
                          tiled=True, blockxsize=256, blockysize=256) as dst:
                dst.write(img)
        if need_label:
            s_lb = os.path.join(SRC, split, 'label', pid + '.tif')
            d_lb = os.path.join(DST, split, 'label', pid + '.tif')
            if not os.path.exists(d_lb):
                with rio.open(s_lb) as src:
                    lab = src.read(1)
                with rio.open(d_lb, 'w', driver='GTiff', height=lab.shape[0], width=lab.shape[1],
                              count=1, dtype='uint8', compress=None, nodata=99,
                              tiled=True, blockxsize=256, blockysize=256) as dst:
                    dst.write(lab[None])
        return 1
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['test', 'val', 'train_expert'])
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    idx = pd.read_csv(os.path.join(SRC, 'index.csv'))
    idx = idx[idx.split.isin(args.splits)]
    for s in args.splits:
        os.makedirs(os.path.join(DST, s, 's2'), exist_ok=True)
        os.makedirs(os.path.join(DST, s, 'label'), exist_ok=True)
    jobs = [(r['split'], r['id'], r['split'] != 'train_weak') for r in idx.to_dict('records')]
    print("da riscrivere: %d patch" % len(jobs))

    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for ok in ex.map(repack_one, jobs, chunksize=20):
            done += ok
            if done % 1000 == 0:
                el = time.time() - t0
                print("  %6d/%d  %.0f patch/min  eta %.1f min"
                      % (done, len(jobs), 60 * done / el, (len(jobs) - done) * el / done / 60), flush=True)
    # l'indice del formato veloce e' quello COMPLETO della sorgente: scrivere solo le
    # righe di questa esecuzione cancellerebbe i riferimenti agli split gia' convertiti.
    pd.read_csv(os.path.join(SRC, 'index.csv')).to_csv(os.path.join(DST, 'index.csv'), index=False)
    print("fatto: %d/%d in %.1f min" % (done, len(jobs), (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
