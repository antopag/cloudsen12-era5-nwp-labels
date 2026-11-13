"""
Script 20: download su larga scala di CloudSEN12+ per la revisione (richiesta editor/revisori).

Risolve tre problemi della versione precedente:
  1. SPLIT UFFICIALE del dataset (tortilla:data_split), disgiunto per roi_id -> niente leakage.
  2. VALIDAZIONE delle maschere: le patch 'high' con maschera degenere (tutta-zero o tutta no-data)
     vengono marcate e NON usate. Nel download precedente 77 patch su 306 erano in questo stato.
  3. Il valore 99 = NO DATA viene PRESERVATO nel file salvato. Non va mai clippato a 3 (shadow),
     come faceva il codice precedente; va ignorato in training (ignore_index) e in valutazione.

Classi CloudSEN12+: 0=clear, 1=thick cloud, 2=thin cloud, 3=cloud shadow, 99=no data.

Output in OUT_DIR:
    index.csv                 un record per patch: split, roi, data, label_type, ERA5 TCC,
                              percentuali per classe misurate sulla maschera, stato di validita'
    <split>/s2/<id>.tif       13 bande L1C uint16, DEFLATE
    <split>/label/<id>.tif    maschera uint8 con 99 preservato (solo dove serve)

Uso:
    python 20_download_scaleup.py --what test val train_expert   # necessari
    python 20_download_scaleup.py --what train_weak --limit 20000
    python 20_download_scaleup.py --what all --limit 20000
Il download e' riprendibile: le patch gia' presenti vengono saltate.
"""
import os
import sys
import argparse
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.io import MemoryFile

OUT_DIR = "cloudsen12_scaleup"
ERA5_CSV = os.path.join("era5_data", "era5_vs_cloudsen12.csv")
NODATA = 99
WORKERS = 8

_tls = threading.local()
_lock = threading.Lock()


def get_ds():
    """Un handle tacoreader per thread (la libreria non e' thread-safe)."""
    if not hasattr(_tls, 'ds'):
        import tacoreader.v1 as tacoreader
        _tls.ds = tacoreader.load("tacofoundation:cloudsen12-l1c")
    return _tls.ds


def build_index(limit_weak):
    """Seleziona le patch da scaricare e le annota con il valore ERA5."""
    import tacoreader.v1 as tacoreader
    df = pd.DataFrame(tacoreader.load("tacofoundation:cloudsen12-l1c"))
    # ATTENZIONE: ds.read(k) interpreta k come POSIZIONE nel dataset, non come
    # etichetta dell'indice del DataFrame, che non e' ne' contiguo ne' ordinato.
    # Usare l'etichetta fa leggere una patch diversa da quella attesa.
    df = df.reset_index(drop=True)
    df['row'] = np.arange(len(df))
    df['date'] = pd.to_datetime(df['stac:time_start'], unit='s').dt.strftime('%Y%m%d')

    era = pd.read_csv(ERA5_CSV)
    era['date'] = era.s2_date.str[:10].str.replace('-', '')
    era = era.rename(columns={'roi_id': 'old_roi_id'})
    keep = ['old_roi_id', 'date', 'era5_tcc_pct', 's2_date',
            'cloudsen_thick_cloud_pct', 'cloudsen_thin_cloud_pct',
            'cloudsen_shadow_pct', 'cloudsen_clear_pct', 'lon', 'lat']
    df = df.merge(era[keep].drop_duplicates(['old_roi_id', 'date']),
                  on=['old_roi_id', 'date'], how='left')

    df['id'] = df['tortilla:id'].str.replace('__', '_', regex=False)
    sp, lt = df['tortilla:data_split'], df['label_type']
    groups = {
        'test':         df[(sp == 'test') & (lt == 'high')],
        'val':          df[(sp == 'validation') & (lt == 'high')],
        'train_expert': df[(sp == 'train') & (lt == 'high')],
        'train_weak':   df[(sp == 'train') & (lt != 'high') & df.era5_tcc_pct.notna()],
    }
    if limit_weak and len(groups['train_weak']) > limit_weak:
        groups['train_weak'] = groups['train_weak'].sample(limit_weak, random_state=42)
    for k, g in groups.items():
        g = g.copy()
        g['split'] = k
        groups[k] = g
    out = pd.concat(groups.values(), ignore_index=True)
    out = _attach_subfiles(out)
    return out


def _subfiles_chunk(rows):
    """Risolve i byte-range per un blocco di posizioni (eseguito in un processo)."""
    import tacoreader.v1 as _t
    d = _t.load("tacofoundation:cloudsen12-l1c")
    out = []
    for r in rows:
        try:
            a = d.read(int(r))['internal:subfile']
            out.append((a.iloc[0], a.iloc[1]))
        except Exception:
            out.append((None, None))
    return out


def _attach_subfiles(sel, workers=6):
    """Risolve i byte-range dei due asset SOLO per le patch selezionate, in parallelo."""
    rows = list(sel['row'].values)
    chunks = [rows[i::workers] for i in range(workers)]
    order = [i for c in range(workers) for i in range(c, len(rows), workers)]
    res = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(_subfiles_chunk, chunks):
            res.extend(part)
    pairs = [None] * len(rows)
    for pos, val in zip(order, res):
        pairs[pos] = val
    sel = sel.copy()
    sel['s2_subfile'] = [p[0] if p else None for p in pairs]
    sel['label_subfile'] = [p[1] if p else None for p in pairs]
    return sel


def save_tif(path, arr, dtype, nodata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if arr.ndim == 2:
        arr = arr[None]
    with rio.open(path, 'w', driver='GTiff', height=arr.shape[1], width=arr.shape[2],
                  count=arr.shape[0], dtype=dtype, compress='deflate', predictor=2,
                  tiled=True, blockxsize=256, blockysize=256,
                  **({'nodata': nodata} if nodata is not None else {})) as dst:
        dst.write(arr.astype(dtype))


# Le patch vengono lette via HTTP range request (/vsicurl). Senza questi parametri GDAL
# non ritenta le richieste incomplete e restituisce un GeoTIFF valido con byte errati,
# e la sua cache ripropone poi lo stesso contenuto corrotto a ogni tentativo.
GDAL_ENV = dict(GDAL_HTTP_MAX_RETRY='8', GDAL_HTTP_RETRY_DELAY='2',
                GDAL_HTTP_TIMEOUT='120', GDAL_HTTP_CONNECTTIMEOUT='30',
                CPL_VSIL_CURL_USE_HEAD='NO', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                CPL_VSIL_CURL_CACHE_SIZE='0', VSI_CACHE='FALSE')


def _clear_curl_cache():
    try:
        from osgeo import gdal
        gdal.VSICurlClearCache()
    except Exception:
        pass


def _parse_subfile(spec):
    """'/vsisubfile/<offset>_<length>,/vsicurl/<url>' -> (url, offset, length)."""
    body = spec.split('/vsisubfile/', 1)[1]
    rng, rest = body.split(',', 1)
    off, ln = rng.split('_')
    url = rest.split('/vsicurl/', 1)[1]
    return url, int(off), int(ln)


def _http_range(url, offset, length, session, timeout=120):
    """Scarica esattamente [offset, offset+length) e verifica la lunghezza ricevuta.
    tacoreader legge gli stessi byte attraverso thread interni che in questo ambiente
    corrompono i dati o fanno crashare l'interprete; qui la lettura e' deterministica."""
    h = {'Range': 'bytes=%d-%d' % (offset, offset + length - 1)}
    r = session.get(url, headers=h, timeout=timeout)
    r.raise_for_status()
    if len(r.content) != length:
        raise IOError('range parziale: %d byte invece di %d' % (len(r.content), length))
    return r.content


def _read_assets(rec, need_label, session):
    """Legge immagine (+ maschera) dai byte, senza passare da tacoreader."""
    from rasterio.io import MemoryFile
    url, off, ln = _parse_subfile(rec['s2_subfile'])
    with MemoryFile(_http_range(url, off, ln, session)) as mf, mf.open() as src:
        img = src.read()
    lab = None
    if need_label:
        url, off, ln = _parse_subfile(rec['label_subfile'])
        with MemoryFile(_http_range(url, off, ln, session)) as mf, mf.open() as src:
            lab = src.read(1)
    return img, lab


def _check(img, lab, rec, need_label):
    """Controlli di integrita'. Le letture via /vsicurl possono restituire byte errati
    senza sollevare eccezioni: un GeoTIFF valido con contenuto corrotto. Validiamo
    confrontando la maschera con le percentuali per classe dichiarate nei metadati."""
    if img.shape != (13, 512, 512):
        return 'bad_image_shape'
    if not np.any(img):
        return 'bad_image_empty'
    if not need_label:
        return None
    if lab is None or lab.shape != (512, 512):
        return 'bad_label_shape'
    # percentuali dichiarate dal DATASET stesso (thick_percentage/thin_percentage),
    # non quelle del CSV ERA5: il merge su (roi,data) non e' uno-a-uno.
    try:
        declared = float(rec['thick_percentage']) + float(rec['thin_percentage'])
    except (KeyError, TypeError, ValueError):
        declared = None
    measured = 100.0 * float(np.isin(lab, (1, 2)).mean())
    nodata = float((lab == NODATA).mean())
    if rec.get('label_type') == 'high' and nodata > 0.02:
        return 'label_nodata_high'
    # tolleranza larga: serve solo a intercettare la corruzione grossolana da lettura HTTP
    # parziale (maschera vuota o scorrelata), non piccole differenze di arrotondamento.
    if declared is not None and abs(declared - measured) > 8.0:
        return 'label_mismatch_metadata'
    return None


def fetch_one(rec, retries=4):
    """Scarica una patch, validando il contenuto e ritentando se corrotto."""
    import requests
    if not hasattr(_tls, 'session'):
        _tls.session = requests.Session()
    split, pid = rec['split'], rec['id']
    need_label = split != 'train_weak'
    s2_path = os.path.join(OUT_DIR, split, 's2', pid + '.tif')
    lab_path = os.path.join(OUT_DIR, split, 'label', pid + '.tif')
    out = {'id': pid, 'split': split, 'status': 'ok', 'attempts': 0}

    if os.path.exists(s2_path) and (not need_label or os.path.exists(lab_path)):
        out['status'] = 'cached'
        return out

    problem = 'not_attempted'
    for attempt in range(1, retries + 1):
        out['attempts'] = attempt
        try:
            img, lab = _read_assets(rec, need_label, _tls.session)
        except Exception as e:
            problem = 'error'
            out['detail'] = str(e)[:120]
            time.sleep(1.5 * attempt)
            continue
        problem = _check(img, lab, rec, need_label)
        if problem is None:
            if need_label:
                u = np.unique(lab)
                out['nodata_frac'] = float((lab == NODATA).mean())
                for k, v in (('clear', 0), ('thick', 1), ('thin', 2), ('shadow', 3)):
                    out['m_' + k] = 100 * float((lab == v).mean())
                out['extra_vals'] = ','.join(str(int(x)) for x in u if x not in (0, 1, 2, 3, NODATA))
                save_tif(lab_path, lab, 'uint8', nodata=NODATA)
            save_tif(s2_path, img, 'uint16')
            return out
        time.sleep(1.0 * attempt)
    out['status'] = problem
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--what', nargs='+', default=['all'],
                    choices=['all', 'test', 'val', 'train_expert', 'train_weak'])
    ap.add_argument('--limit', type=int, default=20000, help='max patch per train_weak')
    ap.add_argument('--workers', type=int, default=WORKERS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("indice dal dataset...")
    idx = build_index(args.limit)
    wanted = ['test', 'val', 'train_expert', 'train_weak'] if 'all' in args.what else args.what
    idx = idx[idx.split.isin(wanted)].reset_index(drop=True)
    print(idx.groupby('split').size().to_string())
    print("ERA5 disponibile per %d/%d patch" % (idx.era5_tcc_pct.notna().sum(), len(idx)))

    recs = idx.to_dict('records')
    done, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, r): r for r in recs}
        for n, f in enumerate(as_completed(futs), 1):
            done.append(f.result())
            if n % 100 == 0 or n == len(recs):
                el = time.time() - t0
                ok = sum(1 for d in done if d['status'] in ('ok', 'cached'))
                print("  %6d/%d  ok %d  %.0f patch/min  eta %.1f h"
                      % (n, len(recs), ok, 60 * n / el, (len(recs) - n) * el / n / 3600), flush=True)
                pd.DataFrame(done).to_csv(os.path.join(OUT_DIR, 'download_log.csv'), index=False)

    log = pd.DataFrame(done)
    log.to_csv(os.path.join(OUT_DIR, 'download_log.csv'), index=False)
    out = idx.merge(log, on=['id', 'split'], how='left')
    cols = ['id', 'split', 'roi_id', 'old_roi_id', 'date', 's2_date', 'label_type', 'lon', 'lat',
            'thick_percentage', 'thin_percentage', 'cloud_shadow_percentage', 'clear_percentage',
            'era5_tcc_pct', 'cloudsen_thick_cloud_pct', 'cloudsen_thin_cloud_pct',
            'cloudsen_shadow_pct', 'cloudsen_clear_pct', 'status', 'nodata_frac',
            'm_clear', 'm_thick', 'm_thin', 'm_shadow', 'extra_vals']
    out[[c for c in cols if c in out.columns]].to_csv(os.path.join(OUT_DIR, 'index.csv'), index=False)

    print("\nstato finale:")
    print(log.groupby(['split', 'status']).size().to_string())
    print("\nindice salvato in", os.path.join(OUT_DIR, 'index.csv'))


if __name__ == '__main__':
    main()
