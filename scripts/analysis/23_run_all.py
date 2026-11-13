"""
Script 23: esegue in sequenza tutti gli esperimenti della revisione.

Ordine pensato perche' i risultati piu' importanti arrivino per primi: se il programma
viene interrotto, i blocchi gia' conclusi restano validi e i loro JSON sono completi.

Uso:  python 23_run_all.py            (tutto)
      python 23_run_all.py --from 3   (riprende dal blocco 3)
      python 23_run_all.py --list     (elenca i blocchi senza eseguirli)
"""
import os
import sys
import time
import argparse
import subprocess

PY = sys.executable
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '21_unet_revision.py')
DATA = os.environ.get("CS12_DATA", "cloudsen12_fast")
OUT = "unet_revision"
WEIGHTS = os.path.join(OUT, 'weak_scale_n20000_b13_s0.pth')   # per il fine-tuning

BLOCKS = [
    # (nome, argomenti)
    ("1. modello esperto, dati completi (5 seed)",
     ['--mode', 'expert', '--n-train', '8490', '--seeds', '0', '1', '2', '3', '4', '--tag', '_full']),

    ("2. modello debole su 20.000 patch (3 seed)",
     ['--mode', 'weak', '--n-train', '20000', '--seeds', '0', '1', '2', '--tag', '_scale']),

    ("3a. scala etichette deboli: 200", ['--mode', 'weak', '--n-train', '200', '--seeds', '0', '1', '2', '--tag', '_n200']),
    ("3b. scala etichette deboli: 1000", ['--mode', 'weak', '--n-train', '1000', '--seeds', '0', '1', '2', '--tag', '_n1000']),
    ("3c. scala etichette deboli: 5000", ['--mode', 'weak', '--n-train', '5000', '--seeds', '0', '1', '2', '--tag', '_n5000']),

    ("4a. budget esperto da zero: 849", ['--mode', 'expert', '--n-train', '849', '--seeds', '0', '1', '2', '--tag', '_b849']),
    ("4b. budget esperto da zero: 2122", ['--mode', 'expert', '--n-train', '2122', '--seeds', '0', '1', '2', '--tag', '_b2122']),
    ("4c. budget esperto da zero: 4245", ['--mode', 'expert', '--n-train', '4245', '--seeds', '0', '1', '2', '--tag', '_b4245']),

    ("5a. fine-tuning da ERA5: 849", ['--mode', 'expert', '--n-train', '849', '--seeds', '0', '1', '2',
                                      '--pretrained', WEIGHTS, '--tag', '_ft849']),
    ("5b. fine-tuning da ERA5: 2122", ['--mode', 'expert', '--n-train', '2122', '--seeds', '0', '1', '2',
                                       '--pretrained', WEIGHTS, '--tag', '_ft2122']),
    ("5c. fine-tuning da ERA5: 4245", ['--mode', 'expert', '--n-train', '4245', '--seeds', '0', '1', '2',
                                       '--pretrained', WEIGHTS, '--tag', '_ft4245']),
    ("5d. fine-tuning da ERA5: 8490", ['--mode', 'expert', '--n-train', '8490', '--seeds', '0', '1', '2',
                                       '--pretrained', WEIGHTS, '--tag', '_ft8490']),

    ("6a. soglia 30%", ['--mode', 'weak', '--n-train', '5000', '--threshold', '30', '--seeds', '0', '1', '2', '--tag', '_t30']),
    ("6b. soglia 40%", ['--mode', 'weak', '--n-train', '5000', '--threshold', '40', '--seeds', '0', '1', '2', '--tag', '_t40']),
    ("6c. soglia 60%", ['--mode', 'weak', '--n-train', '5000', '--threshold', '60', '--seeds', '0', '1', '2', '--tag', '_t60']),
    ("6d. soglia 70%", ['--mode', 'weak', '--n-train', '5000', '--threshold', '70', '--seeds', '0', '1', '2', '--tag', '_t70']),

    ("7a. ablazione 4 bande, esperto", ['--mode', 'expert', '--n-train', '8490', '--bands', '4',
                                        '--seeds', '0', '1', '2', '--tag', '_b4']),
    ("7b. ablazione 4 bande, debole", ['--mode', 'weak', '--n-train', '20000', '--bands', '4',
                                       '--seeds', '0', '1', '2', '--tag', '_b4']),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start', type=int, default=1)
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    if args.list:
        for i, (name, _) in enumerate(BLOCKS, 1):
            print("%2d. %s" % (i, name))
        return

    t_all = time.time()
    for i, (name, extra) in enumerate(BLOCKS, 1):
        if i < args.start:
            continue
        print("\n" + "=" * 78)
        print("[%d/%d] %s" % (i, len(BLOCKS), name), flush=True)
        print("=" * 78, flush=True)
        if '--pretrained' in extra:
            w = extra[extra.index('--pretrained') + 1]
            if not os.path.exists(w):
                print("   SALTATO: pesi di pretraining non trovati (%s)" % w, flush=True)
                continue
        cmd = [PY, SCRIPT, '--data-dir', DATA] + extra
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if any(k in line for k in ('test:', '--- ', 'salvato', 'Error', 'error')):
                print("   " + line.strip(), flush=True)
        if r.returncode != 0:
            print("   FALLITO (codice %d): %s" % (r.returncode, r.stderr.strip()[-300:]), flush=True)
        print("   durata %.1f min  |  totale trascorso %.1f h"
              % ((time.time() - t0) / 60, (time.time() - t_all) / 3600), flush=True)

    print("\nprogramma completato in %.1f ore" % ((time.time() - t_all) / 3600))


if __name__ == '__main__':
    main()
