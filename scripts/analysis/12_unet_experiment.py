"""
Script 12: U-Net cloud segmentation experiment
Confronto: Expert labels vs ERA5 weak labels
Carica immagini GeoTIFF da disco

Prerequisiti:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install segmentation-models-pytorch scikit-learn rasterio
"""

import os
import numpy as np
import pandas as pd
import json
import rasterio as rio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ==============================================================
# CONFIGURAZIONE
# ==============================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "unet_results"
FIGURES_DIR = "figures"
DATA_DIR = "DATA"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
NUM_CLASSES = 4  # 0=clear, 1=thick cloud, 2=thin cloud, 3=shadow
IMG_SIZE = 256
NUM_WORKERS = 0

# Input: B2, B3, B4, B8 (4 bande)
IN_CHANNELS = 4
BAND_INDICES = [1, 2, 3, 7]  # 0-based: B2, B3, B4, B8

print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# ==============================================================
# DATASET da GeoTIFF su disco
# ==============================================================

class CloudGeoTIFFDataset(Dataset):
    """Carica coppie S2 image + label da GeoTIFF locali."""

    def __init__(self, s2_paths, label_paths, img_size=256,
                 use_era5_labels=False, era5_dict=None, era5_threshold=50.0):
        self.s2_paths = s2_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.use_era5_labels = use_era5_labels
        self.era5_dict = era5_dict or {}
        self.era5_threshold = era5_threshold

    def __len__(self):
        return len(self.s2_paths)

    def __getitem__(self, idx):
        # Carica S2
        s2_path = self.s2_paths[idx]
        try:
            with rio.open(s2_path) as src:
                # Leggi bande selezionate (1-based in rasterio)
                bands = []
                for bi in BAND_INDICES:
                    band_num = bi + 1  # rasterio usa 1-based
                    if band_num <= src.count:
                        bands.append(src.read(band_num).astype(np.float32))
                    else:
                        bands.append(np.zeros((src.height, src.width), dtype=np.float32))
                img = np.stack(bands, axis=0)

                # Normalizza
                img = img / 10000.0
                img = np.clip(img, 0, 1)

        except Exception as e:
            img = np.zeros((IN_CHANNELS, 512, 512), dtype=np.float32)

        # Carica label
        if self.use_era5_labels:
            label = self._make_era5_label(idx, img.shape[1], img.shape[2])
        else:
            label = self._load_expert_label(idx)

        # Resize
        img_tensor = torch.from_numpy(img)
        label_tensor = torch.from_numpy(label.astype(np.int64))

        if img_tensor.shape[1] != self.img_size or img_tensor.shape[2] != self.img_size:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0), size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        if label_tensor.shape[0] != self.img_size or label_tensor.shape[1] != self.img_size:
            label_tensor = torch.nn.functional.interpolate(
                label_tensor.float().unsqueeze(0).unsqueeze(0),
                size=(self.img_size, self.img_size), mode='nearest'
            ).squeeze(0).squeeze(0).long()

        return img_tensor, label_tensor

    def _load_expert_label(self, idx):
        """Carica pixel-level label da GeoTIFF."""
        label_path = self.label_paths[idx]
        try:
            with rio.open(label_path) as src:
                label = src.read(1).astype(np.int64)
                label = np.clip(label, 0, NUM_CLASSES - 1)
                return label
        except:
            return np.zeros((512, 512), dtype=np.int64)

    def _make_era5_label(self, idx, h, w):
        """Crea weak label binaria da ERA5 TCC."""
        # Usa nome file per matchare con ERA5
        s2_name = os.path.basename(self.s2_paths[idx]).replace('.tif', '')

        if s2_name in self.era5_dict:
            tcc = self.era5_dict[s2_name]
            if tcc >= self.era5_threshold:
                return np.ones((h, w), dtype=np.int64)  # cloudy
            else:
                return np.zeros((h, w), dtype=np.int64)  # clear
        else:
            return np.zeros((h, w), dtype=np.int64)


# ==============================================================
# U-NET
# ==============================================================

def build_unet(in_channels=4, num_classes=4):
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )
        print(f"  U-Net ResNet34 ({in_channels}ch -> {num_classes}cls)")
        return model
    except ImportError:
        print("  smp non disponibile, uso U-Net custom")
        return SimpleUNet(in_channels, num_classes)


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=4, n_cls=4):
        super().__init__()
        def cb(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
                nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True))
        self.e1, self.e2, self.e3, self.e4 = cb(in_ch,64), cb(64,128), cb(128,256), cb(256,512)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d3, self.d2, self.d1 = cb(768,256), cb(384,128), cb(192,64)
        self.final = nn.Conv2d(64, n_cls, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.up(e4),e3],1))
        d2 = self.d2(torch.cat([self.up(d3),e2],1))
        d1 = self.d1(torch.cat([self.up(d2),e1],1))
        return self.final(d1)


# ==============================================================
# TRAINING
# ==============================================================

def train_model(model, train_loader, val_loader, num_epochs, model_name="model"):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss, n_batch = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batch += 1
        train_loss /= max(n_batch, 1)

        model.eval()
        val_loss, correct, total, n_val = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                n_val += 1
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        val_loss /= max(n_val, 1)
        val_acc = correct / max(total, 1)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{num_epochs} | "
                  f"TrLoss: {train_loss:.4f} | VaLoss: {val_loss:.4f} | VaAcc: {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{model_name}_best.pth'))

    return model, history


# ==============================================================
# EVALUATION
# ==============================================================

def evaluate_model(model, test_loader):
    model = model.to(DEVICE)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.append(preds.flatten())
            all_labels.append(labels.numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))

    class_names = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']
    results = {}

    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        iou = tp / max(tp + fp + fn, 1)
        results[class_names[c]] = {'Precision': precision, 'Recall': recall, 'F1': f1, 'IoU': iou}

    oa = np.diag(cm).sum() / max(cm.sum(), 1)
    mean_iou = np.mean([r['IoU'] for r in results.values()])
    results['Overall'] = {'Accuracy': oa, 'Mean IoU': mean_iou}

    return results, cm


def print_results(results, name):
    print(f"\n--- {name} ---")
    class_names = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']
    print(f"  {'Class':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} {'IoU':>8}")
    print(f"  {'-'*47}")
    for c in class_names:
        r = results[c]
        print(f"  {c:<15} {r['Precision']:>8.3f} {r['Recall']:>8.3f} {r['F1']:>8.3f} {r['IoU']:>8.3f}")
    print(f"  {'-'*47}")
    print(f"  {'Overall Acc':<15} {results['Overall']['Accuracy']:>8.3f}")
    print(f"  {'Mean IoU':<15} {results['Overall']['Mean IoU']:>8.3f}")


# ==============================================================
# FIGURES
# ==============================================================

def plot_training_curves(hist_a, hist_b):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(hist_a['train_loss'], 'b-', alpha=0.5, label='Expert (train)')
    ax.plot(hist_a['val_loss'], 'b-', lw=2, label='Expert (val)')
    ax.plot(hist_b['train_loss'], 'r-', alpha=0.5, label='ERA5 (train)')
    ax.plot(hist_b['val_loss'], 'r-', lw=2, label='ERA5 (val)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training Loss'); ax.legend()

    ax = axes[1]
    ax.plot(hist_a['val_acc'], 'b-', lw=2, label='Expert')
    ax.plot(hist_b['val_acc'], 'r-', lw=2, label='ERA5 weak')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('Validation Accuracy'); ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig9_unet_training.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig9_unet_training.pdf'), bbox_inches='tight')
    plt.close()
    print("  Salvato: fig9_unet_training")


def plot_comparison_bars(res_a, res_b):
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']
    x = np.arange(len(classes))
    w = 0.35
    iou_a = [res_a[c]['IoU'] for c in classes]
    iou_b = [res_b[c]['IoU'] for c in classes]
    b1 = ax.bar(x - w/2, iou_a, w, label='Expert labels', color='steelblue')
    b2 = ax.bar(x + w/2, iou_b, w, label='ERA5 weak labels', color='indianred')
    ax.set_ylabel('IoU'); ax.set_title('U-Net: Expert vs ERA5 Weak Labels')
    ax.set_xticks(x); ax.set_xticklabels(classes); ax.legend(); ax.set_ylim(0, 1)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig10_unet_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig10_unet_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("  Salvato: fig10_unet_comparison")


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("ESPERIMENTO U-NET: EXPERT vs ERA5 WEAK LABELS")
    print("=" * 60)

    # 1. Trova tutte le coppie S2 + label su disco
    print("\nRicerca immagini su disco...")

    s2_dir = os.path.join(DATA_DIR, "downloaded_images", "s2_images")
    label_dir = os.path.join(DATA_DIR, "downloaded_images", "labels")

    if not os.path.exists(s2_dir):
        print(f"  ERRORE: {s2_dir} non trovata")
        return

    s2_files = sorted([f for f in os.listdir(s2_dir) if f.endswith('.tif')])
    print(f"  Immagini S2: {len(s2_files)}")

    # Trova coppie con label
    pairs = []
    for s2_f in s2_files:
        name = s2_f.replace('.tif', '')
        label_f = name + '_label.tif'
        label_path = os.path.join(label_dir, label_f)
        if os.path.exists(label_path):
            pairs.append({
                'name': name,
                's2_path': os.path.join(s2_dir, s2_f),
                'label_path': label_path,
            })

    print(f"  Coppie S2+label: {len(pairs)}")

    if len(pairs) < 30:
        print("  Troppo poche immagini per l'esperimento!")
        return

    # 2. Carica ERA5 per weak labels
    print("\nCaricamento dati ERA5...")
    era5_dict = {}  # name -> tcc_pct

    try:
        df_era5 = pd.read_csv("era5_data/era5_vs_cloudsen12.csv")
        # Match per roi_id
        for _, row in df_era5.iterrows():
            roi_id = row['roi_id']
            # Cerca match nei nomi file
            for p in pairs:
                if roi_id in p['name']:
                    era5_dict[p['name']] = row['era5_tcc_pct']
        print(f"  ERA5 matchati: {len(era5_dict)}/{len(pairs)}")
    except Exception as e:
        print(f"  ERA5 non disponibile: {e}")
        print("  Uso cloud percentages dai metadati...")

    # Fallback: usa metadati JSON per ERA5-like weak labels
    if len(era5_dict) < len(pairs) // 2:
        print("  Caricamento metadati JSON come fallback...")
        meta_dir = os.path.join(DATA_DIR, "downloaded_images", "metadata")
        for p in pairs:
            if p['name'] not in era5_dict and os.path.exists(meta_dir):
                meta_path = os.path.join(meta_dir, p['name'] + '.json')
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    # Usa thick + thin come proxy
                    cloud_pct = meta.get('thick_cloud_pct', 0) + meta.get('thin_cloud_pct', 0)
                    era5_dict[p['name']] = cloud_pct
        print(f"  Con fallback: {len(era5_dict)}/{len(pairs)}")

    # 3. Split train/val/test
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    n = len(pairs)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_pairs = [pairs[i] for i in indices[:n_train]]
    val_pairs = [pairs[i] for i in indices[n_train:n_train + n_val]]
    test_pairs = [pairs[i] for i in indices[n_train + n_val:]]

    print(f"\n  Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    # Verifica label non vuote
    print("\n  Verifica label...")
    with rio.open(train_pairs[0]['label_path']) as src:
        sample_label = src.read(1)
        unique_vals = np.unique(sample_label)
        print(f"  Label valori unici (primo sample): {unique_vals}")

    # 4. Crea DataLoader
    def make_loader(pair_list, era5_labels=False, shuffle=False):
        s2_paths = [p['s2_path'] for p in pair_list]
        label_paths = [p['label_path'] for p in pair_list]
        ds = CloudGeoTIFFDataset(s2_paths, label_paths, IMG_SIZE,
                                  use_era5_labels=era5_labels,
                                  era5_dict=era5_dict)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS)

    # ==============================================================
    # MODEL A: Expert labels
    # ==============================================================
    print("\n" + "=" * 60)
    print("MODEL A: EXPERT LABELS")
    print("=" * 60)

    train_loader_a = make_loader(train_pairs, era5_labels=False, shuffle=True)
    val_loader_a = make_loader(val_pairs, era5_labels=False)
    test_loader = make_loader(test_pairs, era5_labels=False)

    model_a = build_unet(IN_CHANNELS, NUM_CLASSES)
    model_a, hist_a = train_model(model_a, train_loader_a, val_loader_a, NUM_EPOCHS, "expert")
    res_a, cm_a = evaluate_model(model_a, test_loader)
    print_results(res_a, "MODEL A: Expert Labels")

    # ==============================================================
    # MODEL B: ERA5 weak labels (train), expert labels (test)
    # ==============================================================
    print("\n" + "=" * 60)
    print("MODEL B: ERA5 WEAK LABELS")
    print("=" * 60)

    train_loader_b = make_loader(train_pairs, era5_labels=True, shuffle=True)
    val_loader_b = make_loader(val_pairs, era5_labels=True)
    # Test sempre con expert labels!

    model_b = build_unet(IN_CHANNELS, NUM_CLASSES)
    model_b, hist_b = train_model(model_b, train_loader_b, val_loader_b, NUM_EPOCHS, "era5weak")
    res_b, cm_b = evaluate_model(model_b, test_loader)
    print_results(res_b, "MODEL B: ERA5 Weak Labels")

    # ==============================================================
    # CONFRONTO
    # ==============================================================
    print("\n" + "=" * 60)
    print("CONFRONTO FINALE")
    print("=" * 60)

    classes = ['Clear', 'Thick cloud', 'Thin cloud', 'Shadow']
    print(f"\n  {'Class':<15} {'Expert IoU':>12} {'ERA5 IoU':>12} {'Delta':>10}")
    print(f"  {'-'*50}")
    for c in classes:
        ia, ib = res_a[c]['IoU'], res_b[c]['IoU']
        print(f"  {c:<15} {ia:>12.3f} {ib:>12.3f} {ib-ia:>+10.3f}")
    print(f"  {'-'*50}")
    oa_a, oa_b = res_a['Overall']['Accuracy'], res_b['Overall']['Accuracy']
    mi_a, mi_b = res_a['Overall']['Mean IoU'], res_b['Overall']['Mean IoU']
    print(f"  {'Overall Acc':<15} {oa_a:>12.3f} {oa_b:>12.3f} {oa_b-oa_a:>+10.3f}")
    print(f"  {'Mean IoU':<15} {mi_a:>12.3f} {mi_b:>12.3f} {mi_b-mi_a:>+10.3f}")

    # Figure
    print("\nGenerazione figure...")
    plot_training_curves(hist_a, hist_b)
    plot_comparison_bars(res_a, res_b)

    # Salva risultati JSON
    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(os.path.join(OUTPUT_DIR, 'unet_results.json'), 'w') as f:
        json.dump({'expert': res_a, 'era5_weak': res_b,
                   'history_expert': hist_a, 'history_era5': hist_b},
                  f, indent=2, default=convert)

    print(f"\nRisultati in: {OUTPUT_DIR}/unet_results.json")
    print(f"Figure in: {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
