# ERA5 and GFS Cloud Cover vs. CloudSEN12+

Code accompanying the manuscript:

> A. Anzalone and A. Pagliaro, *ERA5 and GFS Cloud Cover vs. CloudSEN12+: Can NWP Replace Expert
> Labels?*, submitted to **Applied Sciences** (MDPI), 2026.

The study asks whether freely available cloud-cover fields from numerical weather prediction (NWP)
products can replace expensive expert pixel-level annotations when training cloud detectors for
Sentinel-2. It has two parts: a collocation of ERA5 and GFS total cloud cover with 48,380
expert-annotated CloudSEN12+ patches, and a set of controlled U-Net experiments in which only the
source of the training labels varies.

## Requirements

```bash
conda create -n cloudsen12 python=3.9
conda activate cloudsen12
pip install -r requirements.txt
```

A CUDA GPU is needed to retrain the U-Net models in reasonable time; the experiments in the paper
were run on a single NVIDIA RTX 4090. The collocation and statistics run on CPU.

Two external accounts are required to re-download the inputs:

- **Copernicus Climate Data Store** for ERA5, with credentials in `~/.cdsapirc`
  (see the header of `scripts/dataset/08_download_era5.py`).
- No account is needed for GFS, which is read from the NOAA Open Data archive on AWS, or for
  CloudSEN12+, which is public.

## Data layout

The scripts use paths relative to a working directory that holds the downloaded data. Run them from
that directory, not from the repository root:

```
<work_dir>/
├── CloudSEN12+_data/
│   └── downloaded_images/
│       ├── s2_images/      # <ROI>_<date>_<tile>.tif, 13 L1C bands
│       ├── labels/         # <ROI>_<date>_<tile>_label.tif, 4 classes
│       └── metadata/
├── era5_data/
│   ├── netcdf/             # era5_tcc_YYYYMMDD.nc
│   └── era5_vs_cloudsen12.csv
├── gfs_data/
│   └── gfs_vs_cloudsen12_full.csv
├── unet_results/           # model weights and JSON results
└── figures/                # output figures
```

## Pipeline

### 1. Data

| Script | Purpose |
| --- | --- |
| `scripts/dataset/01_explore_dataset.py` | Inspect the CloudSEN12+ catalogue |
| `scripts/dataset/02_download_samples.py` | Download the sample metadata table |
| `scripts/dataset/03_download_images.py` | Download the Sentinel-2 patches and expert masks |
| `scripts/dataset/05_download_by_year.py` | Same, restricted to one year |
| `scripts/dataset/07_analyze_locations.py` | Geographic breakdown of the sample |
| `scripts/dataset/08_download_era5.py` | Retrieve ERA5 total cloud cover from the CDS |
| `scripts/analysis/08_download_gfs.py` | Retrieve GFS TCDC from the NOAA AWS archive |

### 2. Collocation and statistics

| Script | Produces |
| --- | --- |
| `scripts/analysis/10_era5_comparison.py` | `era5_vs_cloudsen12.csv`; Table 3 (global statistics) |
| `scripts/analysis/09_gfs_comparison_full.py` | `gfs_vs_cloudsen12_full.csv`; Table 7 (GFS) |
| `scripts/analysis/18_bootstrap_ci.py` | 95% bootstrap confidence intervals for every stratum, Tables 3, 5, 6, 7 |
| `scripts/analysis/19_collocation_sensitivity.py` | Table 4: nearest-node vs bilinear vs 3x3 averaging |

### 3. Figures of the comparison

| Script | Produces |
| --- | --- |
| `scripts/analysis/11_generate_figures.py` | Figures 1, 4, 5 |
| `scripts/analysis/11b_refresh_fig2_fig3.py` | Figures 2 and 3 (shadow class excluded) |

### 4. U-Net experiments

All of them share the same split (seed 42, 70/15/15) and are evaluated on the same 47-patch
expert-labelled test set. The paper reports the 13-band configuration; `--bands 4` reproduces the
ablation.

| Script | Produces |
| --- | --- |
| `scripts/analysis/12_unet_experiment.py` | Models A and B, the original 4-band run |
| `scripts/analysis/12e_unet_13bands.py` | Models A and B with all 13 L1C bands; Table 8, Table 9, Figure 13 |
| `scripts/analysis/12b_unet_finetune.py` | ERA5 pretraining + expert fine-tuning at four annotation budgets; Table 11, Figure 11 |
| `scripts/analysis/12c_weak_label_variants.py` | Binarisation thresholds 30-70% and soft targets; Table 10, Figure 12 |
| `scripts/analysis/12d_refresh_figs.py` | Figures 9 and 10 |

Typical invocations for the results in the paper:

```bash
# Models A and B, 13 bands, 3 seeds
python scripts/analysis/12e_unet_13bands.py

# weak-label formulations, 13 bands
python scripts/analysis/12c_weak_label_variants.py --bands 13 --suffix b13

# pretraining and fine-tuning, 13 bands, starting from the Model B weights
python scripts/analysis/12b_unet_finetune.py --bands 13 --suffix b13 \
    --pretrained unet_results/b13_era5_s0_best.pth
```

Each script writes an incremental JSON file, so an interrupted run does not lose completed
trainings. Expect roughly 4-5 minutes per training on an RTX 4090 at 13 bands.

## Notes on reproducibility

- Results in the paper are means over three seeds (0, 1, 2); single-seed runs of the ERA5-only
  models vary by up to 0.1 in overall accuracy, which is why the seed spread is reported throughout.
- `scripts/analysis/19_collocation_sensitivity.py` is restricted to the patches whose acquisition
  hour matches the archived ERA5 field, because only one hour per date was stored locally.
- The ERA5 value of each patch is matched by region of interest **and** acquisition date. An earlier
  version matched by region of interest alone; that bug is described in the manuscript and is fixed
  in the code published here.

## Data sources

- **CloudSEN12+**: Aybar et al., *Data in Brief* 56 (2024) 110852, <https://cloudsen12.github.io>
- **ERA5**: Hersbach et al., *QJRMS* 146 (2020) 1999, via the Copernicus Climate Data Store
- **GFS**: NOAA NCEP, via the NOAA Open Data Dissemination archive on AWS

## Citation

```bibtex
@article{anzalone2026nwplabels,
  author  = {Anzalone, Anna and Pagliaro, Antonio},
  title   = {{ERA5} and {GFS} Cloud Cover vs. {CloudSEN12+}: Can {NWP} Replace Expert Labels?},
  journal = {Applied Sciences},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT, see `LICENSE`. The datasets keep the licences of their respective providers.
