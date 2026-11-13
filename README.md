# Can Scene-Level NWP Cloud Cover Replace Expert Pixel Labels?

Code and results accompanying the manuscript:

> A. Anzalone and A. Pagliaro, *Can Scene-Level NWP Cloud Cover Replace Expert Pixel Labels? A
> Large-Scale Assessment with ERA5, GFS and CloudSEN12+*, submitted to **Applied Sciences** (MDPI),
> 2026.

The study asks whether freely available cloud-cover fields from numerical weather prediction (NWP)
products can replace expensive expert pixel-level annotations when training cloud detectors for
Sentinel-2. It has two parts: a collocation of ERA5 and GFS total cloud cover with 48,380
expert-annotated CloudSEN12+ patches, and a set of controlled U-Net experiments in which only the
source of the training labels varies.

Version 2.0.0 corresponds to the revised manuscript. Compared with version 1.0.0 it adds the
region-disjoint partition and the rebuilt U-Net experiments (`scripts/analysis/20`-`24`), the grouped
bootstrap and the measurement of the ERA5 temporal offset (`25`, `29`), the two new figures (`26`,
`27`), and the `results/` directory with the collocation tables and the per-seed values of all 56
training runs.

## Requirements

```bash
conda create -n cloudsen12 python=3.10
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

## Results released with the code

Everything needed to check the numbers in the paper without downloading data or training a model.

| File | Content |
| --- | --- |
| `results/collocation/era5_vs_cloudsen12.csv` | The 48,380 ERA5 collocations: ROI, acquisition time, position, annotation quality, ERA5 total cloud cover, CloudSEN12+ class percentages |
| `results/collocation/gfs_vs_cloudsen12_full.csv` | The 81 GFS collocations |
| `results/collocation/temporal_offset.csv` | Offset in minutes between each acquisition and the ERA5 field used, for all 48,380 collocations |
| `results/collocation/collocation_stats_revision.csv` | Confidence intervals under the three bootstrap schemes (per sample, per 0.25° cell, per ROI) |
| `results/collocation/bootstrap_ci.csv` | Per-stratum bootstrap confidence intervals |
| `results/collocation/collocation_sensitivity_stats.csv` | Nearest-node vs bilinear vs 3x3 collocation |
| `results/unet_revision/all_runs.csv` | One row per training run (56 runs): configuration, seed and every metric |
| `results/unet_revision/runs_*.json` | The same runs with the full per-class output, as written by the training script |
| `results/unet_revision/baselines.json` | Untrained baselines: constant majority class, and ERA5 applied directly as a mask; also holds the per-class pixel counts of the test set |
| `results/unet_revision/tab*.csv` | Mean and standard deviation across seeds, as reported in Tables 10 and 12-15 |

Some file names and labels produced by the scripts are in Italian: `tab1_principale` = main
comparison, `tab2_scala` = scale, `tab3_budget` = budget, `tab4_soglia` = threshold, `tab5_bande` =
bands; `per campione` = per sample, `per cella` = per cell.

## Data layout

The scripts use paths relative to a working directory that holds the downloaded data. Run them from
that directory, not from the repository root:

```
<work_dir>/
├── era5_data/
│   ├── netcdf/             # era5_tcc_YYYYMMDD.nc
│   └── era5_vs_cloudsen12.csv
├── gfs_data/
│   └── gfs_vs_cloudsen12_full.csv
├── cloudsen12_scaleup/     # written by script 20: index.csv, <split>/s2, <split>/label
├── cloudsen12_fast/        # written by script 22: same patches, uncompressed, fast to read
├── unet_revision/          # model weights and JSON results of the U-Net experiments
└── figures/                # output figures
```

The directory read by the training script can be changed with the environment variable `CS12_DATA`
or with `--data-dir`. To regenerate the tables and figures from the released results without
training, copy `results/unet_revision/` to `<work_dir>/unet_revision/` and
`results/collocation/era5_vs_cloudsen12.csv` to `<work_dir>/era5_data/`.

## Pipeline

### 1. Data

| Script | Purpose |
| --- | --- |
| `scripts/dataset/01_explore_dataset.py` | Inspect the CloudSEN12+ catalogue |
| `scripts/dataset/02_download_samples.py` | Download the sample metadata table |
| `scripts/dataset/08_download_era5.py` | Retrieve ERA5 total cloud cover from the CDS |
| `scripts/analysis/08_download_gfs.py` | Retrieve GFS TCDC from the NOAA AWS archive |
| `scripts/analysis/20_download_scaleup.py` | Download the patches used by the U-Net experiments with the official CloudSEN12+ partition, verify every patch against the published metadata, write `index.csv` |
| `scripts/analysis/22_repack_fast.py` | Rewrite the downloaded patches as uncompressed tiled GeoTIFFs for fast training |

`scripts/dataset/03`, `05` and `07` are the smaller downloads and the geographic breakdown used in
the exploratory phase.

### 2. Collocation and statistics

| Script | Produces |
| --- | --- |
| `scripts/analysis/10_era5_comparison.py` | `era5_vs_cloudsen12.csv`; Table 3 (global statistics) |
| `scripts/analysis/09_gfs_comparison_full.py` | `gfs_vs_cloudsen12_full.csv`; Table 8 (GFS) |
| `scripts/analysis/18_bootstrap_ci.py` | Per-stratum statistics and confidence intervals, Tables 3 and 5-8 |
| `scripts/analysis/25_collocation_stats.py` | Table 1: per-sample bootstrap vs bootstrap grouped by 0.25° cell and by ROI |
| `scripts/analysis/29_temporal_offset.py` | Distribution of the ERA5 temporal offset in minutes (Section 2.4) |
| `scripts/analysis/19_collocation_sensitivity.py` | Table 4: nearest-node vs bilinear vs 3x3 averaging |

### 3. Figures of the comparison

| Script | Produces |
| --- | --- |
| `scripts/analysis/11_generate_figures.py` | Figures 1, 4, 5 |
| `scripts/analysis/11b_refresh_fig2_fig3.py` | Figures 2 and 3 (shadow class excluded) |

### 4. U-Net experiments

All experiments use the official CloudSEN12+ partition, which is disjoint by region of interest, and
are evaluated on the same 975-patch expert-labelled test set. Unannotated pixels (value 99) are
excluded from the loss and from every metric. Training crops 256x256 windows at native resolution;
evaluation is on the full 512x512 patch.

| Script | Produces |
| --- | --- |
| `scripts/analysis/21_unet_revision.py` | One configuration: `--mode expert`, `weak` or `baselines`, with `--n-train`, `--threshold`, `--bands`, `--seeds`, `--pretrained` |
| `scripts/analysis/23_run_all.py` | All 56 runs of the paper, in blocks; `--list` prints them, `--from N` resumes |
| `scripts/analysis/24_summarize.py` | `all_runs.csv` and the tables: main comparison (Tables 10, 11), weak-label scale (Table 12), expert budget from scratch vs fine-tuning (Table 13), binarisation threshold (Table 14), 4 vs 13 bands (Table 15) |
| `scripts/analysis/26_fig_scale.py` | Figure 6 (`fig14_scale`): performance vs number of weakly labelled patches |
| `scripts/analysis/27_fig_budget.py` | Figure 7 (`fig15_budget`): performance vs expert-label budget |

```bash
python scripts/analysis/20_download_scaleup.py --what test val train_expert
python scripts/analysis/20_download_scaleup.py --what train_weak --limit 20000
python scripts/analysis/22_repack_fast.py --splits test val train_expert train_weak
python scripts/analysis/21_unet_revision.py --mode baselines
python scripts/analysis/23_run_all.py
python scripts/analysis/24_summarize.py
```

Each run writes its results to an incremental JSON file, so an interrupted sequence does not lose
completed trainings.

### Superseded scripts

`scripts/analysis/12_unet_experiment.py`, `12b`, `12c`, `12d` and `12e` are the experiments of the
first submitted version (306 patches, random per-patch split). They are kept for the record and are
replaced by script 21; no result in the revised manuscript comes from them.

## Notes on reproducibility

- The principal comparison uses five seeds for the expert model trained on the full expert set and
  three seeds for every other configuration. Per-seed values are in
  `results/unet_revision/all_runs.csv`.
- The partition is the one distributed with CloudSEN12+ (`tortilla:data_split`), so it is rebuilt by
  `20_download_scaleup.py` rather than stored here.
- CloudSEN12+ records must be read by position, not by DataFrame label: reading by label returns a
  valid but different patch without raising an error. Script 20 reads by position and checks every
  patch against the per-class percentages in the dataset metadata.
- `19_collocation_sensitivity.py` is restricted to the 1,077 collocations for which the archived
  ERA5 field coincides with the acquisition hour.

## Data sources

- **CloudSEN12+**: Aybar et al., *Data in Brief* 56 (2024) 110852, <https://cloudsen12.github.io>
- **ERA5**: Hersbach et al., *QJRMS* 146 (2020) 1999, via the Copernicus Climate Data Store
- **GFS**: NOAA NCEP, via the NOAA Open Data Dissemination archive on AWS

## Citation

```bibtex
@article{anzalone2026nwplabels,
  author  = {Anzalone, Anna and Pagliaro, Antonio},
  title   = {Can Scene-Level {NWP} Cloud Cover Replace Expert Pixel Labels? {A} Large-Scale
             Assessment with {ERA5}, {GFS} and {CloudSEN12+}},
  journal = {Applied Sciences},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT, see `LICENSE`. The datasets keep the licences of their respective providers.
