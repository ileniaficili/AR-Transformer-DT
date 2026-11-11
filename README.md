# AR-Transformer-DT

Source code of the AR-Transformer-DT estimator described in:

> I. Ficili, G. Tricomi, M. Giacobbe, A. Puliafito, *Next-Gen Crop Management: Joint Estimation of Transpiration and Stomatal Conductance for Digital Twin-Driven Irrigation*, Future Generation Computer Systems (under review).

The model jointly estimates daily canopy transpiration (ET) and stomatal conductance (g_s) of individual plants from the 96-step (15-min) intra-day meteorological sequence and the plant's lag-1 physiological measurements. Preprocessing, architecture definition and training/evaluation are contained in a single script: `ar_transformer_dt.py`.

## Requirements

Python 3.11 and the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python ar_transformer_dt.py --data-dir path/to/data --scaler-fit train --seed 7 --out results
```

| Option | Description |
|---|---|
| `--data-dir` | Folder containing the three input CSV files |
| `--scaler-fit` | `train` (default) or `full`, see [Reproducibility](#reproducibility) |
| `--seed` | Random seed (default 7) |
| `--out` | Output folder for the metrics (JSON) and the trained model |

The script prints the test-set metrics (computed on normalised values) and saves them, together with the trained model, in the output folder.

## Input data

The script expects three CSV files in `--data-dir`:

**`piante_pulite.csv`**: one row per plant and day

| Column | Description |
|---|---|
| `Data` | Recording date (YYYY-MM-DD) |
| `Trt` | Irrigation treatment label (e.g. T1–T4) |
| `Dehydratation rate (%)` | Imposed water-deficit level of the plant's treatment (%) |
| `Traspirazione` | Daily canopy transpiration |
| `gs_canopy` | Daily stomatal conductance |

The same plants must appear every day; within each day, plants are identified by their position after sorting by treatment.

**`meteo_esterno_15m.csv`**: external station, 15-min resolution

| Column | Description |
|---|---|
| `Datetime` | Timestamp |
| `Temp_Ext_2m` | External air temperature (°C) |
| `RH_Ext_2m` | External relative humidity (%) |
| `Rad_Solare_Ext` | Global solar radiation (W m⁻²) |

**`meteo_interno_15m.csv`**: internal station, 15-min resolution

| Column | Description |
|---|---|
| `Datetime` | Timestamp |
| `Temp (°C)` | Internal air temperature |
| `RH (%)` | Internal relative humidity |
| `PARLight (µmol/m<sup>2</sup>/s)` | Photosynthetically active radiation |
| `VPD (kPa)` | Vapour pressure deficit |

Days without 96 complete meteorological steps are skipped.

## Protocol

- Chronological split: first 80% of recording days for training, last 20% for testing.
- Min-Max normalisation of meteorological channels, deficit level and targets (scaler fitting selected with `--scaler-fit`).
- Lag-1 features ET(t-1) and g_s(t-1) computed per plant (set to 0 on the first day of each plant).
- Adam (learning rate 1e-3), batch size 32, MSE loss, at most 50 epochs, early stopping on a 10% validation split (patience 10, best weights restored).

## Reproducibility

This repository implements the preprocessing pipeline, the architecture definition and the training/evaluation procedure of the AR-Transformer-DT described in the paper.

**Normalisation.** Two Min-Max scaler settings are supported via `--scaler-fit`:

- `full`: scalers fitted on the full dataset before the chronological split. This is the setting used for all results reported in the paper (Section 5.2).
- `train` (default): scalers fitted on the training partition only, which avoids any exposure of the training procedure to the range of the test period.

To reproduce the configuration of the paper:

    python ar_transformer_dt.py --data-dir path/to/data --scaler-fit full --seed 7 --out results

The values reported in the paper were obtained with seed 7, TensorFlow 2.16.2 on an Apple M4 (Metal backend). Over five random seeds (7, 42, 123, 999, 2025; TensorFlow 2.20 on CPU), the two settings yield comparable accuracy:

| Scaler fitting | R²_ET (mean ± std) | R²_gs (mean ± std) |
|---|---|---|
| `train` (default) | 0.631 ± 0.059 | 0.663 ± 0.070 |
| `full` (paper)    | 0.537 ± 0.105 | 0.674 ± 0.092 |

Results depend on the random seed and on the hardware/software environment, so single-seed values may differ from those reported in the paper.

**Data.** The gravimetric and microclimatic dataset is proprietary to Agrigeos s.r.l. and is not distributed with this repository; it may be provided by the corresponding author upon reasonable request and with the company's permission. The expected input format is described in [Input data](#input-data).

## License

MIT License, see `LICENSE`.
