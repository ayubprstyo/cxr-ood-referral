# From Prediction to Referral — code and derived results

Code accompanying the manuscript *"From Prediction to Referral: Uncertainty-Aware Pneumonia Detection under Cross-Population
Distribution Shift in Chest Radiography"* (Prasetyo A, Amalia F; submitted to *Informatics and Health*).

Two classifiers (EfficientNet-B4, ViT-B/16) are trained on the paediatric Mooney/Kermany chest X-ray dataset and evaluated, frozen,
on adult NIH ChestX-ray14 cohorts used **only** for external evaluation. Monte Carlo Dropout uncertainty drives a selective-prediction
(referral) policy; Grad-CAM attention is audited against lung masks; robustness is checked across training seeds and without letterbox padding.

## Data (not redistributed)
| Dataset | Source |
|---|---|
| Mooney/Kermany paediatric CXR | Kaggle `paultimothymooney/chest-xray-pneumonia` |
| NIH ChestX-ray14 | Kaggle `nih-chest-xrays/data` (NIH Clinical Center) |

Cohorts are identified by SHA-256 fingerprints in `configs/cohort_fingerprints.json`; every notebook stops if its inputs do not match.

## Notebooks (run on Kaggle, GPU T4)
| Notebook | Purpose |
|---|---|
| `01b_dataset_integrity_audit` | audit of the Notebook 01 manifests; locks the Mooney splits and NIH cohorts |
| `02_model_training` | training; validation-only threshold selection |
| `03_id_ood_evaluation` | ID/OOD discrimination, calibration, patient-level bootstrap |
| `04_uncertainty_selective_prediction` | MC Dropout (50 passes), error/OOD detection, referral policy |
| `05_gradcam_safety_audit` | Grad-CAM, lung masks (TorchXRayVision), Lung Attention Ratio |
| `06_robustness_seeds_padding` | additional seeds (43, 44) and a no-padding variant |

The Notebook 01 manifest builder (raw inventory, SHA-256/pHash, NIH cohort extraction) produced the `cxr-ood-notebook01-output` input used by 01b.

## Analysis scripts (`analysis/`, run locally after the notebooks)
`03b_recompute_nb03_statistics_locked_version.py` recomputes the Notebook 03 statistics from the locked model version's logits;
`make_tables.py`, `make_figures.py`, `aspect_posthoc.py`, `make_robustness.py` and `extra_analyses.py` produce the publication tables,
figures and exploratory analyses. Extract each notebook's output ZIP to `outputs/notebook01`, `outputs/notebook01b`, `outputs/notebook02`,
`outputs/notebook03`, `outputs/notebook04`, `outputs/notebook05` and `outputs/notebook06` before running them.

## Results
`results/tables` (Tables 1–8), `results/supp` (Supplementary Tables S1–S15) and `results/metrics` (all estimates with 95% patient-level bootstrap CIs).
Locked operating thresholds and temperatures are in `configs/`.

## Environment
Python 3.13, PyTorch 2.11 (CUDA 12.8), timm 1.0.29, TorchXRayVision 1.5.7, NumPy 2.1, pandas 2.3, scikit-learn, SciPy, Matplotlib.

## Licence
Code: MIT (see `LICENSE`). Datasets remain under their original terms of use.
