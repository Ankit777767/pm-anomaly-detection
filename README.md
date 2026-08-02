# Unsupervised Anomaly Detection for Predictive Maintenance

Detecting machine failures in an industrial milling process **without ever
training on failure labels** — because in real factories, labeled failure
data is rare, expensive, and arrives too late to be useful for training.

> **Core framing:** the dataset happens to include failure labels. This
> project deliberately does not use them for training. They are quarantined
> at the data-loading level and reappear only once, in Phase 5, to *evaluate*
> the unsupervised models. Every visualization in Phase 1 that shows the
> failure label is doing so for diagnostic/EDA purposes, never as a model input.

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | EDA, physics-informed feature engineering, PCA/UMAP diagnostics | ✅ Done |
| 2 | Preprocessing pipeline (scaling, encoding) | 🔜 |
| 3 | 6 unsupervised models + ablation study (feature sets × scalers × dim. reduction × hyperparams) | 🔜 |
| 4 | Evaluation (Precision/Recall/F1/ROC-AUC/PR-AUC) + explainability | 🔜 |
| 5 | Streamlit demo, deployed | 🔜 |

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(Matzka, 2020, *"Explainable Artificial Intelligence for Predictive
Maintenance Applications,"* 3rd Int'l Conf. on AI4I). 10,000 milling-process
records: air/process temperature, rotational speed, torque, tool wear,
product quality variant (L/M/H), and 5 documented failure modes
(TWF, HDF, PWF, OSF, RNF) at a combined 3.4% failure rate.

## Why this project (and not another clustering demo)

- **Genuinely unsupervised, not just unsupervised-flavored.** Labels are
  structurally separated from features in code (`data_loader.py`), not just
  "dropped in a notebook cell" — enforced with a dedicated test.
- **Physics-informed feature engineering**, not generic scaling. Every
  engineered feature maps to a documented failure rule (e.g. `power_w` ties
  directly to the Power Failure threshold), which is verified empirically
  in the EDA notebook, not just asserted.
- **Ablation-first design.** `configs/ablation_config.yaml` is the single
  source of truth for every experiment axis (feature sets, scalers,
  dimensionality reduction, model hyperparameters) — a one-line config change
  adds a new experiment, nothing is hardcoded in a notebook.
- **Both linear and nonlinear dimensionality reduction**, used to cross-check
  each other rather than picking one arbitrarily (see Key Findings below).

## Key findings so far (Phase 1)

1. **Failures are not linearly separable in 2D.** PCA's first two components
   explain 32.2% and 18.9% of variance; failures skew toward high PC1 but
   interleave with normal points rather than forming a distinct blob — ruling
   out a naive "PCA reconstruction only" detector as sufficient on its own.
2. **~5 of 12 components capture 90% of variance** — informs the PCA
   preprocessing setting used in the Phase 3 ablation grid.
3. **UMAP resolves 3 islands driven by product type (L/M/H)**, not by
   process physics — a confound that PCA's global-variance view doesn't
   surface as clearly. This is why the project uses *both* techniques rather
   than treating them as interchangeable.
4. **Engineered features are highly correlated with raw ones**
   (`power_w` ↔ `torque_nm`: r=0.98; `wear_torque_product` ↔ `tool_wear_min`:
   r=0.90) — meaning they may add less new signal than their physical
   motivation suggests. This is exactly why Phase 3 ablates `raw` vs.
   `engineered` feature sets empirically instead of assuming engineering helps.
5. **`temp_diff_k` and `wear_torque_product` show clean bimodal separation**
   for failures (both tails, not one side) — directly matching the documented
   HDF and OSF failure rules. See `reports/figures/03_feature_distributions.png`.

Full write-up with all 7 figures: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) · [`reports/eda_summary.md`](reports/eda_summary.md)

## Repo structure

```
pm-anomaly-detection/
├── data/raw/ai4i2020.csv        # original dataset
├── src/
│   ├── data_loader.py           # loading + leakage-safe feature/label split
│   ├── features.py               # physics-informed feature engineering
│   ├── dimensionality_reduction.py  # PCA + UMAP (EDA and modeling use)
│   ├── visualize.py               # all plotting, consistent style
│   └── run_eda.py                 # Phase 1 entry point
├── notebooks/01_eda.ipynb        # narrated walkthrough (executed, with outputs)
├── configs/ablation_config.yaml  # declarative ablation study catalog
├── reports/figures/               # 7 saved PNGs
├── tests/test_pipeline.py         # leakage + sanity tests (pytest)
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/Ankit777767/pm-anomaly-detection
cd pm-anomaly-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Reproduce Phase 1 end to end
python src/run_eda.py

# Or open the narrated notebook
jupyter lab notebooks/01_eda.ipynb

# Run tests
pytest tests/ -v
```

## Citation

S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
Applications," 2020 Third International Conference on Artificial
Intelligence for Industries (AI4I), 2020.
