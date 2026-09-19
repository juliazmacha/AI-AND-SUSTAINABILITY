# AI-AND-SUSTAINABILITY

**Project:** A Predictive Emissions Monitoring System (PEMS) soft sensor — predicting gas turbine NOx emissions from cheap process sensors, using an MLP and XGBoost, compared on accuracy and computational cost, explained with SHAP, and compressed with two techniques per model. CO (the other measured emission) is deliberately excluded from the model inputs: a PEMS soft sensor must infer emissions from process sensors alone, so using one pollutant reading to predict another would leak target-adjacent information.

## Structure

```
data/                   UCI Gas Turbine CO/NOx dataset — gt_2011.csv ... gt_2015.csv
notebooks/               sequential pipeline, run in numeric order (see below)
requirements.txt          Python dependencies
```

## Environment

```
python -m venv .venv
.venv\Scripts\activate        (Windows)
pip install -r requirements.txt
```

In VS Code: install the "Python" and "Jupyter" extensions, open this folder, select the `.venv` interpreter, then open any notebook and Run All.

**Dependencies:** pandas, numpy, scikit-learn, xgboost, torch, shap, matplotlib, seaborn, jupyter (see `requirements.txt` for the full list). CPU-only throughout — no GPU or specialised hardware required, deliberately, so the project is reproducible on any standard machine.

## Notebooks (run in this order)

1. **`1_data_loading_and_preprocessing.ipynb`** — loads the 5 yearly CSVs, checks for missing values/physically impossible readings (fixes an ambient-humidity issue), reports statistical outliers without removing them, creates two physics-motivated candidate features (`TDROP`, `PRATIO`), and builds a chronological train (2011–2013) / validation (2014) / test (2015) split.
2. **`2_eda.ipynb`** — feature distributions, boxplots, a correlation check against combustion physics, and a year-by-year drift check on mean CO/NOx.
3. **`3_modelling.ipynb`** — starts with a feature-set selection (the two engineered candidates are kept only if they improve validation RMSE; they were rejected, and the decision is recorded in `artifacts/features.json`), then trains an MLP (PyTorch) and XGBoost to predict NOx, first as untuned baselines, then via a validation-based hyperparameter search (not k-fold CV, to respect the time ordering of the data).
4. **`4_evaluation.ipynb`** — the first and only use of the 2015 test set: final accuracy metrics, model size/inference-latency comparison, and SHAP explainability for both models, interpreted against combustion physics.
5. **`5_compression.ipynb`** — applies two compression techniques to each tuned model (MLP: structured pruning + dynamic quantisation; XGBoost: ensemble truncation + knowledge distillation) and re-evaluates every variant on the same test set and metrics.
6. **`6_model_validity_check.ipynb`** — diagnostic notebook separating model quality from concept drift: an in-distribution holdout check, a random-split sensitivity run (literature protocol), and a per-year error breakdown of the tuned models.

## Reproducibility notes

- A fixed random seed (`SEED = 42`) is set in every notebook for NumPy, PyTorch, and Python's `random` module.
- The train/validation/test split is chronological and fixed by year, not randomly sampled, so it is identical on every run.
- XGBoost's hyperparameter search and the MLP's tuning trials both use a fixed random seed for candidate selection, so re-running Notebook 3 should reproduce the same selected hyperparameters (minor floating-point differences in exact timing/loss values between hardware are expected and don't affect conclusions).

## Code sources

No starter code from the module's lab notebooks was copied into this project; the data loading, preprocessing, modelling, evaluation, and compression code was written for this specific dataset and task. Standard library functionality (e.g. `sklearn.preprocessing.StandardScaler`, `torch.quantization.quantize_dynamic`, `shap.Explainer`) is used via its public documented API.

## Known limitations

Both models show meaningfully worse accuracy on the 2015 test set than on the 2014 validation set used to tune them. This is attributed to concept drift, first identified in the year-by-year trend analysis in Notebook 2 (mean NOx shifts noticeably from 2013 onwards) and consistent with the SHAP findings in Notebook 4 (predictions rest heavily on ambient variables, whose relationship to NOx changes across years). It is discussed as the project's central limitation in the report's discussion section.
