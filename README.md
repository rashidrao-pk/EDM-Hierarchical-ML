# EDM-Hierarchical-ML

Interpretable hierarchical machine learning for small-sample prediction of surface and tribological responses in electrical discharge machining.

Reproducible analysis code for the three-stage EDM dataset:

1. `(I, E, D) -> (Ra, Hm, hc)`
2. `(I, E, D, Ra, Hm, hc) -> (Ff, COF, Wd, T, WL)`
3. `(I, E, D, Ra, Hm, hc, Ff, COF, Wd, T, WL) -> Keff`

The repository is designed for a small balanced experiment (20 runs; five `I`
levels and two levels each of `E` and `D`). It deliberately avoids deep neural
networks. It combines design-of-experiments analysis, leakage-safe nested
cross-validation, uncertainty-aware regression, explainability, and
multi-objective optimization.

## What the pipeline produces

- a machine-readable data-quality report;
- cleaned and harmonized tables;
- descriptive statistics, correlations, main-effect and interaction plots;
- partial-F ANOVA tables for all outcomes;
- model comparison using nested leave-one-out cross-validation;
- a stricter leave-one-`I`-level-out extrapolation evaluation;
- out-of-fold predictions and bootstrap confidence intervals;
- permutation importance calculated only from held-out predictions;
- a direct end-to-end model and three hierarchical stage models;
- Pareto-optimal observed operating conditions;
- image extraction and an inventory of the SEM/EDS panels in the DOCX file.

## Important scientific safeguards

- `Hm` differs by an exact factor of `8.181818...` between Type 1 and Types
  2/3. The code reports this but does not silently overwrite it.
- The unnamed constant and duplicated columns in Type 1 are excluded.
- The empty `k` column is reported; only `Keff` is modelled.
- `WL` and `Wl` are harmonized to `WL`.
- Hyperparameters are selected inside each outer fold.
- Results are explicitly labelled exploratory because there are only 20 runs
  and no experimental replicates.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_analysis.py --config configs/study.yaml
```

All outputs are written to `results/`.

For a quicker smoke test:

```bash
python scripts/run_analysis.py --config configs/study.yaml --fast
```

## Publication-oriented evaluation

Report both validation protocols:

- **Nested LOOCV:** estimates interpolation performance across individual
  experimental runs.
- **Leave-one-I-level-out:** evaluates extrapolation to a completely unseen
  current/operating level and is substantially harder.

Do not select the final model from training-set fit or a single random split.
For a paper, report MAE, RMSE, R2, normalized MAE, and bootstrap 95% confidence
intervals. Negative cross-validated R2 is a valid result and should not be
removed.

## Suggested paper framing

**Interpretable Hierarchical Machine Learning and Multi-objective Optimization
for EDM Surface and Tribological Performance under Small-data Constraints**

The strongest contribution is methodological reliability under a very small
designed experiment, not model complexity. Before submission, add experimental
replicates or new operating conditions, document all units, resolve the
hardness conversion, define `hc` and `Keff` precisely, and link every original
SEM/EDS image to its experimental row.
