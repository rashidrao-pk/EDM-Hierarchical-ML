from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, LeaveOneGroupOut

from .models import ModelSpec


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    span = float(np.max(y) - np.min(y))
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "nmae_range": float(mean_absolute_error(y, pred) / span) if span else np.nan,
    }


def bootstrap_metric_ci(y, pred, iterations=2000, seed=42) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    values = {"mae": [], "rmse": [], "r2": []}
    for _ in range(iterations):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        m = metrics(y[idx], pred[idx])
        for key in values:
            values[key].append(m[key])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in values.items()}


def _nested_predictions(X, y, spec: ModelSpec, outer, seed=42):
    pred = np.full(len(y), np.nan)
    selected: list[str] = []
    fold_id = np.full(len(y), -1)
    split_args = (X, y)
    if isinstance(outer, LeaveOneGroupOut):
        groups = X["I"].to_numpy()
        splits = outer.split(X, y, groups)
    else:
        splits = outer.split(X, y)
    for fold, (train, test) in enumerate(splits):
        estimator = clone(spec.estimator)
        inner_splits = min(5, max(2, len(train) // 4))
        inner = KFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold)
        if spec.grid:
            search = GridSearchCV(estimator, spec.grid, scoring="neg_mean_absolute_error",
                                  cv=inner, n_jobs=-1, refit=True)
            search.fit(X.iloc[train], y.iloc[train])
            fitted = search.best_estimator_
            selected.append(str(search.best_params_))
        else:
            fitted = estimator.fit(X.iloc[train], y.iloc[train])
            selected.append("{}")
        pred[test] = np.asarray(fitted.predict(X.iloc[test])).reshape(-1)
        fold_id[test] = fold
    return pred, fold_id, Counter(selected)


def compare_models(X: pd.DataFrame, y: pd.Series, specs: Iterable[ModelSpec],
                   protocol: str, seed=42, bootstrap_iterations=2000):
    outer = LeaveOneOut() if protocol == "nested_loocv" else LeaveOneGroupOut()
    rows, predictions = [], {}
    for spec in specs:
        pred, folds, selected = _nested_predictions(X, y, spec, outer, seed)
        result = metrics(y.to_numpy(), pred)
        ci = bootstrap_metric_ci(y.to_numpy(), pred, bootstrap_iterations, seed)
        rows.append({"model": spec.name, "protocol": protocol, **result,
                     "mae_ci_low": ci["mae"][0], "mae_ci_high": ci["mae"][1],
                     "rmse_ci_low": ci["rmse"][0], "rmse_ci_high": ci["rmse"][1],
                     "r2_ci_low": ci["r2"][0], "r2_ci_high": ci["r2"][1],
                     "most_common_hyperparameters": selected.most_common(1)[0][0]})
        predictions[spec.name] = pd.DataFrame({"observed": y.to_numpy(), "predicted": pred,
                                                "fold": folds}, index=y.index)
    return pd.DataFrame(rows).sort_values(["mae", "rmse"]), predictions

