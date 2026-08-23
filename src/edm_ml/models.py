from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Pipeline
    grid: dict[str, list]


def preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in X if X[c].dtype == "object"]
    numeric = [c for c in X if c not in categorical]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
    ])


def candidate_models(X: pd.DataFrame, seed: int = 42, fast: bool = False) -> list[ModelSpec]:
    p = preprocessor(X)
    trees = 30 if fast else 600
    models = [
        ModelSpec("ridge", Pipeline([("prep", p), ("model", Ridge())]),
                  {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
        ModelSpec("elastic_net", Pipeline([("prep", p), ("model", ElasticNet(max_iter=20000, random_state=seed))]),
                  {"model__alpha": [0.001, 0.01, 0.1, 1.0], "model__l1_ratio": [0.1, 0.5, 0.9]}),
        ModelSpec("svr", Pipeline([("prep", p), ("model", SVR())]),
                  {"model__C": [0.1, 1, 10, 100], "model__epsilon": [0.01, 0.1, 0.2],
                   "model__gamma": ["scale", 0.1, 1.0]}),
        ModelSpec("random_forest", Pipeline([("prep", p),
                  ("model", RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=-1))]),
                  {"model__max_depth": [2, 3, None], "model__min_samples_leaf": [1, 2, 3],
                   "model__max_features": [0.7, 1.0]}),
        ModelSpec("extra_trees", Pipeline([("prep", p),
                  ("model", ExtraTreesRegressor(n_estimators=trees, random_state=seed, n_jobs=-1))]),
                  {"model__max_depth": [2, 3, None], "model__min_samples_leaf": [1, 2, 3],
                   "model__max_features": [0.7, 1.0]}),
        ModelSpec("gaussian_process", Pipeline([("prep", p), ("model", GaussianProcessRegressor(
                  kernel=ConstantKernel(1.0) * Matern(nu=1.5) + WhiteKernel(0.1),
                  normalize_y=True, n_restarts_optimizer=0 if fast else 8, random_state=seed))]), {}),
        ModelSpec("pls", Pipeline([("prep", p), ("model", PLSRegression(scale=False))]),
                  {"model__n_components": [1, 2, 3, 4]}),
    ]
    if fast:
        # Smoke-test mode exercises linear, nonlinear-tree, and probabilistic paths
        # without the full publication grid.
        models[0] = ModelSpec("ridge", models[0].estimator,
                              {"model__alpha": [0.1, 1.0, 10.0]})
        models[4] = ModelSpec("extra_trees", models[4].estimator,
                              {"model__max_depth": [3], "model__min_samples_leaf": [2],
                               "model__max_features": [1.0]})
        return [models[0], models[4]]
    return models
