from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, KFold

from .data import load_study, save_audit
from .evaluation import compare_models
from .images import inventory_and_extract
from .models import candidate_models
from .optimization import pareto_front
from .plots import correlation_heatmap, effects_plot, observed_vs_predicted, setup_style
from .statistics import anova_for_target, descriptive_tables


def _resolve(base: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def run(config_path: str | Path, fast: bool = False) -> Path:
    config_path = Path(config_path).resolve()
    project = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text())
    out = _resolve(project, config["output_dir"])
    for sub in ["audit", "tables", "figures", "predictions", "models", "images"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    frames, audit = load_study(*[_resolve(project, data_cfg[k]) for k in ("type1", "type2", "type3")])
    audit["scientific_limitations"] = [
        "Only 20 experimental runs", "No replicate measurements",
        "Hardness scale differs between Type 1 and Types 2/3",
        "Keff currently has weak cross-validated predictability",
    ]
    save_audit(audit, out / "audit" / "data_quality.json")
    for name, frame in frames.items():
        frame.to_csv(out / "tables" / f"clean_{name}.csv", index=False)

    master = frames["master"]
    descriptive_tables(master, out / "tables")
    setup_style()
    correlation_heatmap(master, out / "figures" / "correlation_heatmap.png")

    all_targets = list(dict.fromkeys(t for task in config["tasks"].values() for t in task["targets"]))
    anova_tables = []
    for target in all_targets:
        if target not in master:
            continue
        table, coefficients = anova_for_target(master, target)
        table.insert(0, "target", target); coefficients.insert(0, "target", target)
        anova_tables.append(table)
        coefficients.to_csv(out / "tables" / f"coefficients_{target}.csv", index=False)
        effects_plot(master, target, out / "figures" / f"effects_{target}.png")
    pd.concat(anova_tables, ignore_index=True).to_csv(out / "tables" / "anova_partial_f.csv", index=False)

    summary_rows = []
    bootstrap = 250 if fast else int(config.get("bootstrap_iterations", 2000))
    for task_name, task in config["tasks"].items():
        X = master[task["features"]]
        specs = candidate_models(X, config["seed"], fast)
        for target in task["targets"]:
            if target not in master:
                continue
            y = master[target].astype(float)
            for protocol in ["nested_loocv", "leave_one_I_level_out"]:
                comparison, predictions = compare_models(
                    X, y, specs, protocol, config["seed"], bootstrap)
                comparison.insert(0, "task", task_name)
                comparison.insert(1, "target", target)
                summary_rows.append(comparison)
                comparison.to_csv(out / "tables" / f"models_{task_name}_{target}_{protocol}.csv", index=False)
                for model_name, pred in predictions.items():
                    pred.assign(**{c: master[c].values for c in ["I", "E", "D"]}).to_csv(
                        out / "predictions" / f"{task_name}_{target}_{protocol}_{model_name}.csv", index=False)
                best = comparison.iloc[0]["model"]
                observed_vs_predicted(predictions[best],
                    f"{target}: {best} ({protocol})",
                    out / "figures" / f"pred_{task_name}_{target}_{protocol}.png")

            # Fit a final exploratory model after evaluation; never use this fit for reported CV metrics.
            loocv_table = summary_rows[-2]
            best_name = loocv_table.iloc[0]["model"]
            best_spec = next(s for s in specs if s.name == best_name)
            cv = KFold(n_splits=5, shuffle=True, random_state=config["seed"])
            search = GridSearchCV(best_spec.estimator, best_spec.grid or {},
                                  scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, refit=True)
            search.fit(X, y)
            joblib.dump({"model": search.best_estimator_, "features": task["features"],
                         "target": target, "training_range": {c: [str(master[c].min()), str(master[c].max())]
                         for c in task["features"]}}, out / "models" / f"{task_name}_{target}.joblib")
            perm = permutation_importance(search.best_estimator_, X, y, scoring="neg_mean_absolute_error",
                                          n_repeats=50 if fast else 300, random_state=config["seed"])
            pd.DataFrame({"feature": task["features"], "importance_mean": perm.importances_mean,
                          "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False).to_csv(
                              out / "tables" / f"permutation_{task_name}_{target}.csv", index=False)

    pd.concat(summary_rows, ignore_index=True).to_csv(out / "tables" / "model_comparison_all.csv", index=False)
    opt = config["optimization"]
    pareto_front(master, opt["minimize"], opt["maximize"]).to_csv(
        out / "tables" / "observed_pareto_front.csv", index=False)
    images_path = _resolve(project, data_cfg["images_docx"])
    if images_path.exists():
        inventory_and_extract(images_path, out / "images")
    (out / "RUN_COMPLETE.txt").write_text(
        "Analysis completed. Results are exploratory until units, Hm conversion, k/Keff definition, and replicates are resolved.\n")
    return out
