from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def anova_for_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partial-F ANOVA for the pre-specified small-data interaction model.

    Every effect is tested after adjustment for all other effects. E and D must
    each have two levels, as in the supplied balanced experiment.
    """
    e_levels, d_levels = sorted(df["E"].unique()), sorted(df["D"].unique())
    if len(e_levels) != 2 or len(d_levels) != 2:
        raise ValueError("The pre-specified ANOVA requires binary E and D")
    i = df["I"].astype(float).to_numpy()
    i = i - i.mean()
    e = (df["E"].to_numpy() == e_levels[1]).astype(float)
    d = (df["D"].to_numpy() == d_levels[1]).astype(float)
    columns = {
        "Intercept": np.ones(len(df)), "I": i, "E": e, "D": d,
        "I:E": i * e, "I:D": i * d, "E:D": e * d,
    }
    names = list(columns)
    X = np.column_stack([columns[n] for n in names])
    y = df[target].astype(float).to_numpy()
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    sse_full = float(residual @ residual)
    df_error = len(y) - rank
    mse = sse_full / df_error
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * mse)
    t_stat = beta / se
    coef = pd.DataFrame({"term": names, "coefficient": beta, "std_error": se,
                         "t": t_stat, "p_value": 2 * stats.t.sf(np.abs(t_stat), df_error)})
    rows = []
    for j, term in enumerate(names[1:], start=1):
        reduced = np.delete(X, j, axis=1)
        reduced_beta = np.linalg.lstsq(reduced, y, rcond=None)[0]
        reduced_residual = y - reduced @ reduced_beta
        ss_effect = float(reduced_residual @ reduced_residual - sse_full)
        f_value = max(0.0, ss_effect / mse)
        rows.append({"term": term, "sum_sq": ss_effect, "df": 1,
                     "F": f_value, "p_value": float(stats.f.sf(f_value, 1, df_error))})
    rows.append({"term": "Residual", "sum_sq": sse_full, "df": df_error,
                 "F": np.nan, "p_value": np.nan})
    return pd.DataFrame(rows), coef


def descriptive_tables(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.describe(include="all").T.to_csv(output_dir / "descriptive_statistics.csv")
    df.select_dtypes(include=np.number).corr(method="pearson").to_csv(output_dir / "pearson_correlations.csv")
    df.select_dtypes(include=np.number).corr(method="spearman").to_csv(output_dir / "spearman_correlations.csv")
