from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_front(df: pd.DataFrame, minimize: list[str], maximize: list[str]) -> pd.DataFrame:
    values = df[minimize + maximize].astype(float).copy()
    values[maximize] *= -1.0
    arr = values.to_numpy()
    efficient = np.ones(len(arr), dtype=bool)
    for i, point in enumerate(arr):
        if not efficient[i]:
            continue
        dominated_by_other = np.any(np.all(arr <= point, axis=1) & np.any(arr < point, axis=1))
        if dominated_by_other:
            efficient[i] = False
    result = df.loc[efficient].copy()
    result.insert(0, "pareto_optimal", True)
    return result.sort_values(["I", "E", "D"])

