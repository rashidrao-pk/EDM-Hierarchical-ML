from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edm_ml.optimization import pareto_front


def test_pareto_front_removes_dominated_point():
    df = pd.DataFrame({"I": [1, 2, 3], "E": ["A"] * 3, "D": ["X"] * 3,
                       "cost": [1.0, 2.0, 1.5], "quality": [3.0, 2.0, 4.0]})
    result = pareto_front(df, minimize=["cost"], maximize=["quality"])
    assert set(result["I"]) == {1, 3}

