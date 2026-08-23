from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["I", "E", "D"]


def _read_sheet(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the three-row-header Excel layout and retain an audit trail."""
    path = Path(path)
    raw = pd.read_excel(path, sheet_name="Data", header=2)
    original_columns = [str(c) for c in raw.columns]
    unnamed = [c for c in raw.columns if str(c).startswith("Unnamed")]
    audit: dict[str, Any] = {
        "file": str(path),
        "original_shape": list(raw.shape),
        "original_columns": original_columns,
        "unnamed_columns": [str(c) for c in unnamed],
        "unnamed_diagnostics": {},
    }
    for col in unnamed:
        values = raw[col]
        duplicate_of = next(
            (c for c in raw.columns if c != col and values.equals(raw[c])), None
        )
        audit["unnamed_diagnostics"][str(col)] = {
            "n_unique": int(values.nunique(dropna=False)),
            "constant": bool(values.nunique(dropna=False) == 1),
            "duplicate_of": None if duplicate_of is None else str(duplicate_of),
        }
    clean = raw.drop(columns=unnamed).rename(columns={"Wl": "WL"})
    clean = clean.dropna(axis=1, how="all")
    clean["E"] = clean["E"].astype(str).str.strip()
    clean["D"] = clean["D"].astype(str).str.strip()
    audit["clean_shape"] = list(clean.shape)
    audit["missing"] = {c: int(v) for c, v in clean.isna().sum().items()}
    audit["duplicates_on_keys"] = int(clean.duplicated(KEYS).sum())
    return clean, audit


def load_study(type1: str | Path, type2: str | Path, type3: str | Path):
    t1, a1 = _read_sheet(type1)
    t2, a2 = _read_sheet(type2)
    t3, a3 = _read_sheet(type3)
    audits = {"type1": a1, "type2": a2, "type3": a3}

    for name, frame in {"type1": t1, "type2": t2, "type3": t3}.items():
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} has duplicate experimental conditions")
        if frame[KEYS].isna().any().any():
            raise ValueError(f"{name} has missing experimental keys")

    key12 = t1[KEYS].reset_index(drop=True).equals(t2[KEYS].reset_index(drop=True))
    key23 = t2[KEYS].reset_index(drop=True).equals(t3[KEYS].reset_index(drop=True))
    audits["cross_file"] = {"type1_type2_keys_equal": key12, "type2_type3_keys_equal": key23}
    if not (key12 and key23):
        raise ValueError("Experimental-condition rows do not align across files")

    shared = sorted((set(t1) & set(t2)) - set(KEYS))
    audits["cross_file"]["shared_type1_type2_max_abs_diff"] = {
        c: float(np.max(np.abs(t1[c].astype(float) - t2[c].astype(float)))) for c in shared
    }
    if "Hm" in t1 and "Hm" in t2:
        ratios = t1["Hm"].astype(float) / t2["Hm"].astype(float)
        audits["cross_file"]["hm_ratio_type1_over_type2"] = {
            "mean": float(ratios.mean()), "std": float(ratios.std()),
            "all_equal": bool(np.allclose(ratios, ratios.iloc[0])),
        }

    # Master table uses the later-stage units because they are consistent in T2/T3.
    master = t3.copy()
    return {"type1": t1, "type2": t2, "type3": t3, "master": master}, audits


def save_audit(audit: dict[str, Any], output: str | Path) -> None:
    Path(output).write_text(json.dumps(audit, indent=2), encoding="utf-8")

