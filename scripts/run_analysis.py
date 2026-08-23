from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edm_ml.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete EDM ML study")
    parser.add_argument("--config", default=str(ROOT / "configs" / "study.yaml"))
    parser.add_argument("--fast", action="store_true", help="Reduced resampling for a smoke test")
    args = parser.parse_args()
    output = run(args.config, fast=args.fast)
    print(f"Completed: {output}")


if __name__ == "__main__":
    main()

