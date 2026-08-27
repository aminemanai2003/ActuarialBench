"""CLI wrapper for ActuarialBench report asset generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from actuarialbench.reporting import generate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    print(generate(args.experiment_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
