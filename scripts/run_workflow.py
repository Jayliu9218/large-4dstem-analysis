from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fourdstem_pipeline import run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the non-visual 4D-STEM analysis workflow.")
    parser.add_argument("--config", default="configs/default_workflow.yaml", help="YAML workflow config path")
    args = parser.parse_args()

    result = run_workflow(args.config)
    print(f"Workflow finished. Summary: {result.summary_path}")


if __name__ == "__main__":
    main()
