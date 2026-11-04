#!/usr/bin/env python3
"""Print a read-only aggregate of a bound, redacted SHADOW replay directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from std0_quant.research.agent_shadow.replay_report import summarize_replay  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="a replay v3 output directory")
    args = parser.parse_args(argv)
    try:
        report = summarize_replay(args.input_dir)
    except OSError as exc:
        print(f"cannot read replay input files ({type(exc).__name__})", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
