"""Run the archived research against the parent repository's replay engine.

Usage from repository root: python -B quant-report/run.py final_report
The archived research implementation is unchanged. This entry point redirects
its input root and outputs without modifying the original experiment files.
"""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import re
import sys
import unittest

BUNDLE = Path(__file__).resolve().parent
PROJECT = BUNDLE.parent


def configure():
    if not (PROJECT / "l2sim" / "engine.py").is_file():
        raise SystemExit("Place quant-report directly inside the original repository, next to l2sim/.")
    # Archived research takes precedence over any root-level research package.
    sys.path[:0] = [str(BUNDLE), str(PROJECT)]
    pipeline = importlib.import_module("research.pipeline")
    pipeline.ROOT = PROJECT
    pipeline.CLEAN = PROJECT / "data" / "cleaned" / pipeline.DAY
    pipeline.OUTPUT = BUNDLE / "reports" / f"factor_research_{pipeline.DAY}"
    execution = importlib.import_module("research.execution_study")
    execution.OUT = BUNDLE / "reports" / f"execution_research_{pipeline.DAY}"
    final = importlib.import_module("research.final_study")
    final.OUT = BUNDLE / "reports" / f"final_research_{pipeline.DAY}"
    assert final.PREVIOUS == execution.OUT


def portable_report_links():
    for path in (BUNDLE / "reports").rglob("*.md"):
        original = path.read_text(encoding="utf-8")
        def replace(match):
            target = Path(match[2])
            if not target.is_absolute():
                return match[0]
            return match[1] + Path(os.path.relpath(target, path.parent)).as_posix() + match[3]
        updated = re.sub(r"(!\[[^\]]*\]\()([^)]*)(\))", replace, original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=["pipeline", "execution_study", "execution_report", "final_study", "final_report", "test"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    configure()
    if args.module == "test":
        suite = unittest.defaultTestLoader.discover(str(BUNDLE / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    module = importlib.import_module("research." + args.module)
    sys.argv = [str(Path(module.__file__)), *args.arguments]
    entry = module.write_report if args.module.endswith("_report") else module.main
    entry()
    portable_report_links()


if __name__ == "__main__":
    main()
