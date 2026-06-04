"""Rebuild the emsim wheel that the web app (docs/) loads in Pyodide.

Run this whenever the Python source changes:

    python scripts/build_web.py

It builds a pure-Python wheel into docs/wheels/ and updates the version
referenced by docs/main.js (kept at emsim-<version>-py3-none-any.whl).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHEELS = ROOT / "docs" / "wheels"


def main() -> int:
    WHEELS.mkdir(parents=True, exist_ok=True)
    for old in WHEELS.glob("emsim-*.whl"):
        old.unlink()
    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(WHEELS)], cwd=ROOT
    )
    built = sorted(WHEELS.glob("emsim-*.whl"))
    print("built:", [p.name for p in built])
    print("If the version changed, update the wheel name in docs/main.js.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
