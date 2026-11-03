#!/usr/bin/env python3
"""Run the CPU checks: core Rust tests and the Python test suite. No GPU or weights needed."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-only", action="store_true",
                        help="Run Python and package checks without Cargo")
    args = parser.parse_args()
    if not args.python_only and shutil.which("cargo") is None:
        parser.error("Install Rust 1.90.0, or use --python-only for the Python checks")
    if not args.python_only:
        subprocess.run(["cargo", "test", "--locked", "-p", "native-core-infra", "--lib"],
                       cwd=ROOT, check=True)
        subprocess.run(["cargo", "build", "--locked", "-p", "native-core-infra", "--lib"], cwd=ROOT, check=True)
    env = dict(os.environ)
    library = ROOT / 'target/debug' / ('libnative_core_infra.dylib' if sys.platform == 'darwin' else 'libnative_core_infra.so')
    if 'NATIVE_STARTUP_LIBRARY' not in env and library.exists():env['NATIVE_STARTUP_LIBRARY'] = str(library)
    if 'NATIVE_STARTUP_LIBRARY' not in env:
        parser.error('Build the native startup library, or set NATIVE_STARTUP_LIBRARY for --python-only')
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "models/wan2.1/tests"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
