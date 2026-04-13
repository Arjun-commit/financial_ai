"""Shared pytest configuration."""

import os
from pathlib import Path


def pytest_configure(config):
    basetemp = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    basetemp.mkdir(exist_ok=True)
    # Only override if the user didn't already pass --basetemp on the CLI
    if not config.option.basetemp:
        config.option.basetemp = str(basetemp)
    os.environ.setdefault("TMPDIR", str(basetemp))
