"""Shared pytest configuration.

The sandboxed runtime used for local test execution cannot always delete
entries created under the system temp directory, which makes pytest's
default `tmp_path_factory` cleanup explode with PermissionError.
Anchoring the base temp inside the repo works around that.
"""

import os
from pathlib import Path


def pytest_configure(config):
    basetemp = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    basetemp.mkdir(exist_ok=True)
    # Only override if the user didn't already pass --basetemp on the CLI
    if not config.option.basetemp:
        config.option.basetemp = str(basetemp)
    os.environ.setdefault("TMPDIR", str(basetemp))
