"""Tests for i-PI driver installation."""

from __future__ import annotations

import subprocess
from importlib.util import find_spec
from pathlib import Path

import pytest


@pytest.fixture
def clean_driver():
    """Remove any existing installed driver before/after a test."""

    spec = find_spec("ipi")
    assert spec
    assert spec.submodule_search_locations

    pes_dir = Path(spec.submodule_search_locations[0]) / "pes"
    target_path = pes_dir / "lorem.py"

    if target_path.exists():
        target_path.unlink()

    yield target_path

    if target_path.exists():
        target_path.unlink()


def test_install_via_cli(clean_driver: Path):
    """Invoke CLI and verify driver is installed with expected contents."""

    target_path = clean_driver

    subprocess.run(
        ["loremjax-install-ipi-driver"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert target_path.exists()
    content = target_path.read_text()
    assert "LOREM_driver" in content
    assert "ASEDriver" in content
