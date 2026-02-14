"""Helper to install the LOREM driver into i-PI."""

import shutil
from importlib.util import find_spec
from pathlib import Path


def install_ipi_driver():
    """Copy the installed `lorem.py` into the i-PI `pes` directory."""

    # Locate i-PI installation
    ipi_spec = find_spec("ipi")
    if ipi_spec is None or not ipi_spec.submodule_search_locations:
        raise RuntimeError(
            "i-PI installation not found. Install i-PI and rerun "
            "`lorem-install-ipi-driver`.",
        )
    pes_dir = Path(ipi_spec.submodule_search_locations[0]) / "pes"

    # Locate source driver module
    source_spec = find_spec("lorem.dynamics.ipi")
    if source_spec is None or source_spec.origin is None:
        raise FileNotFoundError("Could not locate lorem.dynamics.ipi module")
    source_path = Path(source_spec.origin)

    target_path = pes_dir / "lorem.py"
    shutil.copy(source_path, target_path)


if __name__ == "__main__":
    install_ipi_driver()
