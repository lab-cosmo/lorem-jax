# /// script
# requires-python = ">=3.11"
# dependencies = ["flax", "numpy"]
# ///
"""Export a lorem-jax checkpoint for consumers that cannot import lorem-jax.

Needs flax (and so JAX), but neither lorem-jax nor marathon.

Usage: uv run scripts/export.py CHECKPOINT OUT

Writes to OUT:
- params.npz: flax parameter leaves, keyed by `/`-joined paths without `params/`
- model.yaml, baseline.yaml: copied from the checkpoint
- export.yaml: export format version

Leaves are raw: no transposes or reshaping for any particular consumer.
"""

import numpy as np

import shutil
import sys
from pathlib import Path

FORMAT = 1


def export(checkpoint, out):
    from flax.serialization import msgpack_restore

    model = Path(checkpoint) / "model"
    params = msgpack_restore((model / "model.msgpack").read_bytes())
    yamls = [model / name for name in ("model.yaml", "baseline.yaml")]
    for path in yamls:
        if not path.is_file():
            raise FileNotFoundError(path)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "params.npz", **flatten(params["params"]))
    for path in yamls:
        shutil.copy(path, out / path.name)
    (out / "export.yaml").write_text(f"format: {FORMAT}\n")


def flatten(tree, prefix=""):
    flat = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}/"))
        else:
            flat[f"{prefix}{key}"] = np.asarray(value)
    return flat


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    export(*sys.argv[1:])
