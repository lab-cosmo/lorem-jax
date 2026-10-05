import numpy as np
import jax

import importlib.util
from pathlib import Path

import yaml
from flax.traverse_util import flatten_dict
from marathon.io import to_dict, write_msgpack, write_yaml

from lorem.models.bec import LoremBEC

SCRIPT = Path(__file__).parents[1] / "scripts/export.py"


def test_export_roundtrip(tmp_path):
    spec = importlib.util.spec_from_file_location("export", SCRIPT)
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)

    model = LoremBEC(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4)
    params = model.init(jax.random.key(0), *model.dummy_inputs())

    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "model").mkdir(parents=True)
    write_msgpack(checkpoint / "model/model.msgpack", params)
    write_yaml(checkpoint / "model/model.yaml", to_dict(model))
    write_yaml(checkpoint / "model/baseline.yaml", {"elemental": {18: -1.0}})

    out = tmp_path / "out"
    export.export(checkpoint, out)

    with np.load(out / "params.npz") as data:
        exported = {key: data[key] for key in data.files}
    expected = flatten_dict(params["params"], sep="/")

    assert exported.keys() == expected.keys()
    assert "Initial_0/ChemicalEmbedding_0/Embed_0/embedding" in exported
    for key, value in expected.items():
        np.testing.assert_array_equal(exported[key], value)

    for name in ("model.yaml", "baseline.yaml"):
        assert (out / name).read_text() == (checkpoint / "model" / name).read_text()
    assert yaml.safe_load((out / "export.yaml").read_text()) == {"format": 1}
