import jax

from ase.io import read
from marathon import comms
from marathon.data import datasets, get_splits
from marathon.extra.hermes.data_source import prepare

data = read("./bec_example.xyz", format="extxyz", index=":")

# reshape apt from (N, 9) to (N, 3, 3)
for atoms in data:
    if "apt" in atoms.arrays:
        atoms.arrays["apt"] = atoms.arrays["apt"].reshape(-1, 3, 3)

seed = 0
len_train = int(len(data) * 0.8)
len_valid = int(len(data) * 0.2)
idx_train, idx_valid, idx_test = get_splits(
    len(data), len_train, len_valid, 0, jax.random.key(seed)
)

reporter = comms.reporter()
reporter.start("processing")

PROPERTIES = {
    "energy": {
        "shape": (1,),
        "storage": "atoms.calc",
        "report_unit": (1000, "meV"),
        "symbol": "E",
    },
    "forces": {
        "shape": ("atom", 3),
        "storage": "atoms.calc",
        "report_unit": (1000, "meV/Å"),
        "symbol": "F",
    },
    "apt": {
        "shape": ("atom", 3, 3),
        "storage": "atoms.arrays",
        "report_unit": (1, "e"),
        "symbol": "Z",
    },
}

prepare(
    [data[i] for i in idx_train],
    folder=datasets / "bec_example/train",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
    properties=PROPERTIES,
)

prepare(
    [data[i] for i in idx_valid],
    folder=datasets / "bec_example/valid",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
    properties=PROPERTIES,
)

reporter.done()
