import jax

from ase.io import read
from marathon import comms
from marathon.data import datasets, get_splits
from marathon.extra.hermes.data_source import prepare

data = read("./data.xyz", format="extxyz", index=":")

seed = 0
len_train = int(len(data) * 0.8)
len_valid = len(data) - len_train
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
}

prepare(
    [data[i] for i in idx_train],
    folder=datasets / "mlp_example/train",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
    properties=PROPERTIES,
)

prepare(
    [data[i] for i in idx_valid],
    folder=datasets / "mlp_example/valid",
    reporter=reporter,
    batch_size=500,
    samples_per_composition=100,
    properties=PROPERTIES,
)

reporter.done()
