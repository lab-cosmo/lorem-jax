import jax

from ase.io import read
from marathon import comms
from marathon.data import datasets, get_splits
from marathon.grain import prepare

data = read("./data.xyz", format="extxyz", index=":")

seed = 0
len_train = int(len(data) * 0.8)
len_valid = len(data) - len_train
idx_train, idx_valid, idx_test = get_splits(
    len(data), len_train, len_valid, 0, jax.random.key(seed)
)

reporter = comms.reporter()
reporter.start("processing")

# `total_charge` is a model input, not a training label, so lorem/batching.py
# reads it directly from atoms.info rather than through the keys/loss-weight
# mechanism below. But it still needs to be declared here with
# storage="atoms.info" — marathon.grain.prepare() only persists atoms.info
# entries that are listed in this properties dict; anything else is silently
# dropped when the dataset is serialized to disk.
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
    "total_charge": {
        "shape": (1,),
        "storage": "atoms.info",
    },
}

prepare(
    [data[i] for i in idx_train],
    folder=datasets / "charge_conditioning_example/train",
    reporter=reporter,
    batch_size=8,
    samples_per_composition=100,
    properties=PROPERTIES,
)

prepare(
    [data[i] for i in idx_valid],
    folder=datasets / "charge_conditioning_example/valid",
    reporter=reporter,
    batch_size=8,
    samples_per_composition=100,
    properties=PROPERTIES,
)

reporter.done()
