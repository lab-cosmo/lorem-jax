import numpy as np
import jax

from ase.io import read
from marathon import comms
from marathon.data import datasets, get_splits
from marathon.grain import prepare

data = read("./data.xyz", format="extxyz", index=":")

# `bias_charge` is razor's name for the DFT bias charge each frame was
# evaluated at; `total_charge` is the key lorem/batching.py reads to populate
# the model's Q input. This rename is NOT cosmetic: marathon writes NaN for a
# declared-but-absent atoms.info key, and because total_charge is a model
# *input* rather than a label, that NaN reaches the network and takes the loss
# down from step 0. Two runs of this dataset were lost to exactly that.
for atoms in data:
    atoms.info["total_charge"] = atoms.info.pop("bias_charge")

seed = 0
len_train = int(len(data) * 0.75)
len_valid = len(data) - len_train
idx_train, idx_valid, idx_test = get_splits(
    len(data), len_train, len_valid, 0, jax.random.key(seed)
)

reporter = comms.reporter()
reporter.start("processing")

train_wf = np.array([data[i].info["work_function"] for i in idx_train])
comms.talk(f"train work function: mean {train_wf.mean():.3f} V, std {train_wf.std():.3f} V")

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
    "work_function": {
        "shape": (1,),
        "storage": "atoms.info",
        "report_unit": (1000, "mV"),
        "symbol": "Φ",
    },
    # a model input, not a label -- but prepare() only persists atoms.info
    # entries listed here, so it still has to be declared
    "total_charge": {
        "shape": (1,),
        "storage": "atoms.info",
    },
}

prepare(
    [data[i] for i in idx_train],
    folder=datasets / "work_function_example/train",
    reporter=reporter,
    batch_size=8,
    samples_per_composition=100,
    properties=PROPERTIES,
)

prepare(
    [data[i] for i in idx_valid],
    folder=datasets / "work_function_example/valid",
    reporter=reporter,
    batch_size=8,
    samples_per_composition=100,
    properties=PROPERTIES,
)

reporter.done()
