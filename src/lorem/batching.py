import numpy as np

from collections import namedtuple
from functools import partial

from jaxpme.batched_mixed.batching import get_batch as jaxpme_batcher
from jaxpme.batched_mixed.batching import prepare as jaxpme_prepare
from marathon.data.batching import batch_labels, batch_properties
from marathon.data.properties import DEFAULT_PROPERTIES
from marathon.data.sample import to_sample as marathon_to_sample
from marathon.utils import next_size

Batch = namedtuple(
    "Batch",
    (
        "atomic_numbers",
        "sr",
        "nopbc",
        "pbc",
        "inputs",  # what the model reads (+ masks)
        "labels",  # what the model is scored against (+ masks)
    ),
)


def to_batch(
    samples,
    keys,
    inputs=(),
    batch_size=None,
    strategies={"default": "powers_of_2"},
    shapes=None,
    properties=DEFAULT_PROPERTIES,
):
    if batch_size is not None:
        assert batch_size > len(samples)
    else:
        batch_size = next_size(len(samples) + 1, strategy="powers_of_2")

    labels, structures = [], []

    for sample in samples:
        labels.append(sample.labels)
        structures.append(sample.structure)

    if shapes is None:
        default = strategies.pop("default", "powers_of_2")
        _, sr, nopbc, pbc = jaxpme_batcher(
            structures,
            strategy=default,
            num_structures_pbc=strategies.get("fine", default),
            num_pairs_nonpbc=strategies.get("coarse", default),
            num_pairs=strategies.get("coarse", default),
            num_structures=batch_size,
        )
    else:
        kwargs = {
            "num_structures": batch_size,
            "num_structures_pbc": shapes["pbc"],
            "num_atoms": shapes["atoms"],
            "num_atoms_pbc": shapes["atoms_pbc"],
            "num_pairs": shapes["pairs"],
            "num_pairs_nonpbc": shapes["pairs_nonpbc"],
            "num_k": shapes["k"],
            "strategy": "multiples",
        }
        _, sr, nopbc, pbc = jaxpme_batcher(
            structures,
            **kwargs,
        )

    num_structures = sr.cell.shape[0]
    num_atoms = sr.positions.shape[0]

    atomic_numbers = np.zeros(num_atoms, dtype=int)
    Z = np.concatenate([sample.structure["atomic_numbers"] for sample in samples])
    atomic_numbers[: len(Z)] = Z

    labels = batch_labels(labels, num_structures, num_atoms, keys, properties=properties)
    inputs = batch_properties(
        structures,
        inputs,
        num_structures,
        num_atoms,
        float_dtype=sr.positions.dtype,
        properties=properties,
    )

    return Batch(atomic_numbers, sr, nopbc, pbc, inputs, labels)


def to_sample(
    atoms,
    cutoff,
    keys=("energy", "forces"),
    inputs=(),
    lr_wavelength=None,
    smearing=None,
    properties=DEFAULT_PROPERTIES,
):
    return marathon_to_sample(
        atoms,
        cutoff,
        keys=keys,
        inputs=inputs,
        properties=properties,
        structure_fn=partial(to_structure, lr_wavelength=lr_wavelength, smearing=smearing),
    )


def to_structure(atoms, cutoff, float_dtype=None, int_dtype=None, **kwargs):
    # jax-pme structures are float32 regardless of the dtype used for labels and inputs
    return jaxpme_prepare(atoms, cutoff, dtype=np.float32, **kwargs)
