import numpy as np

from collections import namedtuple

from jaxpme.batched_mixed.batching import get_batch as jaxpme_batcher
from jaxpme.batched_mixed.batching import prepare as jaxpme_prepare
from marathon.data.batching import batch_labels
from marathon.data.properties import DEFAULT_PROPERTIES
from marathon.data.sample import to_labels
from marathon.utils import next_size

Batch = namedtuple(
    "Batch",
    (
        "atomic_numbers",
        "sr",
        "nopbc",
        "pbc",
        "labels",
    ),
)


Sample = namedtuple(
    "Sample",
    (
        "structure",
        "labels",
    ),
)


def to_batch(
    samples,
    keys,
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
            halfspace=True,
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
            "halfspace": True,
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

    return Batch(atomic_numbers, sr, nopbc, pbc, labels)


def to_sample(
    atoms,
    cutoff,
    keys=None,
    energy=True,
    forces=True,
    stress=False,
    lr_wavelength=None,
    smearing=None,
    properties=DEFAULT_PROPERTIES,
):
    structure = jaxpme_prepare(
        atoms, cutoff, lr_wavelength=lr_wavelength, smearing=smearing, dtype=np.float32
    )
    labels = to_labels(
        atoms,
        keys=keys,
        energy=energy,
        forces=forces,
        stress=stress,
        properties=properties,
    )

    return Sample(structure, labels)
