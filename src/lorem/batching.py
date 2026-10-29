import numpy as np

from collections import namedtuple
from functools import partial

from jaxpme.batched_mixed.batching import next_size
from marathon.data.batching import batch_labels, batch_properties
from marathon.data.properties import DEFAULT_PROPERTIES
from marathon.data.sample import to_sample as marathon_to_sample

from lorem.longrange import get_kernel

# shared by the short-range and long-range parts: forces and stress are the
# gradients with respect to these positions and cells, and nothing else
Geometry = namedtuple(
    "Geometry",
    (
        "positions",
        "cell",
        "pbc",
        "atom_mask",
        "structure_mask",
        "atom_to_structure",
    ),
)

# short-range neighbor list at the model cutoff, indexing into Geometry's slots
Pairs = namedtuple(
    "Pairs",
    (
        "centers",
        "others",
        "cell_shifts",
        "pair_mask",
        "pair_to_structure",
    ),
)

Batch = namedtuple(
    "Batch",
    (
        "atomic_numbers",
        "geometry",
        "sr",
        "lr",  # private to the long-range kernel, see lorem.longrange
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
    lr_kernel=None,
):
    lr_kernel = get_kernel(lr_kernel)

    if batch_size is not None:
        assert batch_size > len(samples)
    else:
        batch_size = next_size(len(samples) + 1, strategy="powers_of_2")

    structures = [sample.structure for sample in samples]
    labels = [sample.labels for sample in samples]

    total_atoms = sum(len(s["positions"]) for s in structures)
    total_pairs = sum(len(s["centers"]) for s in structures)

    # +1: a padding atom and a padding pair, which padding pairs point to
    num_structures = batch_size
    if shapes is None:
        default = strategies.get("default", "powers_of_2")
        num_atoms = next_size(total_atoms + 1, strategy=default)
        num_pairs = next_size(total_pairs + 1, strategy=strategies.get("coarse", default))
    else:
        num_atoms = next_size(total_atoms + 1, strategy=shapes["atoms"])
        num_pairs = next_size(total_pairs + 1, strategy=shapes["pairs"])

    geometry, sr = _collate(structures, num_structures, num_atoms, num_pairs)

    # same atom and structure slots, so the LR inputs index the shared geometry
    lr = lr_kernel.batch(
        [s["lr"] for s in structures], num_structures, num_atoms, strategies, shapes
    )

    atomic_numbers = np.zeros(num_atoms, dtype=int)
    Z = np.concatenate([s["atomic_numbers"] for s in structures])
    atomic_numbers[: len(Z)] = Z

    labels = batch_labels(labels, num_structures, num_atoms, keys, properties=properties)
    inputs = batch_properties(
        structures,
        inputs,
        num_structures,
        num_atoms,
        float_dtype=geometry.positions.dtype,
        properties=properties,
    )

    return Batch(atomic_numbers, geometry, sr, lr, inputs, labels)


def to_sample(
    atoms,
    cutoff,
    keys=("energy", "forces"),
    inputs=(),
    lr_kernel=None,
    skin=0.0,
    properties=DEFAULT_PROPERTIES,
):
    return marathon_to_sample(
        atoms,
        cutoff,
        keys=keys,
        inputs=inputs,
        properties=properties,
        structure_fn=partial(to_structure, lr_kernel=lr_kernel, skin=skin),
    )


def to_structure(atoms, cutoff, float_dtype=None, int_dtype=None, lr_kernel=None, skin=0.0):
    from jaxpme.batched_mixed.batching import to_structure as jaxpme_structure

    # jax-pme structures are float32 regardless of the dtype used for labels and inputs
    sr = jaxpme_structure(atoms, cutoff + skin, dtype=np.float32)

    # shared geometry and the SR neighbor list; jax-pme's charges and
    # displacements are not used
    keys = (
        "positions",
        "cell",
        "pbc",
        "atomic_numbers",
        "centers",
        "others",
        "cell_shifts",
    )
    structure = {k: sr[k] for k in keys}
    structure["lr"] = get_kernel(lr_kernel).prepare(atoms, cutoff, skin=skin)

    return structure


def _collate(structures, num_structures, num_atoms, num_pairs):
    # same layout as jax-pme's batchers: structures and atoms in order, the
    # last structure and the first atom after the real ones are for padding
    padding_atom_idx = sum(len(s["positions"]) for s in structures)
    padding_structure_idx = num_structures - 1

    dtype = structures[0]["positions"].dtype

    positions = np.zeros((num_atoms, 3), dtype=dtype)
    cell = np.zeros((num_structures, 3, 3), dtype=dtype)
    cell[:] = np.eye(3)
    pbc = np.zeros((num_structures, 3), dtype=bool)
    atom_mask = np.zeros(num_atoms, dtype=bool)
    structure_mask = np.zeros(num_structures, dtype=bool)
    atom_to_structure = np.ones(num_atoms, dtype=int) * padding_structure_idx

    centers = np.ones(num_pairs, dtype=int) * padding_atom_idx
    others = np.ones(num_pairs, dtype=int) * padding_atom_idx
    cell_shifts = np.zeros((num_pairs, 3), dtype=int)
    pair_mask = np.zeros(num_pairs, dtype=bool)
    pair_to_structure = np.ones(num_pairs, dtype=int) * padding_structure_idx

    atom_offset = 0
    pair_offset = 0
    for idx, structure in enumerate(structures):
        num_n = len(structure["positions"])
        num_p = len(structure["centers"])

        atom_slice = slice(atom_offset, atom_offset + num_n)
        pair_slice = slice(pair_offset, pair_offset + num_p)

        positions[atom_slice] = structure["positions"]
        cell[idx] = structure["cell"]
        pbc[idx] = structure["pbc"]
        atom_mask[atom_slice] = True
        structure_mask[idx] = True
        atom_to_structure[atom_slice] = idx

        centers[pair_slice] = structure["centers"] + atom_offset
        others[pair_slice] = structure["others"] + atom_offset
        cell_shifts[pair_slice] = structure["cell_shifts"]
        pair_mask[pair_slice] = True
        pair_to_structure[pair_slice] = idx

        atom_offset += num_n
        pair_offset += num_p

    geometry = Geometry(
        positions=positions,
        cell=cell,
        pbc=pbc,
        atom_mask=atom_mask,
        structure_mask=structure_mask,
        atom_to_structure=atom_to_structure,
    )
    sr = Pairs(
        centers=centers,
        others=others,
        cell_shifts=cell_shifts,
        pair_mask=pair_mask,
        pair_to_structure=pair_to_structure,
    )

    return geometry, sr
