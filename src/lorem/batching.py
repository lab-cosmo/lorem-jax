import numpy as np

from collections import namedtuple

from jaxpme.batched_mixed.batching import get_batch as jaxpme_batcher
from jaxpme.batched_mixed.batching import prepare as jaxpme_prepare
from marathon.data.batching import batch_samples as marathon_batch
from marathon.data.properties import DEFAULT_PROPERTIES
from marathon.data.sample import Sample as MarathonSample
from marathon.data.sample import to_labels, to_structure
from marathon.utils import next_size

# Short-range (message-passing) neighbor list. Built at the model cutoff via
# marathon/vesin -- a plain flat neighbor list with no k-grid. It carries only
# connectivity; positions/cell come from the shared Ewald ``realspace`` batch,
# so forces flow through a single set of positions.
MLIPPairs = namedtuple(
    "MLIPPairs",
    ("centers", "others", "cell_shifts", "pair_mask", "pair_to_structure"),
)

Batch = namedtuple(
    "Batch",
    (
        "atomic_numbers",
        "mlip",
        "realspace",
        "nopbc",
        "pbc",
        "labels",
    ),
)


Sample = namedtuple(
    "Sample",
    (
        "structure",
        "ewald_structure",
        "labels",
    ),
    # structure       = short-range neighbor list (marathon, model cutoff).
    # ewald_structure = Ewald neighbor list (jax-pme, num_k / derived cutoff),
    #                   decoupled from the short-range cutoff. Provides the
    #                   shared positions, the long-range k-space and the
    #                   real-space Ewald pairs.
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

    sr_structures = [s.structure for s in samples]
    ewald_structures = [s.ewald_structure for s in samples]

    # 1. Batch the Ewald structures (jax-pme). This fixes the atom ordering and
    #    padding, and provides the shared positions/cell, the long-range k-space
    #    (pbc), the non-periodic all-pairs list (nopbc) and the real-space Ewald
    #    pairs used by the long-range sum.
    if shapes is None:
        default = strategies.get("default", "powers_of_2")
        coarse = strategies.get("coarse", default)
        _, realspace, nopbc, pbc = jaxpme_batcher(
            ewald_structures,
            strategy=default,
            num_structures_pbc=strategies.get("fine", default),
            num_pairs_nonpbc=coarse,
            num_pairs=coarse,
            num_structures=batch_size,
        )
    else:
        _, realspace, nopbc, pbc = jaxpme_batcher(
            ewald_structures,
            num_structures=batch_size,
            num_structures_pbc=shapes["pbc"],
            num_atoms=shapes["atoms"],
            num_atoms_pbc=shapes["atoms_pbc"],
            num_pairs=shapes["ewald_pairs"],
            num_pairs_nonpbc=shapes["pairs_nonpbc"],
            num_k=shapes["k"],
            strategy="multiples",
        )

    num_structures = realspace.cell.shape[0]
    num_atoms = realspace.positions.shape[0]

    # 2. Batch the short-range connectivity (marathon flat batcher) with the
    #    SAME num_structures/num_atoms so the SR pairs index into the shared
    #    positions. marathon batches labels too, consistently with the shapes.
    if shapes is None:
        total_sr_pairs = sum(len(s["centers"]) for s in sr_structures)
        num_sr_pairs = next_size(total_sr_pairs + 1, strategy=coarse)
    else:
        num_sr_pairs = shapes["pairs"]

    marathon_samples = [MarathonSample(s.structure, s.labels) for s in samples]
    mlip_batch = marathon_batch(
        marathon_samples,
        num_atoms,
        num_sr_pairs,
        keys,
        num_structures=num_structures,
        properties=properties,
    )

    # marathon's flat batch stores pre-computed displacements, not cell shifts,
    # so re-collate the raw cell shifts into the (front-contiguous) real pairs.
    raw_cell_shifts = np.concatenate([s["cell_shifts"] for s in sr_structures])
    cell_shifts = np.zeros((mlip_batch.centers.shape[0], 3), dtype=raw_cell_shifts.dtype)
    cell_shifts[mlip_batch.pair_mask] = raw_cell_shifts

    mlip = MLIPPairs(
        centers=mlip_batch.centers,
        others=mlip_batch.others,
        cell_shifts=cell_shifts,
        pair_mask=mlip_batch.pair_mask,
        pair_to_structure=mlip_batch.pair_to_structure,
    )

    atomic_numbers = np.zeros(num_atoms, dtype=int)
    Z = np.concatenate([s["atomic_numbers"] for s in sr_structures])
    atomic_numbers[: len(Z)] = Z

    return Batch(atomic_numbers, mlip, realspace, nopbc, pbc, mlip_batch.labels)


def to_sample(
    atoms,
    cutoff,
    keys=None,
    energy=True,
    forces=True,
    stress=False,
    num_k=None,
    cutoff_ewald=None,
    lr_wavelength=None,
    smearing=None,
    properties=DEFAULT_PROPERTIES,
):
    """Build a sample with decoupled short-range and Ewald neighbor lists.

    ``cutoff`` sets the short-range/message-passing neighbor list (built via
    marathon/vesin, no k-grid). The Ewald real-space list is built independently
    from ``num_k`` (fixed k-grid) or, if that is not given, from ``cutoff_ewald``
    (falling back to ``cutoff``). When ``num_k`` is set the Ewald cutoff is
    derived by jax-pme and decoupled from the short-range cutoff.
    """
    # Short-range neighbor list at the model cutoff (marathon, no k-grid).
    structure = to_structure(atoms, cutoff, float_dtype=np.float32)

    # Ewald neighbor list: a fixed k-grid (num_k) decouples its cutoff from the
    # short-range cutoff. num_k only makes sense for periodic systems; for
    # non-periodic ones fall back to an explicit Ewald cutoff (the long-range
    # part is an all-pairs sum there anyway).
    ewald_num_k = num_k if atoms.pbc.any() else None
    ewald_cutoff = cutoff_ewald if cutoff_ewald is not None else cutoff
    ewald_structure = jaxpme_prepare(
        atoms,
        cutoff=None if ewald_num_k is not None else ewald_cutoff,
        num_k=ewald_num_k,
        lr_wavelength=lr_wavelength,
        smearing=smearing,
        dtype=np.float32,
    )

    labels = to_labels(
        atoms,
        keys=keys,
        energy=energy,
        forces=forces,
        stress=stress,
        properties=properties,
    )

    return Sample(structure, ewald_structure, labels)
