import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.build import bulk, fcc111, molecule
from grain.python import Record, RecordMetadata

from lorem.batching import to_batch, to_sample
from lorem.calculator import Calculator
from lorem.longrange import (
    Ewald,
    EwaldInputs,
    TiledEwald,
    TiledEwaldInputs,
    potentials,
)
from lorem.models.mlip import Lorem
from lorem.transforms import ToBatch, ToSample

CUTOFF = 5.0

# all converged well beyond float32 noise, with different real-space cutoffs
KERNELS = [
    Ewald(cutoff=8.0),
    Ewald(num_k=200),
    TiledEwald(num_k=200),
]


def make_structures():
    """3D bulk, 2D slab and a molecule: every branch of the Ewald backends."""
    bulk_ = bulk("NaCl", "rocksalt", a=5.64) * [2, 2, 2]
    bulk_.rattle(0.05, seed=1)

    slab = fcc111("Al", size=(3, 3, 2), vacuum=8.0)
    slab.pbc = [True, True, False]
    slab.rattle(0.05, seed=2)

    mol = molecule("CH3CH2OH")
    mol.rattle(0.05, seed=3)

    return [bulk_, slab, mol]


def make_batch(lr_kernel, structures=None):
    structures = make_structures() if structures is None else structures
    samples = [to_sample(a, CUTOFF, keys=(), lr_kernel=lr_kernel) for a in structures]
    return to_batch(samples, [], lr_kernel=lr_kernel)


def make_model():
    model = Lorem(
        cutoff=CUTOFF,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        num_species=20,
    )
    params = model.init(jax.random.key(0), *model.dummy_inputs())
    return model, params


@pytest.fixture(scope="module")
def reference():
    """Model, params, and default-kernel batch and predictions, built once."""
    model, params = make_model()
    batch = jax.tree.map(jnp.array, make_batch(None))
    results = model.predict(params, batch, stress=True)
    return model, params, batch, results


def random_charges(batch, seed=0):
    q = np.random.default_rng(seed).normal(size=batch.geometry.positions.shape[0])
    return jnp.array(q * batch.geometry.atom_mask, dtype=jnp.float32)


def assert_close(actual, desired, rtol=1e-4):
    """Agreement relative to the largest entry: kernels differ by convergence."""
    actual, desired = np.asarray(actual), np.asarray(desired)
    np.testing.assert_allclose(actual, desired, atol=rtol * np.abs(desired).max())


def test_default_kernel_matches_jaxpme():
    """The default kernel is jax-pme's batched Ewald at the model cutoff."""
    from jaxpme.batched_mixed import Ewald as JaxpmeEwald
    from jaxpme.batched_mixed.batching import get_batch, prepare

    structures = make_structures()
    batch = make_batch(None, structures)
    assert isinstance(batch.lr, EwaldInputs)

    _, sr, nopbc, pbc = get_batch(
        [prepare(a, CUTOFF, dtype=np.float32) for a in structures],
        num_structures=batch.geometry.cell.shape[0],
        num_atoms=batch.geometry.positions.shape[0],
    )
    np.testing.assert_array_equal(sr.positions, batch.geometry.positions)
    np.testing.assert_array_equal(sr.atom_to_structure, batch.geometry.atom_to_structure)

    q = random_charges(batch)
    reference = JaxpmeEwald().potentials(q, *jax.tree.map(jnp.array, (sr, nopbc, pbc)))
    actual = potentials(q, *jax.tree.map(jnp.array, (batch.geometry, batch.lr)))

    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)


def test_lr_cutoff_decoupled_from_sr():
    """num_k sets the Ewald real-space cutoff; SR pairs and geometry stay put."""
    default = make_batch(None)
    fixed_k = make_batch(Ewald(num_k=200))

    jax.tree.map(np.testing.assert_array_equal, default.geometry, fixed_k.geometry)
    jax.tree.map(np.testing.assert_array_equal, default.sr, fixed_k.sr)

    # with ~200 k-vectors, real space has to reach much further than 5 Å
    assert fixed_k.lr.pair_mask.sum() > 2 * default.lr.pair_mask.sum()
    assert fixed_k.lr.pbc.k_grid.shape[1] < default.lr.pbc.k_grid.shape[1]


def test_nonperiodic_has_no_realspace_pairs():
    """Non-periodic structures use the all-pairs 1/r sum, not a cutoff list."""
    batch = make_batch(None, [molecule("CH3CH2OH")])
    assert batch.sr.pair_mask.sum() > 0
    assert batch.lr.pair_mask.sum() == 0
    assert batch.lr.nopbc.pair_mask.sum() == 9 * 8 / 2


@pytest.mark.parametrize("lr_kernel", KERNELS, ids=str)
def test_kernels_agree(lr_kernel, reference):
    """Same params, any kernel: potentials, energy, forces, stress agree."""
    model, params, reference, reference_results = reference
    batch = jax.tree.map(jnp.array, make_batch(lr_kernel))

    expected = {TiledEwald: TiledEwaldInputs, Ewald: EwaldInputs}[type(lr_kernel)]
    assert isinstance(batch.lr, expected)

    q = random_charges(reference)
    assert_close(
        potentials(q, batch.geometry, batch.lr),
        potentials(q, reference.geometry, reference.lr),
    )

    results = model.predict(params, batch, stress=True)
    for key in ("energy", "forces", "stress"):
        assert_close(results[key], reference_results[key])


@pytest.mark.parametrize("lr_kernel", KERNELS, ids=str)
def test_grain_transforms_with_kernel(lr_kernel):
    """ToSample and ToBatch carry the kernel; info() reports its sizes."""
    to_sample_t = ToSample(cutoff=CUTOFF, keys=(), lr_kernel=lr_kernel)
    records = [
        Record(RecordMetadata(index=i, record_key=i), to_sample_t.map(atoms))
        for i, atoms in enumerate(make_structures())
    ]
    batcher = ToBatch(batch_size=4, keys=(), drop_remainder=False, lr_kernel=lr_kernel)
    batch = next(iter(batcher(iter(records)))).data

    assert isinstance(batch.lr, (EwaldInputs, TiledEwaldInputs))
    assert batcher.count_samples(batch) == 3

    real, total, shape = jax.jit(batcher.info)(batch)
    assert real["samples"] == 3
    assert real["pbc"] == 2
    assert 0 < real["k"] <= total["k"]
    assert 0 < real["lr_pairs"] <= total["lr_pairs"]
    assert real.keys() == total.keys() == shape.keys()


def test_tobatch_budgets_kernel_k():
    """max_total_k is checked against the kernel's k-grid, not the model's."""
    structures = make_structures()[:2]  # both periodic
    lr_kernel = Ewald(num_k=200)
    to_sample_t = ToSample(cutoff=CUTOFF, keys=(), lr_kernel=lr_kernel)
    records = [
        Record(RecordMetadata(index=i, record_key=i), to_sample_t.map(atoms))
        for i, atoms in enumerate(structures)
    ]
    k = [lr_kernel.sample_shapes(r.data.structure["lr"])["k"] for r in records]

    # room for one structure's k-grid, not for two
    batcher = ToBatch(
        batch_size=4,
        keys=(),
        drop_remainder=False,
        lr_kernel=lr_kernel,
        max_total_k=2 * max(k) - 1,
    )
    batches = list(batcher(iter(records)))
    assert len(batches) == 2


def test_tobatch_budgets_lr_pairs():
    """With num_k, Ewald pairs can outgrow SR pairs; they have their own budget."""
    lr_kernel = Ewald(num_k=200)
    to_sample_t = ToSample(cutoff=CUTOFF, keys=(), lr_kernel=lr_kernel)
    records = [
        Record(RecordMetadata(index=i, record_key=i), to_sample_t.map(atoms))
        for i, atoms in enumerate(make_structures())
    ]
    lr_pairs = [len(r.data.structure["lr"]["centers"]) for r in records]
    pairs = [len(r.data.structure["centers"]) for r in records]
    assert sum(lr_pairs) > sum(pairs)

    def num_batches(**budget):
        batcher = ToBatch(
            batch_size=4, keys=(), drop_remainder=False, lr_kernel=lr_kernel, **budget
        )
        return len(list(batcher(iter(records))))

    assert num_batches(max_total_pairs=sum(pairs)) == 1
    assert num_batches(max_total_lr_pairs=sum(lr_pairs)) == 1
    assert num_batches(max_total_lr_pairs=sum(lr_pairs) - 1) == 2

    with pytest.raises(ValueError, match="max lr_pairs"):
        num_batches(max_total_lr_pairs=max(lr_pairs) - 1)


def test_calculator_kernel(reference):
    """The calculator takes the kernel too; results match the default."""
    model, params, _, _ = reference
    atoms = make_structures()[1]  # slab: the k-space cell differs from the cell

    calc = Calculator.from_model(model, params=params, lr_kernel=TiledEwald(num_k=200))
    default = Calculator.from_model(model, params=params)
    calc.calculate(atoms)
    default.calculate(atoms)

    assert_close(calc.results["energy"], default.results["energy"])
    assert_close(calc.results["forces"], default.results["forces"])


@pytest.mark.parametrize("index", [0, 1], ids=["bulk", "slab"])
def test_calculator_skin_with_decoupled_lists(index, reference):
    """Within the skin, the cached SR and Ewald lists both stay valid."""
    model, params, _, _ = reference
    lr_kernel = Ewald(num_k=200)  # Ewald list reaches much further than SR
    atoms = make_structures()[index]

    calc = Calculator.from_model(model, params=params, lr_kernel=lr_kernel, skin=0.5)
    calc.calculate(atoms)
    lr_shift = int(np.abs(np.array(calc.batch.lr.cell_shifts)).max())
    sr_shift = int(np.abs(np.array(calc.batch.sr.cell_shifts)).max())
    assert calc._nl_cache._max_cell_shift == max(lr_shift, sr_shift)

    displaced = atoms.copy()
    displaced.rattle(0.05, seed=4)
    assert calc._nl_cache.needs_update(displaced) is False
    calc.calculate(displaced)

    fresh = Calculator.from_model(model, params=params, lr_kernel=lr_kernel, skin=0.5)
    fresh.calculate(displaced)

    np.testing.assert_allclose(
        calc.results["energy"], fresh.results["energy"], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        calc.results["forces"], fresh.results["forces"], rtol=1e-5, atol=1e-5
    )
