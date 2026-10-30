import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.build import bulk, fcc111, molecule
from grain.python import Record, RecordMetadata

from lorem.batching import needed_shapes, sample_shapes, to_batch, to_sample
from lorem.calculator import Calculator
from lorem.longrange import Ewald, TiledEwald, potentials
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


def make_cells(*repeats):
    """Cubic NaCl supercells: same shape, growing size."""
    return [bulk("NaCl", "rocksalt", a=5.64, cubic=True) * n for n in repeats]


def make_batch(lr_kernel, structures=None):
    structures = make_structures() if structures is None else structures
    samples = [to_sample(a, CUTOFF, keys=(), lr_kernel=lr_kernel) for a in structures]
    return to_batch(samples, [], lr_kernel=lr_kernel)


def make_records(lr_kernel, structures=None):
    structures = make_structures() if structures is None else structures
    to_sample_t = ToSample(cutoff=CUTOFF, keys=(), lr_kernel=lr_kernel)
    return [
        Record(RecordMetadata(index=i, record_key=i), to_sample_t.map(atoms))
        for i, atoms in enumerate(structures)
    ]


def run_batcher(records, **kwargs):
    batcher = ToBatch(batch_size=4, keys=(), drop_remainder=False, **kwargs)
    return [record.data for record in batcher(iter(records))], batcher


def count_setups(calc):
    """Record every neighbor list rebuild of a calculator."""
    calls = []
    setup = calc.setup

    def counted(atoms):
        calls.append(1)
        setup(atoms)

    calc.setup = counted
    return calls


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

    # jax-pme sizing its own batch arrives at the same shapes
    _, sr, nopbc, pbc = get_batch(
        [prepare(a, CUTOFF, dtype=np.float32) for a in structures],
        num_structures=batch.geometry.cell.shape[0],
    )
    np.testing.assert_array_equal(sr.positions, batch.geometry.positions)
    np.testing.assert_array_equal(sr.atom_to_structure, batch.geometry.atom_to_structure)
    assert pbc.k_grid.shape == batch.lr.pbc.k_grid.shape
    assert pbc.atom_mask.shape == batch.lr.pbc.atom_mask.shape
    assert nopbc.pair_mask.shape == batch.lr.nopbc.pair_mask.shape

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

    q = random_charges(reference)
    assert_close(
        potentials(q, batch.geometry, batch.lr),
        potentials(q, reference.geometry, reference.lr),
    )

    results = model.predict(params, batch, stress=True)
    for key in ("energy", "forces", "stress"):
        assert_close(results[key], reference_results[key])


def test_num_k_fixes_k_while_cutoff_grows():
    """With num_k, the k-grid size stays put and the real-space reach grows."""
    lr_kernel = TiledEwald(num_k=1000)
    shapes = [
        lr_kernel.sample_shapes(r.data.structure["lr"])
        for r in make_records(lr_kernel, make_cells(1, 2, 3))
    ]
    atoms = [8, 64, 216]

    assert len({s["k"] for s in shapes}) == 1
    pairs_per_atom = [s["lr_pairs"] / n for s, n in zip(shapes, atoms)]
    assert pairs_per_atom == sorted(pairs_per_atom)
    assert pairs_per_atom[-1] > 10 * pairs_per_atom[0]


def test_ewald_takes_num_k_or_cutoff():
    """With num_k, the smearing follows it; a cutoff on top would not converge."""
    with pytest.raises(ValueError, match="num_k or cutoff"):
        Ewald(num_k=200, cutoff=8.0)


def test_layouts_must_agree():
    """LR inputs collated for other atom slots than the geometry are refused."""
    lr_kernel = Ewald()
    structures = make_structures()
    batch = make_batch(lr_kernel, structures)
    samples = [to_sample(a, CUTOFF, keys=(), lr_kernel=lr_kernel) for a in structures]
    shapes = needed_shapes([sample_shapes(s.structure) for s in samples])
    shapes = {**shapes, "atoms": batch.geometry.positions.shape[0]}

    swapped = [samples[1].structure["lr"], samples[0].structure["lr"]]
    swapped.append(samples[2].structure["lr"])
    with pytest.raises(AssertionError, match="layouts disagree"):
        lr_kernel.batch(swapped, batch.geometry, shapes)


@pytest.mark.parametrize("lr_kernel", KERNELS, ids=str)
def test_grain_transforms_with_kernel(lr_kernel):
    """ToSample and ToBatch carry the kernel; info() reports its sizes."""
    (batch,), batcher = run_batcher(make_records(lr_kernel), lr_kernel=lr_kernel)
    assert batcher.count_samples(batch) == 3

    real, total, shape = jax.jit(batcher.info)(batch)
    assert real["samples"] == 3
    assert real["pbc"] == 2
    assert 0 < real["k"] <= total["k"]
    assert 0 < real["lr_pairs"] <= total["lr_pairs"]
    assert real.keys() == total.keys() == shape.keys()


def test_kernel_size_strategies():
    """With num_k, k follows the tile grid and LR pairs their own strategy."""
    lr_kernel = TiledEwald(num_k=1000)
    records = make_records(lr_kernel, make_cells(1, 2))
    k = lr_kernel.sample_shapes(records[0].data.structure["lr"])["k"]

    (batch,), batcher = run_batcher(records, lr_kernel=lr_kernel)
    _, _, shape = batcher.info(batch)
    assert shape["k"] == -(-k // lr_kernel.BK) * lr_kernel.BK
    assert shape["atoms_pbc"] % lr_kernel.BM == 0
    # SR pairs follow coarse_strategy (powers of 8), LR pairs do not
    assert np.log(shape["pairs"]) / np.log(8) % 1 == 0
    assert np.log(shape["lr_pairs"]) / np.log(8) % 1 != 0

    # explicit strategies win over the kernel's
    (batch,), batcher = run_batcher(
        records,
        lr_kernel=lr_kernel,
        k_strategy="powers_of_2",
        lr_pairs_strategy="multiples_of_1000",
    )
    _, _, shape = batcher.info(batch)
    assert shape["k"] == 2 ** int(np.ceil(np.log2(k)))
    assert shape["lr_pairs"] % 1000 == 0


@pytest.mark.parametrize("lr_kernel", [Ewald(num_k=200), TiledEwald(num_k=200)], ids=str)
def test_fixed_shapes(lr_kernel):
    """With shapes, every batch has exactly those; too-large samples raise."""
    structures = make_structures()
    records = make_records(lr_kernel, structures * 3)

    # room for one of each structure per batch
    shapes = needed_shapes(
        [sample_shapes(r.data.structure, lr_kernel) for r in records[:3]], lr_kernel
    )
    shapes = {axis: lr_kernel.round(axis, size) for axis, size in shapes.items()}

    batches, batcher = run_batcher(records, lr_kernel=lr_kernel, shapes=shapes)
    assert len(batches) == 3
    assert [batcher.count_samples(b) for b in batches] == [3, 3, 3]
    layouts = [jax.tree.map(np.shape, b) for b in batches]
    assert layouts[0] == layouts[1] == layouts[2]

    _, _, shape = batcher.info(batches[0])
    for axis in ("atoms", "pairs", "lr_pairs", "pairs_nonpbc", "pbc", "k"):
        assert shape[axis] == shapes[axis]

    too_small = {**shapes, "atoms": len(structures[0]) - 1}
    with pytest.raises(ValueError, match="atoms"):
        run_batcher(records, lr_kernel=lr_kernel, shapes=too_small)

    with pytest.raises(ValueError, match="missing"):
        ToBatch(batch_size=4, lr_kernel=lr_kernel, shapes={"atoms": 64})


def test_fixed_shapes_follow_tiles():
    """Tiled shapes off the tile grid would be padded further: refuse them."""
    shapes = dict.fromkeys(["atoms", "pairs", "lr_pairs", "pairs_nonpbc", "pbc"], 64)
    with pytest.raises(ValueError, match="multiple of 128"):
        ToBatch(
            batch_size=4,
            lr_kernel=TiledEwald(num_k=200),
            shapes={**shapes, "atoms_pbc": 64, "k": 200},
        )


# cap, axis, kernel, structures: two samples that both add to the axis
BUDGETS = [
    ("max_total_atoms", "atoms", Ewald(num_k=200), [0, 1]),
    ("max_total_pairs", "pairs", Ewald(num_k=200), [0, 1]),
    ("max_total_lr_pairs", "lr_pairs", Ewald(num_k=200), [0, 1]),
    ("max_total_nobpc_pairs", "pairs_nonpbc", Ewald(num_k=200), [2, 2]),
    ("max_total_k", "k", Ewald(num_k=200), [0, 1]),
    ("max_total_k", "k", TiledEwald(num_k=200), [0, 1]),
    ("max_total_atoms_pbc", "atoms_pbc", Ewald(num_k=200), [0, 1]),
    ("max_total_atoms_pbc", "atoms_pbc", TiledEwald(num_k=200), [0, 1]),
]


@pytest.mark.parametrize("cap, axis, lr_kernel, indices", BUDGETS, ids=str)
def test_budgets(cap, axis, lr_kernel, indices):
    """A cap admits samples while they fit and splits the batch when not."""
    structures = make_structures()
    records = make_records(lr_kernel, [structures[i] for i in indices])
    unbounded = ToBatch(batch_size=4, lr_kernel=lr_kernel)

    def size(records):
        shapes = [sample_shapes(r.data.structure, lr_kernel) for r in records]
        if axis in ("k", "atoms_pbc"):  # capped as padded arrays
            return lr_kernel.array_sizes(unbounded.batch_shapes(shapes))[axis]
        return needed_shapes(shapes, lr_kernel)[axis]

    both = size(records)
    single = max(size(records[:1]), size(records[1:]))
    assert single < both

    assert len(run_batcher(records, lr_kernel=lr_kernel, **{cap: both})[0]) == 1
    assert len(run_batcher(records, lr_kernel=lr_kernel, **{cap: both - 1})[0]) == 2
    with pytest.raises(ValueError, match=axis):
        run_batcher(records, lr_kernel=lr_kernel, **{cap: single - 1})


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


def test_calculator_kernel_from_checkpoint(tmp_path):
    """A checkpoint brings the kernel it was trained with, if it recorded one."""
    from marathon.io import write_yaml

    from lorem.calculator import _checkpoint_lr_kernel

    assert _checkpoint_lr_kernel(tmp_path) is None

    write_yaml(tmp_path / "config.yaml", {"training_pipeline": {"lr_kernel": None}})
    assert _checkpoint_lr_kernel(tmp_path) is None

    spec = {"TiledEwald": {"num_k": 300}}
    write_yaml(tmp_path / "config.yaml", {"training_pipeline": {"lr_kernel": spec}})
    assert _checkpoint_lr_kernel(tmp_path) == TiledEwald(num_k=300)


@pytest.mark.parametrize("index", [0, 1], ids=["bulk", "slab"])
def test_calculator_reuses_lists_within_skin(index, reference):
    """Small moves reuse both cached lists; results match a fresh calculator."""
    model, params, _, _ = reference
    lr_kernel = Ewald(num_k=200)
    atoms = make_structures()[index]

    calc = Calculator.from_model(model, params=params, lr_kernel=lr_kernel, skin=0.5)
    setups = count_setups(calc)
    calc.calculate(atoms)

    displaced = atoms.copy()
    displaced.rattle(0.05, seed=4)
    calc.calculate(displaced)
    assert len(setups) == 1

    fresh = Calculator.from_model(model, params=params, lr_kernel=lr_kernel, skin=0.5)
    fresh.calculate(displaced)
    assert_close(calc.results["energy"], fresh.results["energy"], rtol=1e-5)
    assert_close(calc.results["forces"], fresh.results["forces"], rtol=1e-5)


def test_calculator_rebuilds_for_ewald_cell_shifts(reference):
    """A strain within the SR skin budget but not the Ewald one rebuilds.

    The Ewald list reaches further, so its pairs span more cell images, and a
    cell change moves them more than it moves the SR pairs.
    """
    model, params, _, _ = reference
    lr_kernel = Ewald(num_k=200)
    atoms = make_structures()[0]
    skin = 0.5

    calc = Calculator.from_model(model, params=params, lr_kernel=lr_kernel, skin=skin)
    setups = count_setups(calc)
    calc.calculate(atoms)

    sr_shift = int(np.abs(np.array(calc.batch.sr.cell_shifts)).max())
    lr_shift = int(np.abs(np.array(calc.batch.lr.cell_shifts)).max())
    assert lr_shift > sr_shift

    # change the cell vectors by `change` in total, without moving the atoms
    change = skin / np.sqrt(sr_shift * lr_shift)
    assert sr_shift * change < skin < lr_shift * change
    strained = atoms.copy()
    lengths = np.linalg.norm(atoms.cell[:], axis=1)
    strained.set_cell(atoms.cell[:] * (1 + change / lengths.sum()), scale_atoms=False)

    calc.calculate(strained)
    assert len(setups) == 2

    fresh = Calculator.from_model(model, params=params, lr_kernel=lr_kernel, skin=skin)
    fresh.calculate(strained)
    assert_close(calc.results["energy"], fresh.results["energy"], rtol=1e-5)
    assert_close(calc.results["forces"], fresh.results["forces"], rtol=1e-5)
