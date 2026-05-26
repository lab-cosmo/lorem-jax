"""Convert trained e3x `Lorem` parameters to the e3j `LoremE3j` layout.

Every kernel in `LoremE3j` has a 1:1 twin in `Lorem`. Modules in both
trees are auto-numbered by Flax in execution order, so the mapping is
position-based:

- Plain scalar `Dense_N`: identical keys, copied as-is.
- Per-l spherical `_DenseE3j_N` ↔ per-l e3x `Dense_M`: matched in
  creation order. e3x stores `{l}{+|-}/kernel` sub-dicts; e3j stores
  `kernel_{l}` flat.
- `_TensorDenseE3j_N` ↔ `TensorDense_N`: matched in order; each has
  inner `dense` (per-l) and `tensor` (single kernel).
- `_TensorE3j_N` (standalone) ↔ `Tensor_N`: matched in order.
- `_MessagePassE3j_N` ↔ `MessagePass_N`: matched in order; each has
  inner `filter` (per-l Dense) and `tensor` (single kernel).
- `Initial_0` (e3x) ↔ `InitialE3j_0` (e3j): name remap, otherwise
  identical children.
- `MLP_*`, `Update_*`, `RadialCoefficients_*`: identical paths.

The CG convention difference (e3x = sqrt(2*l_out+1) * e3nn) is folded
into the forward pass of `_TensorE3j`, so parameter arrays are copied
verbatim.
"""

from flax.core import freeze, unfreeze

# ----- helpers --------------------------------------------------------------


def _walk_leaves(tree, prefix=()):
    for k, v in tree.items():
        if isinstance(v, dict):
            yield from _walk_leaves(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def _set_leaf(tree, keys, value):
    node = tree
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def _get(tree, *keys):
    node = tree
    for k in keys:
        node = node[k]
    return node


def _is_per_l_dense_node(node) -> bool:
    """A per-l Dense node has children like `0+`, `1-`, ... (each holding
    `kernel`)."""
    if not isinstance(node, dict):
        return False
    return any(
        (isinstance(k, str) and len(k) == 2 and k[1] in "+-" and k[0].isdigit())
        for k in node.keys()
    )


def _sorted_by_idx(names, prefix):
    """`['Dense_3', 'Dense_5', ...]` sorted by trailing integer."""
    return sorted(
        [n for n in names if n.startswith(prefix)],
        key=lambda n: int(n[len(prefix) :]),
    )


def _copy_per_l_dense(e3j_node, e3x_node):
    """Copy `{l}{+|-}/kernel` entries from e3x into `kernel_{l}` in e3j."""
    kernel_keys = [k for k in e3j_node.keys() if k.startswith("kernel_")]
    local_max = max(int(k.split("_")[1]) for k in kernel_keys)
    for l in range(local_max + 1):
        parity = "+" if l % 2 == 0 else "-"
        e3j_node[f"kernel_{l}"] = e3x_node[f"{l}{parity}"]["kernel"]


# ----- main API -------------------------------------------------------------


def convert_e3x_params_to_e3j(e3x_params, e3j_params_template):
    """Build an e3j params pytree populated from an e3x checkpoint.

    Args:
        e3x_params: pytree returned by `Lorem.init` / loaded from a
            checkpoint.
        e3j_params_template: pytree returned by `LoremE3j.init`. Only the
            structure is used; values are overwritten.

    Returns:
        New pytree with the same structure as `e3j_params_template`,
        populated from `e3x_params`.

    Raises:
        RuntimeError: if module counts don't match or a required slot is
            missing.
    """
    e3x = unfreeze(e3x_params)["params"]
    out = unfreeze(e3j_params_template)["params"]
    filled: set[tuple] = set()

    # ---- 1. Identical-name modules: MLP_*, Update_*, RadialCoefficients_*,
    #         Initial_0 -> InitialE3j_0. Dense_* handled in step 2.
    for keys, _ in _walk_leaves(out):
        first = keys[0]
        if first == "InitialE3j_0":
            src_keys = ("Initial_0",) + keys[1:]
            _set_leaf(out, keys, _get(e3x, *src_keys))
            filled.add(keys)
            continue
        if first.startswith(("MLP_", "Update_", "RadialCoefficients_")):
            _set_leaf(out, keys, _get(e3x, *keys))
            filled.add(keys)
            continue

    # ---- 2. Match Dense modules by class and position.
    # e3j has SCALAR Dense_N (auto-numbered) and PER-L _DenseE3j_N (auto-numbered).
    # e3x has BOTH scalar and per-l interleaved under `Dense_N` — distinguished by
    # whether the node has per-l children (`{l}{+|-}/kernel`).
    e3j_scalar_dense = _sorted_by_idx(out.keys(), "Dense_")
    e3j_per_l_dense = _sorted_by_idx(out.keys(), "_DenseE3j_")
    e3x_all_dense = _sorted_by_idx(e3x.keys(), "Dense_")
    e3x_scalar_dense = [k for k in e3x_all_dense if not _is_per_l_dense_node(e3x[k])]
    e3x_per_l_dense = [k for k in e3x_all_dense if _is_per_l_dense_node(e3x[k])]

    if len(e3j_scalar_dense) != len(e3x_scalar_dense):
        raise RuntimeError(
            f"Scalar Dense count mismatch: e3j={e3j_scalar_dense} "
            f"({len(e3j_scalar_dense)}) vs e3x={e3x_scalar_dense} "
            f"({len(e3x_scalar_dense)})"
        )
    if len(e3j_per_l_dense) != len(e3x_per_l_dense):
        raise RuntimeError(
            f"Per-l Dense count mismatch: e3j={e3j_per_l_dense} "
            f"({len(e3j_per_l_dense)}) vs e3x={e3x_per_l_dense} "
            f"({len(e3x_per_l_dense)})"
        )

    for ej, ex in zip(e3j_scalar_dense, e3x_scalar_dense):
        for sub_k, sub_v in e3x[ex].items():
            out[ej][sub_k] = sub_v
            filled.add((ej, sub_k))

    for ej, ex in zip(e3j_per_l_dense, e3x_per_l_dense):
        _copy_per_l_dense(out[ej], e3x[ex])
        for k in out[ej].keys():
            filled.add((ej, k))

    # ---- 3. `_TensorDenseE3j_N` ↔ `TensorDense_N`.
    e3j_td = _sorted_by_idx(out.keys(), "_TensorDenseE3j_")
    e3x_td = _sorted_by_idx(e3x.keys(), "TensorDense_")
    if len(e3j_td) != len(e3x_td):
        raise RuntimeError(f"TensorDense count mismatch: e3j={e3j_td}, e3x={e3x_td}")
    for ej, ex in zip(e3j_td, e3x_td):
        _copy_per_l_dense(out[ej]["dense"], e3x[ex]["dense"])
        out[ej]["tensor"]["kernel"] = e3x[ex]["tensor"]["kernel"]
        for k in out[ej]["dense"].keys():
            filled.add((ej, "dense", k))
        filled.add((ej, "tensor", "kernel"))

    # ---- 4. `_TensorE3j_N` (standalone) ↔ `Tensor_N`.
    e3j_t = _sorted_by_idx(out.keys(), "_TensorE3j_")
    e3x_t = _sorted_by_idx(e3x.keys(), "Tensor_")
    if len(e3j_t) != len(e3x_t):
        raise RuntimeError(f"Tensor count mismatch: e3j={e3j_t}, e3x={e3x_t}")
    for ej, ex in zip(e3j_t, e3x_t):
        out[ej]["kernel"] = e3x[ex]["kernel"]
        filled.add((ej, "kernel"))

    # ---- 5. `_MessagePassE3j_N` ↔ `MessagePass_N`.
    e3j_mp = _sorted_by_idx(out.keys(), "_MessagePassE3j_")
    e3x_mp = _sorted_by_idx(e3x.keys(), "MessagePass_")
    if len(e3j_mp) != len(e3x_mp):
        raise RuntimeError(f"MessagePass count mismatch: e3j={e3j_mp}, e3x={e3x_mp}")
    for ej, ex in zip(e3j_mp, e3x_mp):
        _copy_per_l_dense(out[ej]["filter"], e3x[ex]["filter"])
        out[ej]["tensor"]["kernel"] = e3x[ex]["tensor"]["kernel"]
        for k in out[ej]["filter"].keys():
            filled.add((ej, "filter", k))
        filled.add((ej, "tensor", "kernel"))

    # ---- sanity: every leaf in target must have been filled.
    expected = {k for k, _ in _walk_leaves(out)}
    missing = expected - filled
    if missing:
        raise RuntimeError(
            "Converter left these e3j parameters unset:\n  "
            + "\n  ".join("/".join(m) for m in sorted(missing))
        )

    return freeze({"params": out})
