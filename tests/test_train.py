def merge_params(target, source):
    if isinstance(target, dict) and isinstance(source, dict):
        result = {}
        for k in target:
            if k in source:
                result[k] = merge_params(target[k], source[k])
            else:
                result[k] = target[k]
        return result
    return source


def test_merge_params_full_match():
    """Source fully matches target structure — all weights replaced."""
    target = {"a": 1, "b": {"c": 2, "d": 3}}
    source = {"a": 10, "b": {"c": 20, "d": 30}}
    result = merge_params(target, source)
    assert result == {"a": 10, "b": {"c": 20, "d": 30}}


def test_merge_params_partial_match():
    """Target has extra keys not in source — they keep target values."""
    target = {"a": 1, "b": 2, "new_layer": 99}
    source = {"a": 10, "b": 20}
    result = merge_params(target, source)
    assert result == {"a": 10, "b": 20, "new_layer": 99}


def test_merge_params_nested_partial():
    """Nested dict with partial overlap."""
    target = {"layer1": {"w": 1}, "layer2": {"w": 2}}
    source = {"layer1": {"w": 10}}
    result = merge_params(target, source)
    assert result == {"layer1": {"w": 10}, "layer2": {"w": 2}}


def test_merge_params_source_extra_keys_ignored():
    """Keys in source but not target are ignored (target structure wins)."""
    target = {"a": 1}
    source = {"a": 10, "extra": 99}
    result = merge_params(target, source)
    assert result == {"a": 10}
