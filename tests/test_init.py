import pytest

comms = pytest.importorskip("comms")


def test_version():
    import lorem

    assert hasattr(lorem, "__version__")
    assert isinstance(lorem.__version__, str)


def test_exports():
    import lorem

    assert hasattr(lorem, "Lorem")
    assert hasattr(lorem, "LoremBEC")
    assert hasattr(lorem, "LOREMCalculator")
    assert hasattr(lorem, "comms")
