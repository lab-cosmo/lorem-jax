import lorem


def test_version():
    assert hasattr(lorem, "__version__")
    assert isinstance(lorem.__version__, str)
