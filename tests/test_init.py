import loremjax


def test_version():
    assert hasattr(loremjax, "__version__")
    assert isinstance(loremjax.__version__, str)
