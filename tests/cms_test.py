def test_import():
    from sovon_cms import main
    print('version:', main.__version__)
    assert 1<2