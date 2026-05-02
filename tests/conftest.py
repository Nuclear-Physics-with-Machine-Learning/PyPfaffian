import pytest


def simple_fixture(name, params):
    @pytest.fixture(name=name, params=params)
    def inner(request):
        return request.param

    return inner


pytest.simple_fixture = simple_fixture

N = pytest.simple_fixture("N", params=(2, 4, 6, 12, 16, 20))
seed = pytest.simple_fixture("seed", params=(0, 42))
