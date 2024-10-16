import pytest
import time


def simple_fixture(name, params):
    @pytest.fixture(name=name, params=params)
    def inner(request):
        return request.param
    return inner

import pytest
pytest.simple_fixture = simple_fixture

N     = pytest.simple_fixture("N", params=(2,4,16,32))
seed  = pytest.simple_fixture("seed", params=(0, time.time()))
dtype = pytest.simple_fixture("dtype", params=("float32", "float64", "complex128"))



N     = pytest.simple_fixture("N", params=(2,4, 6))
seed  = pytest.simple_fixture("seed", params=(0, ))
dtype = pytest.simple_fixture("dtype", params=("float32",))
