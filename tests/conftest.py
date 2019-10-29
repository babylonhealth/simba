import pytest
import numpy as np


@pytest.fixture(autouse=True)
def random_seed():
    np.random.seed(42)
