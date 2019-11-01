import pytest
import numpy as np

INPUTS_ = [
    np.eye(3),
    np.eye(3, 2),
    np.zeros((3, 3)),
    np.zeros((3, 2)),
    np.ones((3, 2)),
    np.ones((3, 3)),
    np.matrix([np.linspace(1, 5, 5), np.linspace(2, -4, 5), np.linspace(-10, -15, 5)])
]

@pytest.fixture(autouse=True)
def random_seed():
    np.random.seed(42)
