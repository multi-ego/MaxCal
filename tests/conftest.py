import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))   # maxcal_poisson.py
sys.path.insert(0, HERE)                             # langevin.py


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run slow validation tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow validation test (use --runslow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow validation test: use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
