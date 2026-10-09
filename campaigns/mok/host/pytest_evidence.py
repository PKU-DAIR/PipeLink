"""Give each torchrun worker its own standard pytest JUnit report."""

import os
from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    directory = Path(os.environ["MOK_TEST_REPORT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    config.option.xmlpath = str(directory / f"tests-rank-{os.environ['RANK']}.xml")
