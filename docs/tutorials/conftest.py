"""Pytest configuration for executable tutorials."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: owner-run tutorial acceptance on PIPELINK_TUTORIAL_GPU"
    )
