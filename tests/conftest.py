# tests/conftest.py
"""
Shared pytest configuration and fixtures.

conftest.py is automatically loaded by pytest before any tests run.
Put shared fixtures, hooks, and configuration here.
"""

import pytest


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "unit: fast unit tests, no external dependencies")
    config.addinivalue_line("markers", "integration: requires database connection")
    config.addinivalue_line("markers", "evals: LLM output quality tests")
    config.addinivalue_line("markers", "slow: tests that take >5 seconds")


def pytest_collection_modifyitems(config, items):
    """
    Auto-mark tests based on their directory location.
    This lets you run: pytest -m unit  (fast only)
    or:               pytest -m "not slow"
    """
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in path:
            item.add_marker(pytest.mark.integration)
        elif "/evals/" in path:
            item.add_marker(pytest.mark.evals)
