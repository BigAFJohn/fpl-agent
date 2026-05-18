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
    config.addinivalue_line("markers", "deepeval: DeepEval semantic metric tests (requires API key)")
    config.addinivalue_line("markers", "slow: tests that take >5 seconds")


def pytest_collection_modifyitems(config, items):
    """
    Auto-mark tests based on their directory location.

    Run modes:
      pytest -m unit           — fast only, no DB, no API
      pytest -m integration    — requires DB
      pytest -m evals          — rule-based agent evals
      pytest -m deepeval       — semantic metrics (costs API $)
      pytest -m "not deepeval" — everything except paid API tests
      pytest -m "not slow"     — skip slow tests
    """
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in path:
            item.add_marker(pytest.mark.integration)
        elif "/evals/" in path:
            item.add_marker(pytest.mark.evals)
            # Also mark DeepEval tests specifically
            if "deepeval" in str(item.fspath).lower():
                item.add_marker(pytest.mark.deepeval)
