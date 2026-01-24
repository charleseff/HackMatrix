"""
Pytest configuration and fixtures for environment parity tests.

Usage:
    pytest tests/                  # runs both [swift] and [jax]
    pytest tests/ -k swift         # runs only [swift] tests
    pytest tests/ -k jax           # runs only [jax] tests
    pytest tests/ -k "not jax"     # runs only [swift] tests (alternative)
"""

from collections.abc import Generator

import pytest

from .jax_env_wrapper import JaxEnvWrapper
from .swift_env_wrapper import SwiftEnvWrapper


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_set_state: mark test as requiring set_state() support"
    )
    config.addinivalue_line(
        "markers", "implementation: mark test as implementation-level (not interface parity)"
    )


@pytest.fixture(params=["swift", "jax"])
def env(request) -> Generator[SwiftEnvWrapper | JaxEnvWrapper, None, None]:
    """
    Parameterized fixture providing either Swift or JAX environment.

    Tests using this fixture run twice - once for each environment.
    Use pytest -k to filter: `pytest -k swift` or `pytest -k jax`

    Yields:
        SwiftEnvWrapper or JaxEnvWrapper instance
    """
    if request.param == "swift":
        with SwiftEnvWrapper() as e:
            yield e
    elif request.param == "jax":
        with JaxEnvWrapper() as e:
            yield e


@pytest.fixture
def swift_env() -> Generator[SwiftEnvWrapper, None, None]:
    """
    Swift-only fixture for tests that specifically need Swift.
    Prefer using `env` fixture for most tests.
    """
    with SwiftEnvWrapper() as e:
        yield e


@pytest.fixture
def jax_env() -> Generator[JaxEnvWrapper, None, None]:
    """
    JAX-only fixture for tests that specifically need JAX.
    Prefer using `env` fixture for most tests.
    """
    with JaxEnvWrapper() as e:
        yield e
