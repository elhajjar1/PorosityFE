"""Shared pytest fixtures for the per-module test suite (#124).

Created as part of the issue #124 split of the monolithic
``tests/test_porosity_fe.py`` into per-module test files. The repo-root
``conftest.py`` already handles ``sys.path`` for imports; this file holds
fixtures that need to be auto-discovered across the per-module tests.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _restore_porosity_logger():
    """Restore the ``porosity_fe_analysis`` logger between tests.

    ``porosity_fe.cli._configure_cli_logging`` flips ``logger.propagate``
    to ``False`` and attaches a stdout handler so the CLI banner survives.
    Before #124 the CLI tests sat *after* the conditioning / FE-solver
    tests in a single monolithic file, so this state pollution never
    bit anything. After the split ``test_cli.py`` collects first
    alphabetically, and tests that use ``caplog`` against the
    ``porosity_fe_analysis`` logger (TestPenaltyFactorAndConditioning,
    etc.) stop seeing records because ``caplog`` attaches to the root
    handler and the CLI broke propagation.

    Resetting propagate/level and stripping the CLI-injected handler
    after every test makes the tests independent of file collection
    order without touching the production behavior of
    ``_configure_cli_logging``.
    """
    logger = logging.getLogger("porosity_fe_analysis")
    original_propagate = logger.propagate
    original_level = logger.level
    yield
    logger.propagate = original_propagate
    logger.setLevel(original_level)
    # Remove any handler the CLI helper attached. Pytest's caplog plugin
    # adds and removes its own LogCaptureHandler -- we only target the
    # CLI marker so we don't fight with the framework.
    for h in list(logger.handlers):
        if getattr(h, "_porosity_cli", False):
            logger.removeHandler(h)
