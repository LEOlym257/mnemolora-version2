"""Stable, publication-facing entry point for the focused LUCID tests."""

from __future__ import annotations

import importlib
import unittest


MODULES = (
    "tests.test_fsd_v2_config",
    "tests.test_fsd_v2_raw_replay",
    "tests.test_fsd_v2_rtrc",
    "tests.test_fsd_v2_budget_controller",
    "tests.test_fsd_v2_deep_sleep",
    "tests.test_fsd_v2_residual_distillation",
    "tests.test_fsd_v2_no_external_data",
    "tests.test_lucid_public_api",
)


def load_tests(loader: unittest.TestLoader, tests, pattern):
    suite = unittest.TestSuite()
    for name in MODULES:
        suite.addTests(loader.loadTestsFromModule(importlib.import_module(name)))
    return suite


if __name__ == "__main__":
    unittest.main(module=__name__, verbosity=2)
