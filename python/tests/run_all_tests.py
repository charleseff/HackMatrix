#!/usr/bin/env python3
"""
Run all environment parity tests using pytest.

Usage:
    cd python && source venv/bin/activate && python tests/run_all_tests.py

Or use pytest directly:
    cd python && source venv/bin/activate && pytest tests/ -v
"""

import sys
import pytest


def main():
    """Run pytest on the tests directory."""
    return pytest.main(["-v", "--tb=short", "tests/"])


if __name__ == "__main__":
    sys.exit(main())
