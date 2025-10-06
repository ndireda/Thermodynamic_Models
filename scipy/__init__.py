"""Lightweight shim providing the subset of SciPy APIs required for the tests."""

from . import optimize  # noqa: F401

__all__ = ["optimize"]
