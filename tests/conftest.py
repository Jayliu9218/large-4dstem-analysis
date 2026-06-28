"""Shared pytest configuration and fixtures for the 4D-STEM pipeline tests.

Applies workarounds for known environment incompatibilities before any
test module is imported.
"""


def pytest_configure(config):
    """Apply workarounds before test collection.

    On Windows, certain OpenBLAS builds trigger an illegal-instruction
    crash (0xc06d007f) inside ``threadpoolctl`` when sklearn tries to
    check the BLAS configuration at import time.  We stub the check so
    that py4DSTEM (and anything else importing sklearn) can load safely.
    """
    import os

    if os.name == "nt":
        _patch_sklearn_openblas_check()


def _patch_sklearn_openblas_check():
    """Replace ``sklearn.utils.fixes._in_unstable_openblas_configuration``
    with a no-op that always returns ``False``."""
    try:
        import sklearn.utils.fixes as fixes
        if hasattr(fixes, "_in_unstable_openblas_configuration"):
            fixes._in_unstable_openblas_configuration = lambda: False
    except Exception:
        pass
