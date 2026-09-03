"""Tests for hydrogym's top-level lazy-loading of solver backends.

(Audit Task 1.4: `hydrogym/__init__.py`'s lazy-loader allowlist omitted
`jax`/`jaxfluids`, so attribute access after a bare `import hydrogym`
failed for those two backends while working for the other three.)

Guarded with `pytest.importorskip`: exercises the real imports wherever the
backend extras are installed (e.g. the GPU-stack devcontainer) and skips
cleanly in a bare environment with only gymnasium+numpy.
"""

import sys

import pytest


def test_all_lists_all_backend_names():
    import hydrogym

    for backend in ("distributed", "firedrake", "jax", "jaxfluids", "maia", "nek"):
        assert backend in hydrogym.__all__


@pytest.mark.parametrize("backend", ["jax", "jaxfluids"])
def test_backend_resolves_via_attribute_access(backend: str):
    # Skip when the backend's own dependencies aren't installed (the lazy
    # loader cannot import it without them -- same skip logic for firedrake/
    # maia/nek would need their deps too, but those were already loadable
    # before Task 1.4 and have MPI import side effects; only the two newly
    # added names are exercised here).
    pytest.importorskip(backend if backend == "jax" else "jaxfluids_rl" if backend == "jaxfluids" else backend)

    import hydrogym

    # Fresh subprocess-free check: the attribute must not be statically
    # present but must resolve via __getattr__ and cache into globals.
    module = getattr(hydrogym, backend)
    assert module.__name__ == f"hydrogym.{backend}"
    assert getattr(hydrogym, backend) is module


def test_lazy_loader_defers_import_until_attribute_access():
    # The whole point of __getattr__ is that a bare `import hydrogym` must
    # NOT eagerly import any backend (MPI-init safety in MPMD mode for
    # maia/nek; heavy GPU imports for jax). Verify jax specifically: before
    # attribute access it must not be loaded.
    import hydrogym  # noqa: F401

    pytest.importorskip("jax")

    # If this process already imported hydrogym.jax through some other test,
    # clear BOTH caches so the deferral is actually observable: the
    # sys.modules entry AND the attribute the lazy loader caches in the
    # package's globals() (an existing module attribute short-circuits
    # __getattr__ entirely, so popping only sys.modules would make the
    # attribute access a plain dict lookup and the test would fail on the
    # sys.modules assertion even though the loader itself works).
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "hydrogym.jax" or k.startswith("hydrogym.jax.")}
    import hydrogym

    cached_attr = None
    # NOTE: probe vars(), not hasattr -- hasattr() itself goes through
    # module __getattr__ and would trigger the very import being tested.
    if "jax" in vars(hydrogym):
        cached_attr = vars(hydrogym).pop("jax")

    try:
        assert "hydrogym.jax" not in sys.modules
        assert "jax" not in vars(hydrogym)
        hydrogym.jax  # noqa: B018 -- attribute access triggers the lazy import
        assert "hydrogym.jax" in sys.modules
        assert getattr(hydrogym, "jax") is sys.modules["hydrogym.jax"]
    finally:
        # Restore whatever was loaded before, so other tests are unaffected.
        sys.modules.update(saved)
        if cached_attr is not None:
            vars(hydrogym)["jax"] = cached_attr
