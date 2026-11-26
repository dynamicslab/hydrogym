"""Git LFS pointer detection for meshes and case files (hydrogym.lfs)."""

import pytest

from hydrogym.lfs import check_not_lfs_pointer, is_lfs_pointer

POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:4d7a214614ab2935c943f9e0ff69d22eadbb8f32b1258daaa5e2ca24d17e2393\n"
    "size 12345\n"
)


def test_pointer_is_detected(tmp_path):
    p = tmp_path / "medium.msh"
    p.write_text(POINTER)
    assert is_lfs_pointer(p)
    with pytest.raises(RuntimeError, match="git lfs pull"):
        check_not_lfs_pointer(p)


def test_real_file_passes(tmp_path):
    p = tmp_path / "medium.msh"
    p.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n")
    assert not is_lfs_pointer(p)
    check_not_lfs_pointer(p)


def test_missing_file_is_not_a_pointer(tmp_path):
    assert not is_lfs_pointer(tmp_path / "missing.re2")


def test_tracked_meshes_are_checked_out():
    """The Firedrake meshes of this checkout are the files themselves (skipped in a clone without Git LFS)."""
    import pathlib

    meshes = sorted(pathlib.Path(__file__).resolve().parents[1].glob("hydrogym/firedrake/envs/*/*.msh"))
    if not meshes:
        pytest.skip("no meshes in this tree")
    pointers = [m for m in meshes if is_lfs_pointer(m)]
    if pointers:
        pytest.skip(f"{len(pointers)} meshes are Git LFS pointers (run `git lfs pull`)")
    assert all(m.stat().st_size > 1024 for m in meshes)
