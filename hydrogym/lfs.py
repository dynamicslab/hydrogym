"""Detect Git LFS pointer files in place of the meshes and case files of a source checkout.

The Firedrake meshes (``*.msh``) and the Nek5000 case files (``*.re2``, ``*.ma2``, restart fields) are stored with
Git LFS. A clone without ``git lfs`` contains small text pointers instead of the files; the solvers then fail with
errors that do not name the cause. The wheels on PyPI contain the files themselves.
"""

from __future__ import annotations

import os

LFS_POINTER_PREFIX = b"version https://git-lfs"


def is_lfs_pointer(path: str | os.PathLike) -> bool:
    """True if ``path`` is a Git LFS pointer file (a text file starting with ``version https://git-lfs``)."""
    try:
        with open(path, "rb") as f:
            return f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except (OSError, IsADirectoryError):
        return False


def check_not_lfs_pointer(path: str | os.PathLike) -> None:
    """Raise ``RuntimeError`` with installation instructions if ``path`` is a Git LFS pointer file."""
    if is_lfs_pointer(path):
        raise RuntimeError(
            f"{os.fspath(path)} is a Git LFS pointer, not the file itself. Install Git LFS and run "
            "`git lfs install && git lfs pull` in the HydroGym checkout (or install HydroGym from PyPI)."
        )
