---
sidebar_label: lfs
title: hydrogym.lfs
---

Detect Git LFS pointer files in place of the meshes and case files of a source checkout.

The Firedrake meshes (`*.msh`) and the Nek5000 case files (`*.re2`, `*.ma2`, restart fields) are stored with
Git LFS. A clone without `git lfs` contains small text pointers instead of the files; the solvers then fail with
errors that do not name the cause. The wheels on PyPI contain the files themselves.

#### is\_lfs\_pointer

```python
def is_lfs_pointer(path: str | os.PathLike) -> bool
```

True if `path` is a Git LFS pointer file (a text file starting with `version https://git-lfs`).

#### check\_not\_lfs\_pointer

```python
def check_not_lfs_pointer(path: str | os.PathLike) -> None
```

Raise `RuntimeError` with installation instructions if `path` is a Git LFS pointer file.

