"""Gust file of the NACA0012Gust environment: the selected gust is put where m-AIA reads it (its working directory,
linked there by the workspace preparation), atomically, so that m-AIA, which loads the gust file at start-up while
the controller prepares the first episode, never reads a partly written file; a link into the data cache is replaced,
not written through."""

import os

import pytest

from hydrogym.data_manager import HFDataManager

HEADER = b"MAIAGUST"


def _write(path, payload):
    with open(path, "wb") as f:
        f.write(HEADER + payload)


def test_workspace_links_gust_files(tmp_path):
    env = tmp_path / "NACA0012Gust_2D_Re1000_AOA20"
    env.mkdir()
    (env / ".MAIA_LB").touch()
    for f in ("properties_run.toml", "geometry.toml", "environment_config.yaml", "grid.Netcdf", "restart_.Netcdf"):
        (env / f).write_text("x")
    (env / "stl").mkdir()
    for name in ("gust.bin", "gust_fig2c.bin", "gust_slug.bin"):
        _write(env / name, name.encode())
    dm = HFDataManager(use_clean_cache=False, cache_dir=str(tmp_path / "cache"))
    work = tmp_path / "work"
    dm.prepare_working_directory(str(env), str(work))
    for name in ("gust.bin", "gust_fig2c.bin", "gust_slug.bin"):
        assert (work / name).is_symlink()
        assert (work / name).read_bytes() == HEADER + name.encode()


def test_workspace_without_gust_files(tmp_path):
    env = tmp_path / "Cylinder_2D_Re200"
    env.mkdir()
    (env / ".MAIA_LB").touch()
    (env / "properties_run.toml").write_text("x")
    dm = HFDataManager(use_clean_cache=False, cache_dir=str(tmp_path / "cache"))
    dm.prepare_working_directory(str(env), str(tmp_path / "work"))
    assert not list((tmp_path / "work").glob("gust*"))


pytest.importorskip("mpi4py")
pytest.importorskip("omegaconf")

from hydrogym.maia.envs import naca0012  # noqa: E402


def test_install_replaces_link_not_cache(tmp_path):
    cache = tmp_path / "cache_gust.bin"
    _write(cache, b"slug")
    dst = tmp_path / "gust.bin"
    dst.symlink_to(cache)
    src = tmp_path / "gust_fig2c.bin"
    _write(src, b"fig2c" * 1000)
    naca0012.install_gust_file(str(src), str(dst))
    assert not dst.is_symlink()
    assert dst.read_bytes() == HEADER + b"fig2c" * 1000
    assert cache.read_bytes() == HEADER + b"slug"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cache_gust.bin", "gust.bin", "gust_fig2c.bin"]
    assert (dst.stat().st_mode & 0o777) == (src.stat().st_mode & 0o777)


def test_install_never_exposes_a_partial_file(tmp_path, monkeypatch):
    dst = tmp_path / "gust.bin"
    _write(dst, b"old")
    src = tmp_path / "new.bin"
    _write(src, b"new" * 1000)
    copy = naca0012.shutil.copyfile
    seen = []

    def copy_and_look(a, b, **kw):
        out = copy(a, b, **kw)
        seen.append(dst.read_bytes())  # what a concurrent reader of dst sees while the copy is in progress
        return out

    monkeypatch.setattr(naca0012.shutil, "copyfile", copy_and_look)
    naca0012.install_gust_file(str(src), str(dst))
    assert seen == [HEADER + b"old"]
    assert dst.read_bytes() == HEADER + b"new" * 1000


def test_gust_field_path(tmp_path):
    data, work = tmp_path / "data", tmp_path / "work"
    data.mkdir()
    work.mkdir()
    _write(data / "gust.bin", b"x")
    assert naca0012.gust_field_path("gust.bin", str(data), str(work)) == str(data / "gust.bin")
    (work / "gust.bin").symlink_to(data / "gust.bin")
    assert naca0012.gust_field_path("gust.bin", str(data), str(work)) == str(work / "gust.bin")
    assert naca0012.gust_field_path(str(data / "gust.bin"), "/elsewhere", str(work)) == str(data / "gust.bin")
    assert os.path.isabs(naca0012.gust_field_path("gust.bin", str(data)))


def test_named_gust_alias(tmp_path):
    for name in ("gust_fig2c.bin", "gust_slug.bin"):
        _write(tmp_path / name, name.encode())
    env = naca0012.NACA0012Gust.__new__(naca0012.NACA0012Gust)
    env.env_data_path = str(tmp_path)
    assert env._gust_path("transverse") == str(tmp_path / "gust_fig2c.bin")
    assert env._gust_path("fig2c") == str(tmp_path / "gust_fig2c.bin")
    assert env._gust_path("slug") == str(tmp_path / "gust_slug.bin")
