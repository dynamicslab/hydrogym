#!/usr/bin/env python3
"""Verify that HydroGym distributions preserve metadata and solver data files."""

from __future__ import annotations

import argparse
import email
import tarfile
import zipfile
from pathlib import Path

EXPECTED_EXTRAS = {"all", "firedrake", "jax", "jaxfluids", "maia", "nek"}
DATA_SUFFIXES = {".geo", ".msh"}


def project_data_files() -> set[str]:
    return {path.as_posix() for path in Path("hydrogym").rglob("*") if path.is_file() and path.suffix in DATA_SUFFIXES}


def normalized_archive_data(names: list[str]) -> set[str]:
    normalized = set()
    for name in names:
        marker = "hydrogym/"
        offset = name.find(marker)
        if offset >= 0 and Path(name).suffix in DATA_SUFFIXES:
            normalized.add(name[offset:])
    return normalized


def verify_wheel(wheel: Path, expected_data: set[str]) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(archive.read(metadata_name))

    assert normalized_archive_data(names) == expected_data
    assert metadata["Name"] == "hydrogym"
    assert metadata["Version"] == "1.0.0"
    assert metadata["Requires-Python"].replace(" ", "") == ">=3.10,<3.15"
    assert set(metadata.get_all("Provides-Extra", [])) == EXPECTED_EXTRAS

    requirements = metadata.get_all("Requires-Dist", [])
    assert any(requirement.startswith("numpy>=1.23") for requirement in requirements)
    pyvista = [requirement for requirement in requirements if requirement.startswith("pyvista")]
    assert len(pyvista) == 2
    assert any("extra == 'jaxfluids'" in requirement for requirement in pyvista)
    assert any("extra == 'all'" in requirement for requirement in pyvista)


def verify_sdist(sdist: Path, expected_data: set[str]) -> None:
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
    assert normalized_archive_data(names) == expected_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", type=Path)
    args = parser.parse_args()

    wheels = sorted(args.dist.glob("*.whl"))
    sdists = sorted(args.dist.glob("*.tar.gz"))
    assert len(wheels) == 1, f"Expected one wheel, found {wheels}"
    assert len(sdists) == 1, f"Expected one sdist, found {sdists}"

    expected_data = project_data_files()
    assert len(expected_data) == 18, f"Expected 18 mesh files, found {len(expected_data)}"
    verify_wheel(wheels[0], expected_data)
    verify_sdist(sdists[0], expected_data)
    print(f"Verified {wheels[0].name}, {sdists[0].name}, and {len(expected_data)} mesh files")


if __name__ == "__main__":
    main()
