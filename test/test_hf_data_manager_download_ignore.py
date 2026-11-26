"""Nek5000 environments download only the restart set the case reads (restart_files/), not other sets next to it."""

import pytest

pytest.importorskip("huggingface_hub")

from huggingface_hub.utils import filter_repo_objects  # noqa: E402

from hydrogym.data_manager import HFDataManager  # noqa: E402

ENV = "NACA0012_3D_Re200000_AOA0"
FILES = [
    f"{ENV}/environment_config.yaml",
    f"{ENV}/naca_wing.re2",
    f"{ENV}/restart_files/rs8naca_wing0.f00001",
    f"{ENV}/restart_files_skipTransient/init_1/rs8naca_wing0.f00001",
    "TCFmini_3D_Re180/restart_files/rs8tcf0.f00001",
]


@pytest.mark.parametrize("profile", ["NEK5000_v17", "NEK5000_v19"])
def test_nek5000_skips_other_restart_sets(profile):
    ignore = HFDataManager._download_ignore(ENV, profile)
    kept = list(filter_repo_objects(FILES, allow_patterns=f"{ENV}/**", ignore_patterns=ignore))
    assert kept == FILES[:3]


@pytest.mark.parametrize("profile", ["MAIA_LB", "JAX", None])
def test_other_profiles_download_everything(profile):
    assert HFDataManager._download_ignore(ENV, profile) is None
