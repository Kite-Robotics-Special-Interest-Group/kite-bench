"""Locate (and if needed fetch) the MuJoCo WidowX-250 model.

The arm model comes from ``mujoco_menagerie`` (Apache-2.0) and is not vendored in
this repository. Set ``KITEBENCH_WX250S`` to point at an existing
``trossen_wx250s`` directory, otherwise the model is sparse-cloned into
``~/.cache/kitebench/models`` on first use.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

MENAGERIE = "https://github.com/google-deepmind/mujoco_menagerie.git"
MODEL_DIR = "trossen_wx250s"


def cache_root() -> Path:
    return Path(os.environ.get("KITEBENCH_CACHE", Path.home() / ".cache" / "kitebench"))


def model_path() -> Path:
    override = os.environ.get("KITEBENCH_WX250S")
    if override:
        return Path(override)
    return cache_root() / "models" / MODEL_DIR


def ensure_model(*, allow_download: bool = True) -> Path:
    """Return the directory holding ``wx250s.xml``, fetching it if necessary."""
    path = model_path()
    if (path / "wx250s.xml").exists():
        return path
    if not allow_download:
        raise FileNotFoundError(
            f"missing {path / 'wx250s.xml'}; set KITEBENCH_WX250S or allow_download=True"
        )

    target = cache_root() / "models"
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "menagerie"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             MENAGERIE, str(repo)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "sparse-checkout", "set", MODEL_DIR],
            check=True,
            capture_output=True,
        )
        shutil.copytree(repo / MODEL_DIR, path)
    return path

