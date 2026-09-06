"""
stratum_app.py

Standalone-app entry point for the STRATUM GUI - what the packaged .exe runs.

It does three things before handing over to the same workflow the scripts use:

  1. Anchors the working directory to the folder holding the executable, so the
     cached weights are found and every output lands next to the app.
  2. Gives sys.stdout/stderr safe writers when running windowed (a --windowed exe
     has no console, so they are None).
  3. Seeds the MNIST cache (~/.keras/datasets/mnist.npz) from the bundled copy.

No workflow logic lives here: it launches stratum_platform.Platform with the
standard study registry.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):          # PyInstaller exe
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _bundle_dir() -> Path:
    # Where PyInstaller unpacks bundled data files (_internal in one-dir mode).
    return Path(getattr(sys, "_MEIPASS", _base_dir()))


class _NullWriter:
    """stdout/stderr stand-in for windowed mode (no console attached)."""

    def write(self, s: str) -> int:
        return len(s)

    def flush(self) -> None:
        pass


def _seed_file(bundled: Path, dest: Path) -> None:
    if dest.exists() or not bundled.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundled, dest)


def main() -> None:
    base = _base_dir()
    os.chdir(base)

    if sys.stdout is None:
        sys.stdout = _NullWriter()
    if sys.stderr is None:
        sys.stderr = _NullWriter()

    bundle = _bundle_dir()
    # MNIST dataset -> the keras-format cache mnist_ann.load_mnist() reads.
    _seed_file(bundle / "seed_data" / "mnist.npz",
               Path.home() / ".keras" / "datasets" / "mnist.npz")
    # Trained MLP weights -> next to the exe (MNIST_WEIGHTS_PATH is relative).
    _seed_file(bundle / "seed_data" / "mnist_mlp_weights.npz",
               base / "mnist_mlp_weights.npz")

    from stratum_platform import Platform, build_default_studies

    Platform(build_default_studies(), initial="mlp").launch()


if __name__ == "__main__":
    main()
