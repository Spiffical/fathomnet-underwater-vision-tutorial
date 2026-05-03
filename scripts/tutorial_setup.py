"""Runtime setup helpers for the underwater vision tutorial.

The notebook is designed for Google Colab, but these helpers also work on a
local workstation. Heavy packages such as torch are imported lazily so the
module can be inspected even before the teaching environment is fully ready.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


DEFAULT_IMPORTS = {
    "ultralytics": "ultralytics",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
}


def find_repo_root(start: str | Path | None = None) -> Path:
    """Return the nearest parent containing the tutorial `scripts` directory."""

    start_path = Path(start or Path.cwd()).resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "scripts").exists() and (candidate / "notebooks").exists():
            return candidate
    return start_path


def set_reproducible_seed(seed: int = 42) -> int:
    """Set common pseudo-random seeds and return the seed that was used."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(seed)

    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    return seed


def _run_text(command: list[str], timeout: int = 10) -> str | None:
    """Run a short diagnostic command and return stdout if it succeeds."""

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def detect_runtime() -> dict[str, object]:
    """Summarize the active notebook/runtime environment."""

    in_colab = "google.colab" in sys.modules
    payload: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "in_colab": in_colab,
        "has_cuda": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "torch_version": None,
        "nvidia_smi": None,
    }

    if importlib.util.find_spec("torch") is not None:
        import torch

        payload["torch_version"] = torch.__version__
        payload["has_cuda"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            payload["cuda_device_name"] = torch.cuda.get_device_name(0)

    payload["nvidia_smi"] = _run_text(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        timeout=5,
    )
    return payload


def print_runtime_summary(runtime: dict[str, object] | None = None) -> dict[str, object]:
    """Pretty-print the runtime summary and return it for later cells."""

    runtime = runtime or detect_runtime()
    print(json.dumps(runtime, indent=2))
    return runtime


def ensure_dependencies(
    packages: dict[str, str] | None = None,
    *,
    install: bool = False,
    extra_pip_args: Iterable[str] = (),
) -> dict[str, object]:
    """Check or install notebook dependencies.

    Parameters
    ----------
    packages:
        Mapping from import name to pip package name.
    install:
        If True, missing packages are installed into the active interpreter.
        Keep this False while merely inspecting the repository.
    extra_pip_args:
        Extra arguments passed to pip, for example `("--quiet",)`.
    """

    packages = dict(packages or DEFAULT_IMPORTS)
    missing = [
        pip_name
        for import_name, pip_name in packages.items()
        if importlib.util.find_spec(import_name) is None
    ]
    result: dict[str, object] = {"missing": missing, "installed": []}

    if missing and install:
        command = [sys.executable, "-m", "pip", "install", *extra_pip_args, *missing]
        print("Installing:", " ".join(missing))
        subprocess.check_call(command)
        result["installed"] = missing
        result["missing"] = [
            pip_name
            for import_name, pip_name in packages.items()
            if importlib.util.find_spec(import_name) is None
        ]

    if result["missing"]:
        print("Missing optional dependencies:", ", ".join(result["missing"]))
    else:
        print("All requested dependencies are importable.")
    return result


def download_tutorial_bundle(
    *,
    bundle_url: str | None = None,
    bundle_zip_path: str | Path | None = None,
    output_dir: str | Path = "data/fathomnet_underwater_tutorial_bundle",
    force: bool = False,
) -> Path:
    """Download or unpack the prebuilt tutorial data bundle.

    The lookup order is:
    1. Existing `output_dir` with a manifest, unless `force=True`.
    2. Explicit `bundle_zip_path`.
    3. Local repo zip at `data/fathomnet_underwater_tutorial_bundle.zip`.
    4. `bundle_url`.
    """

    output_path = Path(output_dir).expanduser().resolve()
    manifest_path = output_path / "manifest.json"
    if manifest_path.exists() and not force:
        return output_path

    if force and output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = find_repo_root()
    candidates: list[Path] = []
    if bundle_zip_path is not None:
        candidates.append(Path(bundle_zip_path).expanduser())
    candidates.append(repo_root / "data" / "fathomnet_underwater_tutorial_bundle.zip")

    zip_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if zip_path is None and bundle_url:
        zip_path = output_path.parent / "fathomnet_underwater_tutorial_bundle.zip"
        print(f"Downloading tutorial bundle from {bundle_url}")
        urllib.request.urlretrieve(bundle_url, zip_path)

    if zip_path is None:
        raise FileNotFoundError(
            "Could not find a tutorial bundle zip. Set BUNDLE_URL or build "
            "data/fathomnet_underwater_tutorial_bundle.zip first."
        )

    print(f"Unpacking {zip_path} -> {output_path.parent}")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(output_path.parent)

    if not manifest_path.exists():
        nested = output_path.parent / "fathomnet_underwater_tutorial_bundle" / "manifest.json"
        if nested.exists() and nested.parent != output_path:
            if output_path.exists():
                shutil.rmtree(output_path)
            nested.parent.rename(output_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Bundle unpacked, but no manifest was found at {manifest_path}")
    return output_path

