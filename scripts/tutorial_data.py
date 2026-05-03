"""Dataset helpers for the underwater vision tutorial."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_manifest(bundle_root: str | Path) -> dict:
    """Load the bundle manifest."""

    manifest_path = Path(bundle_root) / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _image_files(path: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_runtime_yolo_yaml(dataset_dir: str | Path, output_name: str = "dataset.runtime.yaml") -> Path:
    """Write a portable Ultralytics YAML with an absolute `path` entry."""

    dataset_path = Path(dataset_dir).resolve()
    source_yaml = dataset_path / "dataset.yaml"
    with source_yaml.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload["path"] = str(dataset_path)

    runtime_yaml = dataset_path / output_name
    with runtime_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return runtime_yaml


def get_task_paths(task: str, bundle_root: str | Path) -> dict[str, Path]:
    """Return important paths for a tutorial task.

    `task` may be `classification`, `detect`, `segment`, `coco`, `sam3`, or
    `cached_training`.
    """

    root = Path(bundle_root).resolve()
    normalized = task.lower().strip()
    if normalized in {"classification", "classify", "cls"}:
        return {"root": root / "classification_crops"}
    if normalized in {"detect", "detection", "bbox"}:
        dataset_dir = root / "yolo_detect_binary"
        return {"root": dataset_dir, "yaml": make_runtime_yolo_yaml(dataset_dir)}
    if normalized in {"segment", "segmentation", "seg"}:
        dataset_dir = root / "yolo_segment_binary"
        return {"root": dataset_dir, "yaml": make_runtime_yolo_yaml(dataset_dir)}
    if normalized == "coco":
        return {"root": root / "coco", "json": root / "coco" / "subset.json"}
    if normalized == "sam3":
        return {"root": root / "sam3_cached_outputs", "index": root / "sam3_cached_outputs" / "index.json"}
    if normalized == "cached_training":
        return {"root": root / "cached_training"}
    if normalized == "licenses":
        return {"root": root / "licenses", "attribution": root / "licenses" / "attribution.csv"}
    raise ValueError(f"Unknown task: {task}")


def summarize_classification_dataset(dataset_root: str | Path) -> dict[str, object]:
    """Count images by split and class in an ImageFolder-style dataset."""

    root = Path(dataset_root)
    summary: dict[str, object] = {"root": str(root), "splits": {}}
    for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        class_counts = {
            class_dir.name: len(_image_files(class_dir))
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir())
        }
        summary["splits"][split_dir.name] = class_counts
    return summary


def _load_dataset_yaml(dataset_yaml: str | Path) -> tuple[Path, dict]:
    yaml_path = Path(dataset_yaml).resolve()
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    root = Path(payload.get("path", yaml_path.parent)).expanduser()
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root, payload


def validate_yolo_dataset(dataset_yaml: str | Path, *, task: str = "detect") -> dict[str, object]:
    """Check that a YOLO dataset has image/label pairs and sane labels."""

    root, payload = _load_dataset_yaml(dataset_yaml)
    names = payload.get("names", {})
    class_count = len(names) if isinstance(names, (list, dict)) else int(payload.get("nc", 0))
    line_lengths = Counter()
    invalid_lines: list[str] = []
    split_summaries: dict[str, dict[str, object]] = {}

    for split in ["train", "val", "test"]:
        split_value = payload.get(split)
        if not split_value:
            continue
        image_dir = root / split_value
        label_dir = root / str(split_value).replace("images", "labels", 1)
        image_paths = _image_files(image_dir) if image_dir.exists() else []
        label_paths = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in label_paths}
        empty_labels = 0

        for label_path in label_paths:
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                empty_labels += 1
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                line_lengths[len(parts)] += 1
                try:
                    class_id = int(float(parts[0]))
                    values = [float(value) for value in parts[1:]]
                except Exception:
                    invalid_lines.append(f"{label_path}:{line_number}: non-numeric label")
                    continue
                if class_id < 0 or (class_count and class_id >= class_count):
                    invalid_lines.append(f"{label_path}:{line_number}: bad class id {class_id}")
                if any(value < 0 or value > 1 for value in values):
                    invalid_lines.append(f"{label_path}:{line_number}: coordinates outside [0, 1]")
                if task == "detect" and len(parts) != 5:
                    invalid_lines.append(f"{label_path}:{line_number}: detection labels need 5 values")
                if task == "segment" and (len(parts) < 7 or len(values) % 2 != 0):
                    invalid_lines.append(f"{label_path}:{line_number}: segmentation labels need class plus >= 3 xy points")

        split_summaries[split] = {
            "images": len(image_paths),
            "labels": len(label_paths),
            "empty_label_files": empty_labels,
            "missing_label_files": sorted(image_stems - label_stems)[:10],
            "labels_without_images": sorted(label_stems - image_stems)[:10],
        }

    return {
        "root": str(root),
        "class_count": class_count,
        "names": names,
        "splits": split_summaries,
        "label_line_lengths": dict(line_lengths),
        "invalid_lines": invalid_lines[:25],
        "valid": not invalid_lines,
    }


def summarize_dataset(bundle_root: str | Path) -> dict[str, object]:
    """Summarize all major bundle components."""

    root = Path(bundle_root)
    return {
        "manifest": load_manifest(root),
        "classification": summarize_classification_dataset(root / "classification_crops"),
        "detection": validate_yolo_dataset(make_runtime_yolo_yaml(root / "yolo_detect_binary"), task="detect"),
        "segmentation": validate_yolo_dataset(make_runtime_yolo_yaml(root / "yolo_segment_binary"), task="segment"),
    }


def make_tiny_detection_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    train_images: int = 8,
    val_images: int = 8,
) -> Path:
    """Copy a tiny YOLO detection dataset for quick overfit/debugging labs.

    The tiny dataset is intentionally created under `tmp/` by the notebook. It
    should not be committed; it is a disposable local teaching artifact.
    """

    source_root = Path(source_root)
    output_root = Path(output_root)
    for split, count in {"train": train_images, "val": val_images}.items():
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        labels = sorted((source_root / "labels" / split).glob("*.txt"))[:count]
        for label_path in labels:
            image_candidates = list((source_root / "images" / split).glob(f"{label_path.stem}.*"))
            if not image_candidates:
                continue
            shutil.copy2(image_candidates[0], output_root / "images" / split / image_candidates[0].name)
            shutil.copy2(label_path, output_root / "labels" / split / label_path.name)

    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "object"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path
