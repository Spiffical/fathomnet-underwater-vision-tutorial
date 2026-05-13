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


def classification_examples_by_class(
    dataset_root: str | Path,
    *,
    split: str = "val",
    examples_per_class: int = 1,
    max_classes: int | None = None,
) -> list[dict[str, object]]:
    """Select labelled classification images from an ImageFolder-style split.

    Each returned item has an `image_path` and `class_name`. The helper keeps
    the notebook focused on the modelling question: here the target is one class
    label per image, with no boxes or masks.
    """

    split_dir = Path(dataset_root) / split
    if not split_dir.exists():
        return []

    rows: list[dict[str, object]] = []
    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if max_classes is not None:
        class_dirs = class_dirs[:max_classes]

    for class_dir in class_dirs:
        for image_path in _image_files(class_dir)[:examples_per_class]:
            rows.append({"image_path": image_path, "class_name": class_dir.name})
    return rows


def yolo_label_instance_count(label_path: str | Path) -> int:
    """Count non-empty instance rows in one YOLO label file."""

    path = Path(label_path)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _matching_image_for_label(label_path: Path, image_dir: Path) -> Path | None:
    matches = [
        image_path
        for image_path in image_dir.glob(f"{label_path.stem}.*")
        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(matches)[0] if matches else None


def select_yolo_examples(
    dataset_root: str | Path,
    *,
    split: str = "val",
    min_instances: int = 2,
    limit: int = 4,
) -> list[dict[str, object]]:
    """Select YOLO examples with several labelled objects in one image.

    Multi-object examples are useful early in the tutorial because they make
    the distinction between image classification, object detection, and
    instance segmentation visible without needing a long explanation.
    """

    root = Path(dataset_root)
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.exists() or not label_dir.exists():
        return []

    examples: list[dict[str, object]] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        instance_count = yolo_label_instance_count(label_path)
        if instance_count < min_instances:
            continue
        image_path = _matching_image_for_label(label_path, image_dir)
        if image_path is None:
            continue
        examples.append(
            {
                "image_path": image_path,
                "label_path": label_path,
                "instance_count": instance_count,
            }
        )

    examples.sort(key=lambda item: (-int(item["instance_count"]), str(item["label_path"])))
    return examples[:limit]


def select_coco_category_examples(
    coco_json_path: str | Path,
    image_roots: str | Path | list[str | Path],
    *,
    limit: int = 6,
    min_annotations: int = 1,
) -> list[dict[str, object]]:
    """Select full images with their ground-truth COCO category labels.

    This is a classification-style view of full images: it shows which taxa or
    object categories are annotated as present, but it intentionally hides the
    boxes and masks so the geometric tasks remain visually distinct.
    """

    with Path(coco_json_path).open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    if isinstance(image_roots, (str, Path)):
        roots = [Path(image_roots)]
    else:
        roots = [Path(root) for root in image_roots]

    images_by_stem: dict[str, Path] = {}
    for root in roots:
        for image_path in _image_files(root):
            images_by_stem.setdefault(image_path.stem, image_path)

    names_by_category_id = {category["id"]: category["name"] for category in coco.get("categories", [])}
    image_records = {image["id"]: image for image in coco.get("images", [])}
    category_names_by_image: dict[int, list[str]] = {}
    annotation_count_by_image: Counter[int] = Counter()

    for annotation in coco.get("annotations", []):
        image_id = annotation["image_id"]
        category_name = names_by_category_id.get(annotation["category_id"], str(annotation["category_id"]))
        category_names_by_image.setdefault(image_id, []).append(category_name)
        annotation_count_by_image[image_id] += 1

    rows: list[dict[str, object]] = []
    for image_id, category_names in category_names_by_image.items():
        if annotation_count_by_image[image_id] < min_annotations:
            continue
        image_record = image_records.get(image_id)
        if not image_record:
            continue
        image_stem = Path(image_record["file_name"]).stem
        image_path = images_by_stem.get(image_stem)
        if image_path is None:
            continue
        rows.append(
            {
                "image_path": image_path,
                "category_names": sorted(set(category_names)),
                "annotation_count": annotation_count_by_image[image_id],
            }
        )

    rows.sort(key=lambda item: (-int(item["annotation_count"]), str(item["image_path"])))
    return rows[:limit]


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
    """Summarise all major bundle components."""

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
    should not be committed; it is a disposable local teaching artefact.
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
