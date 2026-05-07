"""Build the compact data bundle used by the underwater vision tutorial.

The builder uses local artifacts from the existing ONC YOLO/SAM3 work and
creates a portable bundle with:

- ImageFolder-style classification crops.
- Binary YOLO detection labels.
- Binary YOLO segmentation labels.
- A COCO subset JSON for inspection.
- SAM3-like cached prompt outputs for fallback classroom use.
- Small cached training curves for CPU/no-GPU runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image


RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


SOURCE_ROOT = Path("/home/sbialek/ONC/yolo_segmentation")
SOURCE_SAM3_ROOT = Path("/home/sbialek/ONC/sam3")
YOLO100 = SOURCE_ROOT / "data" / "yolo_subset"
SUBSET100 = SOURCE_ROOT / "data" / "subset"
SUBSET500 = SOURCE_ROOT / "data" / "subset_500"
LABEL_PLAN = SOURCE_ROOT / "configs" / "label_plans.yaml"
YOLO_MIN_BOX_AREA = 0.005
CLASSIFICATION_MAX_CLASSES = 12
CLASSIFICATION_MIN_PER_CLASS = 20

WORKSHOP_CLASS_MAP = {
    "fish": "fish",
    "gelatinous": "gelatinous",
    "sponge": "sponge_coral",
    "coral_anemone": "sponge_coral",
    "crustacean": "crustacean",
    "echinoderm": "echinoderm",
}

PROMPT_TO_CLASS = {
    "fish": "fish",
    "sponge": "sponge_coral",
    "gelatinous animal": "gelatinous",
    "small crab": "crustacean",
    "echinoderm": "echinoderm",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def convert_image_to_jpeg(src: Path, dst: Path, *, quality: int = 88, max_side: int | None = None) -> tuple[int, int]:
    """Copy an image as RGB JPEG, optionally resizing by maximum side."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image = image.convert("RGB")
        if max_side is not None:
            width, height = image.size
            scale = min(1.0, max_side / max(width, height))
            if scale < 1.0:
                image = image.resize((round(width * scale), round(height * scale)), RESAMPLE_LANCZOS)
        image.save(dst, "JPEG", quality=quality, optimize=True)
        return image.size


def read_yolo_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append([float(part) for part in line.split()])
    return rows


def segment_to_detection_row(row: list[float]) -> list[float] | None:
    """Convert one YOLO polygon row to one YOLO normalized bbox row."""

    if len(row) < 7:
        return None
    class_id = int(row[0])
    coords = row[1:]
    xs = coords[0::2]
    ys = coords[1::2]
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return None
    return [class_id, (x0 + x1) / 2, (y0 + y1) / 2, width, height]


def yolo_row_box_area(row: list[float], *, task: str) -> float:
    """Return normalized bounding-box area for a YOLO detection or segment row."""

    coords = row[1:]
    if task == "detect":
        if len(coords) != 4:
            return 0.0
        return max(0.0, coords[2]) * max(0.0, coords[3])
    if len(coords) < 6:
        return 0.0
    xs = coords[0::2]
    ys = coords[1::2]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def format_yolo_row(row: list[float]) -> str:
    return f"{int(row[0])} " + " ".join(f"{value:.6f}" for value in row[1:])


def matching_image(label_path: Path, image_dir: Path) -> Path:
    for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        candidate = image_dir / f"{label_path.stem}{extension}"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No image found for {label_path}")


def write_yolo_yaml(dataset_dir: Path, *, task: str) -> None:
    content = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "object"},
        "tutorial_task": task,
        "tutorial_note": f"Objects with normalized bbox area below {YOLO_MIN_BOX_AREA} are omitted for workshop-scale training.",
    }
    with (dataset_dir / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, sort_keys=False)


def build_yolo_tasks(output_root: Path, *, image_quality: int, min_box_area: float) -> dict:
    """Build binary YOLO datasets, filtering tiny objects for teachability."""

    task_counts = {"yolo_min_box_area": min_box_area}
    for task in ["detect", "segment"]:
        dataset_dir = output_root / f"yolo_{task}_binary"
        for split in ["train", "val"]:
            image_out = dataset_dir / "images" / split
            label_out = dataset_dir / "labels" / split
            image_out.mkdir(parents=True, exist_ok=True)
            label_out.mkdir(parents=True, exist_ok=True)

            source_label_dir = YOLO100 / "labels" / split
            source_image_dir = YOLO100 / "images" / split
            image_count = 0
            instance_count = 0
            for label_path in sorted(source_label_dir.glob("*.txt")):
                source_image = matching_image(label_path, source_image_dir)
                convert_image_to_jpeg(source_image, image_out / f"{label_path.stem}.jpg", quality=image_quality)

                rows = read_yolo_rows(label_path)
                if task == "segment":
                    output_rows = rows
                else:
                    output_rows = [row for row in (segment_to_detection_row(row) for row in rows) if row is not None]
                output_rows = [
                    row
                    for row in output_rows
                    if yolo_row_box_area(row, task=task) >= min_box_area
                ]
                (label_out / label_path.name).write_text(
                    "\n".join(format_yolo_row(row) for row in output_rows),
                    encoding="utf-8",
                )
                image_count += 1
                instance_count += len(output_rows)

            task_counts[f"{task}_{split}_images"] = image_count
            task_counts[f"{task}_{split}_instances"] = instance_count

        write_yolo_yaml(dataset_dir, task=task)
    return task_counts


def load_workshop_class_mapping() -> dict[str, str]:
    with LABEL_PLAN.open("r", encoding="utf-8") as handle:
        plans = yaml.safe_load(handle)["plans"]["coarse_v1_bio8"]["classes"]
    mapping: dict[str, str] = {}
    for coarse_name, concept_names in plans.items():
        workshop_name = WORKSHOP_CLASS_MAP.get(coarse_name)
        if workshop_name is None:
            continue
        for concept_name in concept_names:
            mapping[concept_name] = workshop_name
    return mapping


def slugify_class_name(name: str) -> str:
    """Return a filesystem-friendly class folder name for a FathomNet concept."""

    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "unnamed_class"


def expanded_bbox(bbox: list[float], width: int, height: int, margin_fraction: float = 0.12) -> tuple[int, int, int, int]:
    x, y, box_w, box_h = bbox
    margin = margin_fraction * max(box_w, box_h)
    x0 = max(0, int(x - margin))
    y0 = max(0, int(y - margin))
    x1 = min(width, int(x + box_w + margin))
    y1 = min(height, int(y + box_h + margin))
    return x0, y0, x1, y1


def build_classification_crops(
    output_root: Path,
    *,
    seed: int,
    max_per_class: int,
    max_classes: int,
    min_per_class: int,
    crop_size: int,
    image_quality: int,
) -> dict:
    """Build an ImageFolder dataset from common source-level FathomNet concepts.

    Earlier versions collapsed these crops into five broad classes. That made
    top-5 accuracy mathematically uninformative because every class could fit
    inside the top five predictions. Here we use common source concepts instead:
    the classes are still small enough for Colab, but ranking metrics now teach
    something real.
    """

    rng = random.Random(seed)
    data = load_json(SUBSET500 / "subset.json")
    image_by_id = {image["id"]: image for image in data["images"]}
    category_by_id = {category["id"]: category["name"] for category in data["categories"]}

    candidates: dict[str, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        concept = category_by_id.get(annotation["category_id"])
        if concept is None:
            continue
        image_info = image_by_id.get(annotation["image_id"])
        if image_info is None:
            continue
        source_image = SUBSET500 / "images" / image_info["file_name"]
        if not source_image.exists():
            continue
        _, _, box_width, box_height = annotation["bbox"]
        if box_width < 8 or box_height < 8:
            continue
        candidates[concept].append(
            {
                "annotation": annotation,
                "image": image_info,
                "concept": concept,
                "source_image": source_image,
            }
        )

    eligible = [
        (concept, rows)
        for concept, rows in candidates.items()
        if len(rows) >= min_per_class
    ]
    eligible.sort(key=lambda item: (-len(item[1]), item[0].lower()))
    selected_concepts = eligible[:max_classes]
    source_concepts_by_class: dict[str, str] = {}

    crop_manifest_rows: list[dict] = []
    class_counts: dict[str, dict[str, int]] = {}
    for source_concept, rows in selected_concepts:
        class_name = slugify_class_name(source_concept)
        if class_name in source_concepts_by_class:
            class_name = f"{class_name}_{len(source_concepts_by_class)}"
        source_concepts_by_class[class_name] = source_concept

        rng.shuffle(rows)
        selected = rows[:max_per_class]
        split_cut = max(1, int(len(selected) * 0.8))
        class_counts[class_name] = {"train": 0, "val": 0, "available_annotations": len(rows)}
        for index, row in enumerate(selected):
            split = "train" if index < split_cut else "val"
            image_info = row["image"]
            annotation = row["annotation"]
            with Image.open(row["source_image"]) as image:
                image = image.convert("RGB")
                x0, y0, x1, y1 = expanded_bbox(annotation["bbox"], image.width, image.height)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                crop = image.crop((x0, y0, x1, y1)).resize((crop_size, crop_size), RESAMPLE_LANCZOS)

            crop_name = f"{class_name}_{annotation['id']}_{Path(image_info['file_name']).stem}.jpg"
            relative_path = Path("classification_crops") / split / class_name / crop_name
            output_path = output_root / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(output_path, "JPEG", quality=image_quality, optimize=True)
            class_counts[class_name][split] += 1
            crop_manifest_rows.append(
                {
                    "relative_path": str(relative_path),
                    "split": split,
                    "class_name": class_name,
                    "source_concept": source_concept,
                    "source_image": image_info["file_name"],
                    "annotation_id": annotation["id"],
                }
            )

    manifest_path = output_root / "classification_crops" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "split",
                "class_name",
                "source_concept",
                "source_image",
                "annotation_id",
            ],
        )
        writer.writeheader()
        writer.writerows(crop_manifest_rows)

    return {
        "classification_counts": class_counts,
        "classification_crops": len(crop_manifest_rows),
        "classification_source_concepts": source_concepts_by_class,
        "classification_max_classes": max_classes,
        "classification_min_per_class": min_per_class,
    }


def polygon_box(polygon: list[list[float]]) -> list[float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_to_normalized_xyxy(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, box_w, box_h = bbox
    return [
        max(0.0, min(1.0, x / width)),
        max(0.0, min(1.0, y / height)),
        max(0.0, min(1.0, (x + box_w) / width)),
        max(0.0, min(1.0, (y + box_h) / height)),
    ]


def build_sam3_cached_outputs(output_root: Path) -> dict:
    """Create SAM3-like cached prompt outputs from available labels."""

    subset = load_json(SUBSET100 / "subset.json")
    concept_to_workshop = load_workshop_class_mapping()
    images_by_stem = {Path(image["file_name"]).stem: image for image in subset["images"]}
    category_by_id = {category["id"]: category["name"] for category in subset["categories"]}
    anns_by_stem: dict[str, list[dict]] = defaultdict(list)
    for annotation in subset["annotations"]:
        image_info = next((image for image in subset["images"] if image["id"] == annotation["image_id"]), None)
        if image_info is not None:
            anns_by_stem[Path(image_info["file_name"]).stem].append(annotation)

    label_paths = sorted((output_root / "yolo_segment_binary" / "labels" / "val").glob("*.txt"))
    if len(label_paths) < 5:
        label_paths = sorted((output_root / "yolo_segment_binary" / "labels" / "train").glob("*.txt"))[:5]
    else:
        label_paths = label_paths[:5]

    cache_dir = output_root / "sam3_cached_outputs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "description": "Cached SAM3-like prompt outputs for workshop fallback use.",
        "images": {},
    }

    prompts = ["marine organism", "object", "fish", "sponge", "gelatinous animal", "small crab", "echinoderm"]
    for label_path in label_paths:
        stem = label_path.stem
        split = "val" if (output_root / "yolo_segment_binary" / "images" / "val" / f"{stem}.jpg").exists() else "train"
        image_rel = Path("yolo_segment_binary") / "images" / split / f"{stem}.jpg"
        image_info = images_by_stem.get(stem)
        if image_info is None:
            continue

        segment_polygons = []
        for row in read_yolo_rows(label_path):
            coords = row[1:]
            polygon = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
            if len(polygon) >= 3:
                segment_polygons.append(polygon)

        index["images"][stem] = {"image": str(image_rel), "prompts": {}}
        for prompt in prompts:
            boxes: list[list[float]] = []
            polygons: list[list[list[float]]] = []
            source = "cached_yolo_segmentation_labels"

            if prompt in {"marine organism", "object"}:
                polygons = segment_polygons
                boxes = [polygon_box(polygon) for polygon in polygons]
            else:
                wanted_class = PROMPT_TO_CLASS[prompt]
                for annotation in anns_by_stem.get(stem, []):
                    concept = category_by_id.get(annotation["category_id"])
                    workshop_class = concept_to_workshop.get(concept)
                    if workshop_class == wanted_class:
                        box = bbox_to_normalized_xyxy(annotation["bbox"], image_info["width"], image_info["height"])
                        boxes.append(box)
                        x0, y0, x1, y1 = box
                        polygons.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                source = "cached_coco_boxes_in_sam3_like_format"

            scores = [round(max(0.51, 0.94 - 0.05 * index_value), 3) for index_value in range(len(boxes))]
            result = {
                "image_id": stem,
                "prompt": prompt,
                "source": source,
                "coordinate_mode": "normalized_xy",
                "boxes": boxes,
                "polygons": polygons,
                "scores": scores,
            }
            prompt_slug = prompt.replace(" ", "_")
            result_name = f"{stem}__{prompt_slug}.json"
            write_json(cache_dir / result_name, result)
            index["images"][stem]["prompts"][prompt] = result_name

    write_json(cache_dir / "index.json", index)
    return {"sam3_cached_images": len(index["images"])}


def build_cached_training(output_root: Path) -> dict:
    """Copy or synthesize small training-curve CSVs for fallback mode."""

    cache_root = output_root / "cached_training"
    cache_root.mkdir(parents=True, exist_ok=True)

    classification_rows = [
        {"epoch": 1, "train/loss": 2.34807, "val/loss": 2.37240, "metrics/accuracy_top1": 0.23944, "metrics/accuracy_top5": 0.59155},
        {"epoch": 2, "train/loss": 1.72468, "val/loss": 1.74512, "metrics/accuracy_top1": 0.45070, "metrics/accuracy_top5": 0.88732},
        {"epoch": 3, "train/loss": 1.43959, "val/loss": 1.59505, "metrics/accuracy_top1": 0.53521, "metrics/accuracy_top5": 0.91549},
        {"epoch": 4, "train/loss": 1.23903, "val/loss": 1.19613, "metrics/accuracy_top1": 0.64789, "metrics/accuracy_top5": 0.90141},
        {"epoch": 5, "train/loss": 1.09098, "val/loss": 1.30697, "metrics/accuracy_top1": 0.69014, "metrics/accuracy_top5": 0.91549},
        {"epoch": 6, "train/loss": 0.88288, "val/loss": 0.81433, "metrics/accuracy_top1": 0.69014, "metrics/accuracy_top5": 0.91549},
        {"epoch": 7, "train/loss": 0.74171, "val/loss": 0.92708, "metrics/accuracy_top1": 0.70423, "metrics/accuracy_top5": 0.94366},
        {"epoch": 8, "train/loss": 0.74191, "val/loss": 1.02311, "metrics/accuracy_top1": 0.73239, "metrics/accuracy_top5": 0.97183},
        {"epoch": 9, "train/loss": 0.71847, "val/loss": 0.77734, "metrics/accuracy_top1": 0.76056, "metrics/accuracy_top5": 0.94366},
        {"epoch": 10, "train/loss": 0.62716, "val/loss": 0.73226, "metrics/accuracy_top1": 0.76056, "metrics/accuracy_top5": 0.92958},
    ]
    classification_dir = cache_root / "classification"
    classification_dir.mkdir(exist_ok=True)
    with (classification_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=classification_rows[0].keys())
        writer.writeheader()
        writer.writerows(classification_rows)

    source_results = {
        "detection": SOURCE_ROOT
        / "data"
        / "trained-models"
        / "detect_binary_9951521"
        / "det_bin_megalodon_b8_lr5e4_s42"
        / "results.csv",
        "segmentation": SOURCE_ROOT
        / "data"
        / "trained-models"
        / "fn_like_yolo11x_b8_lr1e2_s0"
        / "results.csv",
    }
    copied = {"classification": True}
    for task, source_path in source_results.items():
        task_dir = cache_root / task
        task_dir.mkdir(exist_ok=True)
        if source_path.exists():
            shutil.copy2(source_path, task_dir / "results.csv")
            copied[task] = True
        else:
            copied[task] = False
    return {"cached_training": copied}


def copy_coco_subset(output_root: Path) -> dict:
    coco_dir = output_root / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SUBSET100 / "subset.json", coco_dir / "subset.json")
    data = load_json(coco_dir / "subset.json")
    return {
        "coco_images": len(data["images"]),
        "coco_annotations": len(data["annotations"]),
        "coco_categories": len(data["categories"]),
    }


def write_attribution(output_root: Path) -> dict:
    """Write a compact attribution and license note table."""

    subset = load_json(SUBSET100 / "subset.json")
    rows = []
    for image in subset["images"]:
        rows.append(
            {
                "file_name": image["file_name"],
                "source": "FathomNet-derived local tutorial subset",
                "source_url": "https://database.fathomnet.org/fathomnet/#/",
                "terms_url": "https://www.fathomnet.org/terms",
                "data_use_url": "https://www.fathomnet.org/datause",
                "license_note": "Visual content is subject to contributor-selected Creative Commons terms; intended here for ML training/development.",
            }
        )

    licenses_dir = output_root / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    with (licenses_dir / "attribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return {"attribution_rows": len(rows)}


def write_manifest(output_root: Path, stats: dict) -> None:
    manifest = {
        "name": "fathomnet_underwater_tutorial_bundle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Compact data bundle for an underwater computer vision tutorial.",
        "source_repositories": {
            "yolo_segmentation": str(SOURCE_ROOT),
            "sam3": str(SOURCE_SAM3_ROOT),
        },
        "source_urls": {
            "fathomnet_database": "https://database.fathomnet.org/fathomnet/#/",
            "fathomnet_data_use": "https://www.fathomnet.org/datause",
            "fathomnet_terms": "https://www.fathomnet.org/terms",
            "ultralytics_detect_format": "https://docs.ultralytics.com/datasets/detect/",
            "ultralytics_segment_format": "https://docs.ultralytics.com/datasets/segment/",
            "sam3_repository": "https://github.com/facebookresearch/sam3",
        },
        "tasks": {
            "classification": "classification_crops",
            "detection": "yolo_detect_binary/dataset.yaml",
            "segmentation": "yolo_segment_binary/dataset.yaml",
            "coco_inspection": "coco/subset.json",
            "sam3_fallback": "sam3_cached_outputs/index.json",
        },
        "classification_classes": sorted(stats.get("classification_source_concepts", {}).keys()),
        "classification_source_concepts": stats.get("classification_source_concepts", {}),
        "workshop_classes": sorted(set(WORKSHOP_CLASS_MAP.values())),
        "stats": stats,
    }
    write_json(output_root / "manifest.json", manifest)


def write_zip_hash_sidecar(zip_path: Path) -> Path:
    """Write a standard SHA256 sidecar for the completed bundle archive."""

    sidecar_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sidecar_path.write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    return sidecar_path


def build_bundle(
    output_root: Path,
    *,
    force: bool,
    seed: int,
    max_classification_per_class: int,
    classification_max_classes: int,
    classification_min_per_class: int,
    crop_size: int,
    image_quality: int,
) -> Path:
    """Build the data bundle and zip archive."""

    if output_root.exists() and force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stats: dict = {}
    stats.update(build_yolo_tasks(output_root, image_quality=image_quality, min_box_area=YOLO_MIN_BOX_AREA))
    stats.update(build_classification_crops(
        output_root,
        seed=seed,
        max_per_class=max_classification_per_class,
        max_classes=classification_max_classes,
        min_per_class=classification_min_per_class,
        crop_size=crop_size,
        image_quality=image_quality,
    ))
    stats.update(copy_coco_subset(output_root))
    stats.update(build_sam3_cached_outputs(output_root))
    stats.update(build_cached_training(output_root))
    stats.update(write_attribution(output_root))

    write_manifest(output_root, stats)
    zip_base = output_root.parent / output_root.name
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=output_root.parent, base_dir=output_root.name))
    sidecar_path = write_zip_hash_sidecar(zip_path)
    print(f"Wrote bundle directory: {output_root}")
    print(f"Wrote bundle zip: {zip_path}")
    print(f"Wrote zip hash: {sidecar_path}")
    print(f"Zip size: {zip_path.stat().st_size / (1024 * 1024):.1f} MiB")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the underwater tutorial data bundle.")
    parser.add_argument("--output-root", type=Path, default=Path("data/fathomnet_underwater_tutorial_bundle"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-classification-per-class", type=int, default=30)
    parser.add_argument("--classification-max-classes", type=int, default=CLASSIFICATION_MAX_CLASSES)
    parser.add_argument("--classification-min-per-class", type=int, default=CLASSIFICATION_MIN_PER_CLASS)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--image-quality", type=int, default=88)
    args = parser.parse_args()

    build_bundle(
        args.output_root.resolve(),
        force=args.force,
        seed=args.seed,
        max_classification_per_class=args.max_classification_per_class,
        classification_max_classes=args.classification_max_classes,
        classification_min_per_class=args.classification_min_per_class,
        crop_size=args.crop_size,
        image_quality=args.image_quality,
    )


if __name__ == "__main__":
    main()
