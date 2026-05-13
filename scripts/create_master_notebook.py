"""Generate the instructor/master version of the underwater vision notebook."""

from __future__ import annotations

import json
from pathlib import Path

from create_tutorial_notebook import build_notebook, md


STUDENT_TITLE = "# Underwater Computer Vision With FathomNet"
MASTER_TITLE = "# Underwater Computer Vision With FathomNet (Master Notebook)"
STUDENT_COLAB_URL = (
    "https://colab.research.google.com/github/Spiffical/fathomnet-underwater-vision-tutorial/"
    "blob/main/notebooks/fathomnet_underwater_vision_tutorial.ipynb"
)
MASTER_COLAB_URL = (
    "https://colab.research.google.com/github/Spiffical/fathomnet-underwater-vision-tutorial/"
    "blob/main/notebooks/fathomnet_underwater_vision_tutorial_master.ipynb"
)


def source_text(cell: dict) -> str:
    """Return a notebook cell source as a single string."""

    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def set_source(cell: dict, text: str) -> None:
    """Set a notebook cell source using the same list-of-lines form as the tutorial."""

    cell["source"] = text.strip("\n").splitlines(keepends=True)


def replace_once(notebook: dict, old: str, new: str) -> None:
    """Replace one exact source fragment in the notebook."""

    candidates = [old]
    stripped = old.rstrip("\n")
    if stripped != old:
        candidates.append(stripped)

    for cell in notebook["cells"]:
        text = source_text(cell)
        for candidate in candidates:
            if candidate in text:
                set_source(cell, text.replace(candidate, new.rstrip("\n")))
                return
    raise ValueError(f"Could not find expected notebook fragment: {old[:80]!r}")


def replace_cell_containing(notebook: dict, marker: str, new: str) -> None:
    """Replace the whole source of the first cell containing `marker`."""

    for cell in notebook["cells"]:
        if marker in source_text(cell):
            set_source(cell, new)
            return
    raise ValueError(f"Could not find notebook cell containing: {marker!r}")


def insert_after_heading(notebook: dict, heading: str, cell: dict) -> None:
    """Insert a cell immediately after the markdown cell containing a heading."""

    for index, existing in enumerate(notebook["cells"]):
        if existing.get("cell_type") == "markdown" and heading in source_text(existing):
            notebook["cells"].insert(index + 1, cell)
            return
    raise ValueError(f"Could not find heading: {heading}")


def add_master_badge(notebook: dict) -> None:
    """Update the title and add a short instructor-facing note."""

    first_cell = notebook["cells"][0]
    first_cell_source = source_text(first_cell)
    set_source(
        first_cell,
        first_cell_source.replace(STUDENT_TITLE, MASTER_TITLE).replace(STUDENT_COLAB_URL, MASTER_COLAB_URL)
        + "\n\nThis instructor copy includes worked code solutions and short answer-key notes. Use the non-master notebook for the live participant version.\n",
    )


def fill_code_answers(notebook: dict) -> None:
    """Fill the executable exercise answers."""

    replace_once(
        notebook,
        '''def coco_bbox_to_yolo_exercise(coco_bbox, image_width, image_height):
    ''' + "'''" + '''Convert one COCO [x_min, y_min, width, height] box to YOLO geometry.

    Return `(center_x, center_y, width, height)`, normalised by image size.
    Leave the return value as `None` until you are ready to test your answer.
    ''' + "'''" + '''

    # TODO: fill this in during the exercise.
    return None
''',
        '''def coco_bbox_to_yolo_exercise(coco_bbox, image_width, image_height):
    ''' + "'''" + '''Convert one COCO [x_min, y_min, width, height] box to YOLO geometry.

    Return `(center_x, center_y, width, height)`, normalised by image size.
    ''' + "'''" + '''

    x_min, y_min, box_width, box_height = coco_bbox
    center_x = (x_min + box_width / 2) / image_width
    center_y = (y_min + box_height / 2) / image_height
    width = box_width / image_width
    height = box_height / image_height
    return (center_x, center_y, width, height)
''',
    )

    replace_once(
        notebook,
        '''if candidate_box is None:
    print("Exercise ready: implement coco_bbox_to_yolo_exercise(...) when you reach the detection lab.")
else:
    print("Your YOLO box:", candidate_box)
    assert len(candidate_box) == 4
    assert all(0 <= value <= 1 for value in candidate_box)
''',
        '''print("Your YOLO box:", candidate_box)
assert candidate_box == (0.4, 0.25, 0.4, 0.3)
assert len(candidate_box) == 4
assert all(0 <= value <= 1 for value in candidate_box)
''',
    )

    replace_once(
        notebook,
        '''def coco_polygon_to_yolo_row_exercise(class_id, polygon, image_width, image_height):
    ''' + "'''" + '''Convert one COCO polygon list to one YOLO segmentation row.

    Return a list like `[class_id, x1, y1, x2, y2, ...]`.
    Leave the return value as `None` until you are ready to test your answer.
    ''' + "'''" + '''

    # TODO: fill this in during the exercise.
    return None
''',
        '''def coco_polygon_to_yolo_row_exercise(class_id, polygon, image_width, image_height):
    ''' + "'''" + '''Convert one COCO polygon list to one YOLO segmentation row.

    Return a list like `[class_id, x1, y1, x2, y2, ...]`.
    ''' + "'''" + '''

    if len(polygon) < 6:
        raise ValueError("A YOLO segmentation polygon needs at least 3 points.")
    if len(polygon) % 2:
        raise ValueError("Polygon coordinates must be x/y pairs.")

    row = [int(class_id)]
    for index in range(0, len(polygon), 2):
        x = polygon[index] / image_width
        y = polygon[index + 1] / image_height
        row.extend([x, y])
    return row
''',
    )

    replace_once(
        notebook,
        '''if candidate_row is None:
    print("Exercise ready: implement coco_polygon_to_yolo_row_exercise(...) in the segmentation lab.")
else:
    print("Your YOLO segmentation row:", candidate_row)
    assert len(candidate_row) >= 7
    assert len(candidate_row[1:]) % 2 == 0
    assert all(0 <= value <= 1 for value in candidate_row[1:])
''',
        '''print("Your YOLO segmentation row:", candidate_row)
assert candidate_row == [0, 0.1, 0.1, 0.4, 0.1, 0.4, 0.3, 0.1, 0.3]
assert len(candidate_row) >= 7
assert len(candidate_row[1:]) % 2 == 0
assert all(0 <= value <= 1 for value in candidate_row[1:])
''',
    )

    replace_once(
        notebook,
        '''mask_a = [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
mask_b = [[1, 0, 0], [0, 1, 1], [0, 0, 0]]
print("mask IoU:", mask_iou(mask_a, mask_b))
''',
        '''mask_a = [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
mask_b = [[1, 0, 0], [0, 1, 1], [0, 0, 0]]
print("mask IoU:", mask_iou(mask_a, mask_b))

assert mask_iou([[0, 0]], [[0, 0]]) == 1.0
assert mask_iou([[1, 0]], [[0, 1]]) == 0.0
assert mask_iou([[1, 1]], [[1, 1]]) == 1.0
''',
    )

    replace_cell_containing(
        notebook,
        "def audit_yolo_split_exercise(dataset_root):",
        r'''
def audit_yolo_split_exercise(dataset_root):
    """Audit leakage and rough distribution shift for a YOLO detection dataset."""

    import statistics

    dataset_root = Path(dataset_root)

    def image_stems(split):
        image_dir = dataset_root / "images" / split
        return {
            path.stem
            for path in image_dir.glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        }

    def label_stats(split):
        label_dir = dataset_root / "labels" / split
        instance_count = 0
        box_areas = []
        instances_per_image = []
        for label_path in sorted(label_dir.glob("*.txt")):
            rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
            instances_per_image.append(len(rows))
            instance_count += len(rows)
            for row in rows:
                if len(row) == 5:
                    _, _, _, width, height = map(float, row)
                    box_areas.append(width * height)
        return {
            "instances": instance_count,
            "images_with_label_files": len(instances_per_image),
            "mean_instances_per_image": sum(instances_per_image) / len(instances_per_image) if instances_per_image else 0.0,
            "median_box_area": statistics.median(box_areas) if box_areas else None,
            "min_box_area": min(box_areas) if box_areas else None,
            "max_box_area": max(box_areas) if box_areas else None,
        }

    train_stems = image_stems("train")
    val_stems = image_stems("val")
    return {
        "train_images": len(train_stems),
        "val_images": len(val_stems),
        "overlapping_image_stems": sorted(train_stems & val_stems),
        "train": label_stats("train"),
        "val": label_stats("val"),
    }


split_audit = audit_yolo_split_exercise(detect_paths["root"])
print(json.dumps(split_audit, indent=2))
''',
    )

    replace_cell_containing(
        notebook,
        "def build_confusion_matrix_from_predictions(model_or_path, val_root):",
        r'''
RUN_CLASSIFICATION_CONFUSION_ADVANCED = False

def build_confusion_matrix_from_predictions(model_or_path, val_root):
    """Predict validation crops and return `(matrix, class_names)`."""

    from sklearn.metrics import confusion_matrix
    from ultralytics import YOLO

    val_root = Path(val_root)
    class_dirs = sorted(path for path in val_root.iterdir() if path.is_dir())
    class_names = [path.name for path in class_dirs]
    class_to_index = {name: index for index, name in enumerate(class_names)}

    model = YOLO(str(model_or_path))
    y_true = []
    y_pred = []

    for true_index, class_dir in enumerate(class_dirs):
        for image_path in sorted(class_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            result = model.predict(source=str(image_path), imgsz=224, verbose=False)[0]
            predicted_index = int(result.probs.top1)
            predicted_name = result.names.get(predicted_index, str(predicted_index))
            if predicted_name in class_to_index:
                predicted_index = class_to_index[predicted_name]
            if 0 <= predicted_index < len(class_names):
                y_true.append(true_index)
                y_pred.append(predicted_index)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return matrix, class_names


if RUN_CLASSIFICATION_CONFUSION_ADVANCED:
    model_path = classification_best_model_path
    if model_path is None:
        print("Run live classification training first so `classification_best_model_path` exists.")
    else:
        matrix, class_names = build_confusion_matrix_from_predictions(model_path, CLASSIFY_ROOT / "val")
        plot_confusion_matrix(matrix, class_names, normalize=True, title="Validation confusion matrix")
else:
    print("Advanced confusion-matrix exercise is ready.")
    print("Set RUN_CLASSIFICATION_CONFUSION_ADVANCED = True after implementing the helper.")
''',
    )

    replace_cell_containing(
        notebook,
        "def yolo_polygon_row_to_mask(row, image_width, image_height):",
        r'''
def yolo_polygon_row_to_mask(row, image_width, image_height):
    """Rasterise one YOLO segmentation polygon row into a binary mask."""

    import numpy as np
    from PIL import Image, ImageDraw

    coords = row[1:]
    if len(coords) < 6:
        raise ValueError("A polygon needs at least 3 points.")
    if len(coords) % 2:
        raise ValueError("Polygon coordinates must be x/y pairs.")

    points = [
        (coords[index] * image_width, coords[index + 1] * image_height)
        for index in range(0, len(coords), 2)
    ]
    mask_image = Image.new("L", (int(image_width), int(image_height)), 0)
    ImageDraw.Draw(mask_image).polygon(points, outline=1, fill=1)
    return np.asarray(mask_image, dtype=bool)


example_segment_row = [float(part) for part in first_segment_label.read_text().splitlines()[0].split()]
from PIL import Image

with Image.open(first_segment_image) as example_segment_image:
    example_width, example_height = example_segment_image.size

candidate_mask = yolo_polygon_row_to_mask(example_segment_row, example_width, example_height)
import numpy as np
import matplotlib.pyplot as plt

candidate_mask = np.asarray(candidate_mask).astype(bool)
print("mask shape:", candidate_mask.shape, "mask pixels:", int(candidate_mask.sum()))
plt.figure(figsize=(6, 4))
plt.imshow(candidate_mask, cmap="gray")
plt.axis("off")
plt.title("Rasterised YOLO polygon")
plt.show()
''',
    )

    replace_cell_containing(
        notebook,
        "def sam3_result_to_yolo_pseudo_labels(result, class_id=0, score_threshold=0.5):",
        r'''
def sam3_result_to_yolo_pseudo_labels(result, class_id=0, score_threshold=0.5):
    """Convert SAM3-style polygons into YOLO segmentation pseudo-label rows."""

    rows = []
    scores = result.get("scores", [])
    for index, polygon in enumerate(result.get("polygons", [])):
        score = float(scores[index]) if index < len(scores) else 1.0
        if score < score_threshold:
            continue
        row = [int(class_id)]
        for x, y in polygon:
            row.extend([float(x), float(y)])
        if len(row) >= 7:
            rows.append(row)
    return rows


pseudo_label_rows = sam3_result_to_yolo_pseudo_labels(
    sam3_result,
    class_id=0,
    score_threshold=CONFIDENCE_THRESHOLD,
)

print(f"pseudo-label rows: {len(pseudo_label_rows)}")
for row in pseudo_label_rows[:3]:
    print(row)
''',
    )

    replace_cell_containing(
        notebook,
        "RUN_MEGALODON_ADVANCED = False",
        r'''
RUN_MEGALODON_ADVANCED = False

MEGALODON_REPO_ID = "FathomNet/megalodon"
MEGALODON_FILENAME = "mbari-megalodon-yolov8x.pt"
MEGALODON_MODEL_PATH = None

if RUN_MEGALODON_ADVANCED:
    from huggingface_hub import hf_hub_download

    MEGALODON_MODEL_PATH = Path(
        hf_hub_download(
            repo_id=MEGALODON_REPO_ID,
            filename=MEGALODON_FILENAME,
            cache_dir=REPO_ROOT / ".cache" / "huggingface",
        )
    )
    print(f"Megalodon weights: {MEGALODON_MODEL_PATH}")
else:
    print("Advanced Megalodon path is off by default.")
    print("Set RUN_MEGALODON_ADVANCED = True to download the FathomNet checkpoint.")
''',
    )

    replace_cell_containing(
        notebook,
        "RUN_MEGALODON_PREDICT = False",
        r'''
RUN_MEGALODON_PREDICT = False

if RUN_MEGALODON_PREDICT and MEGALODON_MODEL_PATH is not None:
    from ultralytics import YOLO
    import matplotlib.pyplot as plt

    megalodon_model = YOLO(str(MEGALODON_MODEL_PATH))
    megalodon_prediction = megalodon_model.predict(
        source=str(first_detect_image),
        imgsz=640,
        conf=0.25,
        save=False,
        project=REPO_ROOT / "runs" / "megalodon",
        name="predict_before_finetune",
        exist_ok=True,
        verbose=False,
    )[0]

    plt.figure(figsize=(8, 5))
    plt.imshow(megalodon_prediction.plot()[..., ::-1])
    plt.axis("off")
    plt.title("Megalodon prediction before fine-tuning")
    plt.show()
else:
    print("Prediction exercise ready.")
    print("Set RUN_MEGALODON_PREDICT = True after you have downloaded the checkpoint.")
''',
    )

    replace_cell_containing(
        notebook,
        "RUN_MEGALODON_FINE_TUNE = False",
        r'''
RUN_MEGALODON_FINE_TUNE = False

if RUN_MEGALODON_ADVANCED and RUN_MEGALODON_FINE_TUNE and RUN_LIVE_TRAINING and MEGALODON_MODEL_PATH is not None:
    from ultralytics import YOLO

    megalodon_finetune_args = build_train_args(
        n_epochs=2,
        imgsz=640,
        batch=2,
        lr0=0.0005,
        patience=5,
        project=REPO_ROOT / "runs" / "megalodon",
        name="finetune_tutorial_binary",
    )

    megalodon_finetune_model = YOLO(str(MEGALODON_MODEL_PATH))
    megalodon_finetune_result = megalodon_finetune_model.train(
        data=str(DETECT_YAML),
        **megalodon_finetune_args,
    )
    megalodon_finetune_save_dir = (
        getattr(megalodon_finetune_result, "save_dir", None)
        or getattr(megalodon_finetune_model.trainer, "save_dir", None)
    )
    plot_training_curves(
        Path(megalodon_finetune_save_dir) / "results.csv",
        metric_columns=["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
        include_training=False,
        title="Megalodon fine-tuning metrics",
    )
else:
    print("Fine-tuning scaffold is ready.")
    print("Set RUN_MEGALODON_ADVANCED = True, RUN_MEGALODON_FINE_TUNE = True, and use a GPU.")
''',
    )


def add_answer_key_cells(notebook: dict) -> None:
    """Add compact answer-key notes after the main exercise prompts."""

    insert_after_heading(
        notebook,
        "## Notebook Map",
        md(
            r"""
### Suggested Instructor Timing

For a three-hour live session, a reasonable pacing target is:

1. `0-15 min`: setup, runtime check, and first look at FathomNet imagery.
2. `15-30 min`: YOLO warm-up on a familiar image.
3. `30-50 min`: dataset and annotation exploration.
4. `50-75 min`: classification from organism crops.
5. `75-115 min`: binary object detection with YOLO.
6. `115-150 min`: instance segmentation with YOLO.
7. `150-175 min`: SAM3 text-prompt segmentation.
8. `175-180 min`: wrap-up and extension ideas.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "## First Look At FathomNet Imagery",
        md(
            r"""
### Master Notes: First Look

- Let participants look before naming every task. Good early prompts are: "What is easy to see?", "What is small?", and "What would be hard to annotate consistently?"
- This section intentionally uses full images before crops, boxes, or masks so the modelling choices feel motivated by the visual problem.
- If a participant says the image has many unlabelled things, that is a useful opening for incomplete labels and distribution shift.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### YOLO Warm-Up Exercises",
        md(
            r"""
### Master Notes: YOLO Warm-Up

- Raising `YOLO_WARMUP_CONF` from `0.25` to `0.6` should remove lower-confidence detections first; this is a thresholding operation after the model has already scored candidate boxes.
- `bus.jpg` is still detection: the model predicts classes and boxes for multiple objects in one image.
- A generic COCO-pretrained detector often misses underwater organisms because many taxa are outside its label vocabulary, and it may hallucinate familiar COCO classes from shapes or textures.
- This is the same basic pattern as the Ultralytics tutorial: `YOLO(weights)`, then `.predict(...)`, then inspect results.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Dataset Exploration Questions",
        md(
            r"""
### Master Notes: Dataset Exploration

- Good obvious examples usually have high contrast, a large organism, and a clean background. Subtle examples often involve small objects, partial bodies, or organism-background camouflage.
- The visual views are intentionally distinct: full images with category labels show "what is present," classification crops show one source-level concept label per crop, detection labels show multiple boxes per full image, and segmentation labels show multiple object-shaped regions per full image.
- Keep this section about truth labels only. Save model predictions for the YOLO warm-up and training/evaluation sections.
- Collapsing many categories into `object` removes biological identity, taxonomy, and ecological meaning, but makes a short detection exercise easier and more stable.
- Restoring tiny boxes would push the area histogram left, increase the number of labels per image, and likely lower short-run mAP for a small model because localisation becomes much harder.
- The COCO subset is useful for reasoning about categories and conversion; the prepared YOLO folders are useful for short live training.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Advanced Exercise: Audit The Train/Validation Split",
        md(
            r"""
### Master Notes: Split Audit

- The expected overlap between train and validation image stems is zero.
- This audit is intentionally simple: leakage, object counts, and box-size summaries catch many common split problems.
- A representative validation set matters more than a polished training curve. If validation differs sharply from training, mAP can mislead.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Mini-Lab: Learning Rate As An Optimisation Knob",
        md(
            r"""
### Master Notes: Learning Rate Lab

- `1e-4` is likely to look slow in a one-epoch or two-epoch workshop run.
- `1e-3` is a reasonable default for this tiny classification fine-tuning exercise.
- `1e-2` may improve quickly or become unstable, depending on batch order and initialisation.
- The teaching point is not that one run proves the best learning rate. It is that optimisation behaviour is visible in curves, and validation accuracy is a noisy estimate on this small split.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Mini-Lab: Matching Predictions To Labels",
        md(
            r"""
### Master Notes: Matching Predictions

- At `conf >= 0.25`, the toy example keeps all three predictions: one true positive and two false positives, so precision is `1/3` and recall is `1`.
- At `conf >= 0.5`, it keeps the first two predictions: one true positive and one false positive, so precision is `1/2` and recall is `1`.
- At `conf >= 0.8`, it keeps only the best prediction: one true positive, zero false positives, so precision is `1` and recall is `1`.
- In real multi-object AP, matching is repeated over many objects and thresholds, but this one-object case is the core logic.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Mini-Lab: Confidence Threshold Sweep",
        md(
            r"""
### Master Notes: Threshold Sweep

- Lower thresholds tend to increase the number of detections and increase recall opportunities.
- Higher thresholds tend to suppress weak boxes, which often improves precision but can remove real objects.
- The model weights do not change during this lab. Only the decision rule changes.
- If the generic COCO detector finds little in underwater images, that is itself a domain-shift result.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Mini-Lab: Overfit A Tiny Dataset",
        md(
            r"""
### Master Notes: Tiny Overfit

- A successful tiny-overfit run should drive training loss down sharply.
- Validation mAP may remain noisy because the validation set is also tiny and may contain different-looking objects.
- If training loss does not drop, first suspect data paths, label format, learning rate, batch size, or whether the labels actually align with the images.
- This is a debugging test, not a final model-quality test.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Advanced Exercise: Start From FathomNet Megalodon",
        md(
            r"""
### Master Notes: Megalodon

- This should be framed as an advanced optional path. The checkpoint is much larger than `yolo11n.pt`, and YOLOv8x fine-tuning can be slow on small GPUs.
- The useful comparison is not only metric-vs-metric. Ask whether Megalodon finds underwater object-like regions that generic COCO weights miss.
- Keep `n_epochs` modest at first. The first fine-tuning question is whether the run is wired correctly and whether early validation behaviour looks plausible.
- Because Megalodon is already a one-class FathomNet detector, it is a good match for the tutorial's binary `object` labels.
- The participant notebook intentionally gives hints rather than complete code here. For a quick solution, use `hf_hub_download(...)`, then `YOLO(str(path)).predict(...)`, then `YOLO(str(path)).train(data=str(DETECT_YAML), ...)`.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Mini-Lab: Error Taxonomy",
        md(
            r"""
### Master Notes: Error Taxonomy

- The most productive discussion is usually not "the model is bad," but "which failure mode is dominant?"
- Missed small objects suggest resolution, label filtering, or data quantity issues.
- Object-like background false positives suggest thresholding, hard-negative examples, or stronger domain-specific pretraining.
- Poor localisation suggests label geometry, image size, or training duration.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Classification Exercises",
        md(
            r"""
### Master Notes: Classification

- The direct Ultralytics pattern is `from ultralytics import YOLO`, `model = YOLO("yolo11n-cls.pt")`, then `model.train(data=str(CLASSIFY_ROOT), ...)`.
- Increasing `n_epochs` should usually improve the tiny validation run at first, but the curve can be noisy because the validation set is intentionally small.
- In a 5-epoch GPU sweep with this 12-class crop bundle, `lr0=1e-4` reached about `0.61` top-1, `lr0=1e-3` peaked near `0.69` top-1, and `lr0=1e-2` reached about `0.58` top-1. The exact values can move, but the pattern is useful.
- In the toy confusion matrix, adjacent off-diagonal entries are deliberate. Good discussion targets are visual similarity, partial views, label granularity, and the difference between a true biological ambiguity and a model error.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Advanced Exercise: Build A Confusion Matrix From Predictions",
        md(
            r"""
### Master Notes: Prediction Confusion Matrix

- Keep this advanced exercise gated until a trained classification checkpoint exists.
- If a generic ImageNet classifier is used instead of the tutorial-trained checkpoint, predicted class names may not match the tutorial's FathomNet concept folders.
- The useful teaching point is the mapping from validation folder names to true labels and from model outputs to predicted labels.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Detection Exercises",
        md(
            r"""
### Master Notes: Detection

- The two essential training lines are `model = YOLO("yolo11n.pt")` and `model.train(data=str(DETECT_YAML), **DETECT_ARGS)`.
- COCO `[x_min, y_min, width, height]` maps to YOLO `(x_min + width/2)/W`, `(y_min + height/2)/H`, `width/W`, `height/H`.
- A more conservative confidence threshold usually raises precision and lowers recall.
- Raising `imgsz` can help small objects because the resized image preserves more spatial detail, but it costs more memory and time.
- A whole-file COCO converter needs to group annotations by image, map category ids to contiguous YOLO class ids, normalise geometry by each image size, skip invalid boxes, and write one `.txt` label file per image.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Segmentation Exercises",
        md(
            r"""
### Master Notes: Segmentation

- YOLO segmentation requires at least three `(x, y)` polygon points per object.
- Box mAP can be higher than mask mAP because a predicted rectangle can be acceptable even when the boundary shape is poor.
- COCO polygon coordinates are normalised pairwise: `x/W`, `y/H`, preceded by the YOLO class id.
- Dropping tiny polygons often improves short workshop metrics by reducing ambiguous labels, but it can reduce ecological recall for small organisms.
- Binary object segmentation is useful as a stepping stone and for generic saliency, but it is less scientifically expressive than taxonomic or functional-group labels.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Advanced Exercise: Rasterise A YOLO Polygon",
        md(
            r"""
### Master Notes: Rasterization

- This exercise connects the set notation for masks to the polygon text file format.
- Rasterization introduces pixel-grid choices: rounding, boundary inclusion, and holes can all matter in production.
- Comparing mask area to bounding-box area helps explain why mask metrics and box metrics need not move together.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### SAM3 Exercises",
        md(
            r"""
### Master Notes: SAM3

- Broad prompts such as `marine organism` tend to increase recall and false positives.
- Specific prompts such as `small crab` can improve precision when the concept is visually present, but often return nothing when the image lacks the target or the prompt misses the model's visual vocabulary.
- Raising `CONFIDENCE_THRESHOLD` filters low-score masks; expect fewer detections, higher apparent precision, and lower recall.
- SAM3 pseudo-labels are useful when they reduce annotation effort, but noisy pseudo-labels can teach a supervised model the prompt model's mistakes.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Advanced Exercise: Export SAM3 Pseudo-Labels",
        md(
            r"""
### Master Notes: SAM3 Pseudo-Labels

- The cached polygons are normalised, so the conversion to YOLO rows is mostly filtering and flattening.
- The hard part is not the file format; it is deciding whether prompt-generated labels are trustworthy enough to train on.
- Good discussion prompt: which pseudo-label errors would a supervised model amplify?
"""
        ),
    )


def rebuild_cell_ids(notebook: dict) -> None:
    """Give generated cells stable ids after insertions."""

    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"master-cell-{index:03d}"


def build_master_notebook() -> dict:
    """Build the master notebook from the student notebook definition."""

    notebook = build_notebook()
    add_master_badge(notebook)
    fill_code_answers(notebook)
    add_answer_key_cells(notebook)
    rebuild_cell_ids(notebook)
    return notebook


def main() -> None:
    out_path = Path("notebooks/fathomnet_underwater_vision_tutorial_master.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(build_master_notebook(), handle, indent=1)
        handle.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
