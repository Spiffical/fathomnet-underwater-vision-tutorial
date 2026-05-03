"""Generate the instructor/master version of the underwater vision notebook."""

from __future__ import annotations

import json
from pathlib import Path

from create_tutorial_notebook import build_notebook, md


STUDENT_TITLE = "# Underwater Computer Vision With FathomNet"
MASTER_TITLE = "# Underwater Computer Vision With FathomNet (Master Notebook)"


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
    set_source(
        first_cell,
        source_text(first_cell).replace(STUDENT_TITLE, MASTER_TITLE)
        + "\n\nThis instructor copy includes worked code solutions and short answer-key notes. Use the non-master notebook for the live participant version.\n",
    )


def fill_code_answers(notebook: dict) -> None:
    """Fill the executable exercise answers."""

    replace_once(
        notebook,
        '''def coco_bbox_to_yolo_exercise(coco_bbox, image_width, image_height):
    ''' + "'''" + '''Convert one COCO [x_min, y_min, width, height] box to YOLO geometry.

    Return `(center_x, center_y, width, height)`, normalized by image size.
    Leave the return value as `None` until you are ready to test your answer.
    ''' + "'''" + '''

    # TODO: fill this in during the exercise.
    return None
''',
        '''def coco_bbox_to_yolo_exercise(coco_bbox, image_width, image_height):
    ''' + "'''" + '''Convert one COCO [x_min, y_min, width, height] box to YOLO geometry.

    Return `(center_x, center_y, width, height)`, normalized by image size.
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


def add_answer_key_cells(notebook: dict) -> None:
    """Add compact answer-key notes after the main exercise prompts."""

    insert_after_heading(
        notebook,
        "## First Look At FathomNet Imagery",
        md(
            r"""
### Master Notes: First Look

- Let the room look before naming every task. Good early prompts are: "What is easy to see?", "What is small?", and "What would be hard to annotate consistently?"
- This section intentionally uses full images before crops, boxes, or masks so the modeling choices feel motivated by the visual problem.
- If a participant says the image has many unlabeled things, that is a useful opening for incomplete labels and distribution shift.
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
- Collapsing many categories into `object` removes biological identity, taxonomy, and ecological meaning, but makes a short detection exercise easier and more stable.
- Restoring tiny boxes would push the area histogram left, increase the number of labels per image, and likely lower short-run mAP for a small model because localization becomes much harder.
- The COCO subset is useful for reasoning about categories and conversion; the prepared YOLO folders are useful for short live training.
"""
        ),
    )

    insert_after_heading(
        notebook,
        "### Mini-Lab: Learning Rate As An Optimization Knob",
        md(
            r"""
### Master Notes: Learning Rate Lab

- `1e-4` is likely to look slow in a one-epoch or two-epoch workshop run.
- `1e-3` is a reasonable default for this tiny classification fine-tuning exercise.
- `1e-2` may improve quickly or become unstable, depending on batch order and initialization.
- The teaching point is not that one run proves the best learning rate. It is that optimization behavior is visible in curves, and validation accuracy is a noisy estimate on this small split.
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
        "### Mini-Lab: Error Taxonomy",
        md(
            r"""
### Master Notes: Error Taxonomy

- The most productive discussion is usually not "the model is bad," but "which failure mode is dominant?"
- Missed small objects suggest resolution, label filtering, or data quantity issues.
- Object-like background false positives suggest thresholding, hard-negative examples, or stronger domain-specific pretraining.
- Poor localization suggests label geometry, image size, or training duration.
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
- Increasing `epochs` should usually improve the tiny validation run at first, but the curve can be noisy because the validation set is intentionally small.
- Increasing `lr0` by `10x` may converge faster or destabilize; decreasing it by `10x` may make one or two epochs look almost unchanged.
- In the toy confusion matrix, `gelatinous` and `sponge_coral` are mutually confusable, and `crustacean` can leak into `fish`. Good discussion targets are transparency, partial views, posture, and texture.
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
- A whole-file COCO converter needs to group annotations by image, map category ids to contiguous YOLO class ids, normalize geometry by each image size, skip invalid boxes, and write one `.txt` label file per image.
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
- COCO polygon coordinates are normalized pairwise: `x/W`, `y/H`, preceded by the YOLO class id.
- Dropping tiny polygons often improves short workshop metrics by reducing ambiguous labels, but it can reduce ecological recall for small organisms.
- Binary object segmentation is useful as a stepping stone and for generic saliency, but it is less scientifically expressive than taxonomic or functional-group labels.
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
