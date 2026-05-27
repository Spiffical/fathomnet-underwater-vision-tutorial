# Underwater Computer Vision Tutorial: Exercise Answer Key

This document collects the beginner, intermediate, and advanced prompts from the tutorial notebook, together with suggested answers and worked code where appropriate. It is intended as an instructor reference, not as participant-facing material.

Student notebook: `notebooks/fathomnet_underwater_vision_tutorial.ipynb`  
Master notebook: `notebooks/fathomnet_underwater_vision_tutorial_master.ipynb`

## Table Of Contents

- [YOLO Warm-Up](#yolo-warm-up)
- [Dataset Exploration](#dataset-exploration)
- [Part 1: Classification](#part-1-classification)
- [Part 2: Object Detection](#part-2-object-detection)
- [Part 3: Instance Segmentation](#part-3-instance-segmentation)
- [Part 4: SAM3 Promptable Segmentation](#part-4-sam3-promptable-segmentation)

## YOLO Warm-Up

### Beginner

**Question:** Change `YOLO_WARMUP_CONF` from `0.25` to `0.6`. Which detections disappear first?

**Answer:** Lower-confidence detections disappear first. The model has already proposed and scored candidate boxes; changing `YOLO_WARMUP_CONF` only changes which scored predictions are displayed. This is a thresholding decision, not retraining.

### Intermediate

**Question:** Change `YOLO_WARMUP_SOURCE` to `https://ultralytics.com/images/bus.jpg`. Read the printed class names and decide whether the model is solving classification, detection, or segmentation.

**Answer:** It is solving object detection. The model predicts both a class label and a bounding box for each detected object. If it were classification, it would return one image-level label. If it were segmentation, it would return object masks or polygons.

### Advanced

**Question:** Run the same pretrained model on one underwater validation image before fine-tuning. What does the model miss, and what does it hallucinate?

**Answer:** A generic COCO-pretrained model usually misses many underwater organisms because they are outside its training vocabulary and visual domain. It may hallucinate familiar COCO classes from shape, texture, or scene context. The useful teaching point is domain shift: the model has learned useful visual features, but its label space and priors are not marine-science-specific.

## Dataset Exploration

### Beginner

**Question:** Pick one image where the object is visually obvious and one where it is subtle. Which source of difficulty is most visible: contrast, scale, clutter, partial view, or taxonomy?

**Answer:** Good obvious examples usually have high contrast, a large organism, and a clean background. Subtle examples often involve small organisms, low contrast, partial bodies, camouflage, clutter, or ambiguous taxonomy. Any well-justified visual explanation is acceptable.

### Intermediate

**Question:** Compare the `coco/subset.json` category counts with the binary YOLO labels. What information is lost when many biological categories are collapsed into `object`?

**Answer:** Collapsing labels into `object` removes biological identity, taxonomic resolution, ecological meaning, and information about which organism groups are being detected. It makes the short detection exercise easier and more stable, but it no longer answers species- or group-level scientific questions.

**Question:** Compare the category-only full-image view with the box and mask views. What extra information do you gain when the truth labels include geometry?

**Answer:** Category-only labels say what is present somewhere in the image. Boxes add approximate location and scale. Masks add object shape and pixel-level extent. Geometry makes it possible to train localisation models, count object instances, separate neighbouring organisms, and reason about size, coverage, and interactions with habitat.

### Advanced

**Question:** The live YOLO labels drop extremely tiny boxes. Predict how the histogram and mAP would change if those boxes were restored.

**Answer:** The object-area histogram would shift left and become more heavily concentrated near zero. Short-run mAP would likely decrease for a small model because tiny-object localisation is harder, annotation noise matters more, and small errors can cause IoU to fall below threshold. Recall for tiny organisms could improve only if the model and image resolution are adequate.

**Question:** The segmentation labels contain more geometry than boxes. Where would that extra information matter scientifically?

**Answer:** Masks matter when shape, coverage, contact with substrate, overlap, or pixel-level area is scientifically meaningful. Examples include estimating organism area, habitat coverage, benthic community structure, morphology, or avoiding double-counting objects that are close together.

### Advanced Coding: Audit The Train/Validation Split

**Question:** Implement `audit_yolo_split_exercise(...)` so it reports image counts, train/validation overlap, object instances per split, and median box area per split.

**Worked answer:**

```python
def audit_yolo_split_exercise(dataset_root):
    """Audit leakage and rough distribution shift for a YOLO detection dataset."""

    import statistics
    from pathlib import Path

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
            "mean_instances_per_image": sum(instances_per_image) / len(instances_per_image)
            if instances_per_image
            else 0.0,
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
```

**Expected audit for the current bundle:** `80` train images, `20` validation images, `0` overlapping image stems, `92` train instances, and `31` validation instances. The median normalised box areas are approximately `0.0196` for train and `0.0283` for validation.

## Part 1: Classification

### Mini-Lab: Learning Rate As An Optimisation Knob

**Question:** Try one or more learning rates from `lr0 = 1e-4`, `1e-3`, and `1e-2`. Decide whether the curves look slow, useful, or unstable.

**Answer:** `1e-4` will often move slowly in a short run. `1e-3` is a reasonable default for this small fine-tuning task. `1e-2` can improve quickly or become unstable, depending on batch order and the run. The goal is not to prove a universal best learning rate; it is to recognise optimisation behaviour in the training and validation curves.

### Confusion Matrix Reading Practice

**Question:** Which classes are most confusable in the toy confusion matrix, and what could explain those mistakes?

**Answer:** The off-diagonal entries show the confusions. Adjacent off-diagonal errors are deliberately inserted for discussion. Good explanations include visual similarity, partial views, low contrast, label granularity, and the difference between true biological ambiguity and model error.

### Beginner

**Question:** Change `CLASSIFY_N_EPOCHS` from `10` to another value, then rerun the training cell. Decide whether the validation curve is improving, noisy, or overfitting.

**Answer:** More epochs usually improve the small validation run at first, but the curve can be noisy because the validation split is small. Overfitting is suggested when training loss keeps improving while validation accuracy plateaus or degrades.

**Question:** Run or reason through one learning-rate trial.

**Answer:** The expected answer is a qualitative read of the curve: too small moves slowly, useful improves steadily, and too large may jump around or degrade.

### Intermediate

**Question:** Use the Ultralytics classification docs to find the direct `YOLO(...).train(...)` pattern for classification.

**Answer:**

```python
from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
model.train(data=str(CLASSIFY_ROOT), epochs=CLASSIFY_N_EPOCHS, imgsz=224)
```

The tutorial passes most arguments through `build_train_args(...)`, but the underlying Ultralytics pattern is `YOLO(weights)` followed by `.train(...)`.

**Question:** Compare one `lr0` result with another `lr0` result.

**Answer:** In a previous 5-epoch GPU sweep with the 12-class crop bundle, `lr0=1e-4` reached about `0.61` top-1 accuracy, `lr0=1e-3` peaked near `0.69`, and `lr0=1e-2` reached about `0.58`. Exact values can move between runs; the useful result is the relative curve behaviour.

**Question:** Change `imgsz` from `224` to `320`. Was the extra computation worth it?

**Answer:** It is only worth it if validation accuracy improves enough to justify the added memory and time. Larger crops may preserve useful detail, but for a tiny dataset the validation estimate can be noisy.

**Question:** Pick one class and inspect three crops that seem visually ambiguous.

**Answer:** A good response names the visual source of ambiguity: partial organism, similar texture, low contrast, small crop, taxonomic similarity, or crop contamination from background.

### Advanced

**Question:** Design a fair pretrained-vs-random-initialisation comparison. What would you keep fixed?

**Answer:** Keep the train/validation split, architecture, image size, epochs, batch size, optimiser, learning rate schedule, augmentations, random seed strategy, and evaluation code fixed. Change only the initial weights. Ideally repeat several seeds because small validation sets are noisy.

**Question:** Complete `build_confusion_matrix_from_predictions(...)` and replace the toy matrix with predictions from your live model.

**Worked answer:**

```python
def build_confusion_matrix_from_predictions(model_or_path, val_root):
    """Predict validation crops and return `(matrix, class_names)`."""

    from pathlib import Path
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
```

**Question:** Which two classes are most confusable, and what visual ambiguity might explain it?

**Answer:** Use the largest off-diagonal entries in the real confusion matrix. A strong answer connects the confusion to visual evidence, such as shared texture, similar morphology, partial views, or label granularity.

**Question:** Propose a better class grouping for this small crop dataset.

**Answer:** A useful grouping should merge visually or biologically related classes while preserving a scientific purpose. For example, group by broad morphology or functional group rather than fine taxonomy if the small crop dataset cannot support fine-grained labels.

## Part 2: Object Detection

### Mini-Lab: Matching Predictions To Labels

**Question:** Complete the toy matching exercise: what are TP, FP, and FN at each confidence threshold?

**Answer:**

- `conf >= 0.25`: all three predictions are kept. One matches the ground truth, two do not. `TP=1`, `FP=2`, `FN=0`, precision `1/3`, recall `1`.
- `conf >= 0.50`: two predictions are kept. One matches and one does not. `TP=1`, `FP=1`, `FN=0`, precision `1/2`, recall `1`.
- `conf >= 0.80`: only the best prediction is kept. It matches. `TP=1`, `FP=0`, `FN=0`, precision `1`, recall `1`.

### Mini-Lab: Confidence Threshold Sweep

**Question:** Describe what changes from `0.10` to `0.80` on the training image versus the validation image.

**Answer:** The model weights do not change. Lower thresholds keep more candidate detections, increasing recall opportunities but usually adding false positives. Higher thresholds suppress weak boxes, often improving apparent precision while missing more real objects. Comparing a training image with a validation image helps distinguish memorisation from generalisation.

### Mini-Lab: Overfit A Tiny Dataset

**Question:** Turn on `RUN_TINY_OVERFIT_LAB` and check whether the model can memorise four large-object training images.

**Answer:** A successful tiny-overfit run should drive training loss down sharply and improve same-image validation mAP. In this lab, validation deliberately uses the same images as training, so the metric is a memorisation sanity check, not a generalisation estimate. If it fails, first suspect data paths, label format, label-image alignment, learning rate, batch size, or augmentation settings.

### Advanced Megalodon Path

**Question:** Download FathomNet Megalodon from Hugging Face.

**Worked answer:**

```python
from pathlib import Path
from huggingface_hub import hf_hub_download

MEGALODON_MODEL_PATH = Path(
    hf_hub_download(
        repo_id="FathomNet/megalodon",
        filename="mbari-megalodon-yolov8x.pt",
        cache_dir=REPO_ROOT / ".cache" / "huggingface",
    )
)
```

**Question:** Load Megalodon and predict on `first_detect_image`.

**Worked answer:**

```python
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
```

**Question:** Fine-tune Megalodon briefly on `DETECT_YAML`.

**Worked answer:**

```python
from pathlib import Path
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
    metric_columns=[
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ],
    include_training=False,
    title="Megalodon fine-tuning metrics",
)
```

### COCO Boxes To YOLO Boxes

**Question:** Complete `coco_bbox_to_yolo_exercise(...)`.

**Worked answer:**

```python
def coco_bbox_to_yolo_exercise(coco_bbox, image_width, image_height):
    """Convert one COCO [x_min, y_min, width, height] box to YOLO geometry."""

    x_min, y_min, box_width, box_height = coco_bbox
    center_x = (x_min + box_width / 2) / image_width
    center_y = (y_min + box_height / 2) / image_height
    width = box_width / image_width
    height = box_height / image_height
    return (center_x, center_y, width, height)
```

For `example_coco_bbox = [20, 10, 40, 30]` in a `100 x 100` image, the answer is `(0.4, 0.25, 0.4, 0.3)`.

### Error Taxonomy

**Question:** Assign one error category to inspected predictions and explain the likely cause.

**Answer:** A strong answer names a failure mode and a modelling response. Missed small objects suggest higher resolution, better labels, or more small-object examples. Object-like background false positives suggest hard negatives or domain-specific pretraining. Poor localisation suggests label geometry, image size, or training duration. Duplicate detections suggest non-maximum suppression or threshold tuning.

### Beginner

**Question:** Use the Ultralytics training guide to identify the two lines that load a model and start training.

**Answer:**

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data=str(DETECT_YAML), **DETECT_ARGS)
```

### Intermediate

**Question:** Change `imgsz` from `640` to `320`, then compare speed and validation metrics.

**Answer:** Smaller `imgsz` should usually train faster and use less memory. It may hurt small-object localisation because fewer pixels represent each organism. Larger `imgsz` can help small objects but costs time and memory.

**Question:** Change `DETECT_N_EPOCHS`, `lr0`, or `batch`, then compare `mAP50` and recall.

**Answer:** More epochs can help until overfitting or saturation. Learning rate changes optimisation speed and stability. Batch size affects memory, gradient noise, and throughput. The answer should compare curves, not just one final number.

### Advanced

**Question:** Explain how the precision-recall curve would move if the model became more conservative.

**Answer:** More conservative scoring or thresholding usually reduces recall because fewer predictions are kept. Precision may improve if the removed predictions are mostly false positives. Across the full precision-recall curve, a genuinely better conservative model would push the curve upward; simply raising one threshold moves to a different point on the same curve.

**Question:** Sketch how you would convert the whole `coco/subset.json` file into YOLO detection labels.

**Answer:** Parse images, categories, and annotations; map category ids to contiguous YOLO class ids; group annotations by image; normalise each COCO box by that image's width and height; skip invalid or unwanted boxes; write one `.txt` file per image; copy or symlink images into `images/train`, `images/val`, and optionally `images/test`; write a dataset YAML with paths, class count, and class names.

## Part 3: Instance Segmentation

### Mask IoU

**Question:** Complete `mask_iou(...)` for one extra edge case: two empty masks, two disjoint masks, or two identical masks.

**Answer:** The current helper returns `1.0` for two empty masks, `0.0` for disjoint masks, and `1.0` for identical masks. Those are reasonable conventions for this exercise.

```python
assert mask_iou([[0, 0]], [[0, 0]]) == 1.0
assert mask_iou([[1, 0]], [[0, 1]]) == 0.0
assert mask_iou([[1, 1]], [[1, 1]]) == 1.0
```

### Advanced Coding: Rasterise A YOLO Polygon

**Question:** Complete `yolo_polygon_row_to_mask(...)`.

**Worked answer:**

```python
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
```

### COCO Polygons To YOLO Segmentation Rows

**Question:** Complete `coco_polygon_to_yolo_row_exercise(...)`.

**Worked answer:**

```python
def coco_polygon_to_yolo_row_exercise(class_id, polygon, image_width, image_height):
    """Convert one COCO polygon list to one YOLO segmentation row."""

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
```

For `example_polygon = [10, 10, 40, 10, 40, 30, 10, 30]` in a `100 x 100` image, the answer is:

```python
[0, 0.1, 0.1, 0.4, 0.1, 0.4, 0.3, 0.1, 0.3]
```

### Beginner

**Question:** Inspect two masks: one clean and one messy.

**Answer:** A clean mask should follow the visible object boundary reasonably well. A messy mask may include background, miss transparent or thin structures, merge neighbouring objects, or reflect ambiguous annotation.

**Question:** Why can box mAP be higher than mask mAP?

**Answer:** A rectangle can localise an object roughly even when the predicted boundary is poor. Mask mAP is stricter about shape and pixel coverage, so it can lag behind box mAP.

**Question:** Use the Ultralytics segmentation docs to identify the minimum number of polygon points per object.

**Answer:** A polygon needs at least three `(x, y)` points, meaning at least six geometry numbers after the class id.

### Intermediate

**Question:** Change the confidence threshold and inspect the visual result.

**Answer:** Higher confidence thresholds should produce fewer masks, usually removing low-score false positives but also missing lower-confidence true objects. Lower thresholds produce more masks and more potential false positives.

### Advanced

**Question:** Compare the rasterised mask area with the polygon's bounding-box area.

**Answer:** The rasterised mask area should usually be less than or equal to the bounding-box area. The gap is large for thin, curved, or irregular organisms. This explains why box metrics and mask metrics can tell different stories.

**Question:** Is binary `"object"` segmentation scientifically useful, or only a stepping stone?

**Answer:** It can be useful for generic saliency, annotation triage, object counting, and pseudo-label generation. For many science questions, it is only a stepping stone because it omits taxonomy, functional group, and ecological identity.

**Question:** Propose a rule for dropping or keeping tiny polygons, then predict how that rule will affect recall and mAP.

**Answer:** A rule might keep polygons above a minimum area fraction or visible-pixel threshold. Dropping tiny polygons often improves short-run mAP by removing hard, noisy labels, but it reduces recall for small organisms and can bias the model toward large, obvious objects.

## Part 4: SAM3 Promptable Segmentation

### Beginner

**Question:** Try `PROMPT = "fish"`, `"sponge"`, `"gelatinous animal"`, and `"small crab"`. Which prompts are too broad? Which are too specific?

**Answer:** Broad prompts such as `"marine organism"` tend to return more masks and more false positives. Specific prompts such as `"small crab"` can be precise when the target is present but may return nothing when the object is absent, visually unusual, or outside the prompt vocabulary. The correct answer depends on the chosen image.

**Question:** Increase `CONFIDENCE_THRESHOLD` from `0.5` to `0.8`.

**Answer:** Fewer masks should be displayed. This can improve apparent precision by filtering weak detections, but it can lower recall by removing real lower-score objects.

### Advanced

**Question:** If live SAM3 works, compare cached fallback with live masks.

**Answer:** Live and cached outputs may differ because package versions, checkpoints, thresholds, and prompt handling can change. The comparison should focus on whether the same objects are found, whether boundaries are similar, and whether the prompt produces false positives or misses.

**Question:** Complete `sam3_result_to_yolo_pseudo_labels(...)`, then reason about when pseudo-label noise helps or hurts supervised training.

**Worked answer:**

```python
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
```

**Pseudo-label discussion answer:** Pseudo-labels help when they add many reasonably accurate labels that are expensive for humans to draw. They hurt when systematic SAM3 errors are treated as truth; the supervised model can then learn and amplify those mistakes. Human review, score thresholds, active learning, and spot checks are important safeguards.
