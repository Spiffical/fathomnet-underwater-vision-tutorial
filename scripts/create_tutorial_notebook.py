"""Generate the underwater computer vision tutorial notebook."""

from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def build_notebook() -> dict:
    def section_bootstrap(section: str) -> str:
        return f"""
from scripts.tutorial_setup import bootstrap_section

_ = bootstrap_section({section!r}, globals())
"""

    cells = [
        md(
            r"""
# Underwater Computer Vision With FathomNet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Spiffical/fathomnet-underwater-vision-tutorial/blob/main/notebooks/fathomnet_underwater_vision_tutorial.ipynb)

This notebook introduces computer vision workflows for underwater imagery using FathomNet-derived data. You will train and compare image classification, object detection, instance segmentation, and promptable segmentation workflows.

You only need a mathematical foundation to start. You can follow the guided path from classification to segmentation, or jump into the section that matches your comfort level.

You will work with images derived from the [FathomNet Database](https://database.fathomnet.org/fathomnet/#/), an expert-annotated underwater image database designed to support marine science and machine learning. The problem is simple to state and hard to solve: given an underwater image, decide what organisms or biological structures are present, where they are, and sometimes which pixels belong to each object.

Underwater imagery is not just "ImageNet, but blue." Vehicle lights, water-column attenuation, turbidity, motion blur, scale changes, partial animals, transparent bodies, and long-tailed taxonomy all make this a useful stress test for computer vision.

Reference links:

- FathomNet Database: https://database.fathomnet.org/fathomnet/#/
- FathomNet data use policy: https://www.fathomnet.org/datause
- Ultralytics training guide: https://docs.ultralytics.com/modes/train/
- Ultralytics classification task: https://docs.ultralytics.com/tasks/classify/
- Ultralytics detection format: https://docs.ultralytics.com/datasets/detect/
- Ultralytics segmentation format: https://docs.ultralytics.com/datasets/segment/
- SAM3 repository: https://github.com/facebookresearch/sam3
- SAM3 image predictor example: https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb
"""
        ),
        md(
            r"""
## Notebook Map

Start with setup and the first image grid so the visual problem is concrete. After that, you can follow the notebook in order or jump to the section that matches your comfort level. Each later section starts with a short **section bootstrap** cell that loads the variables and helpers needed for that section.

What you will do:

- **Setup and first look:** load the compact tutorial bundle, check the runtime, and immediately look at real underwater images.
- **Dataset exploration:** inspect the same imagery through different label views: image-level crop classes, bounding boxes, and segmentation polygons.
- **Pretrained YOLO warmup:** use an existing YOLO model before training anything, so prediction outputs and confidence scores have a concrete meaning.
- **Classification:** train a small crop classifier and connect softmax, cross-entropy, learning rate, and confusion matrices to the observed results.
- **Object detection:** fine-tune YOLO on binary organism boxes and study box parameterization, intersection over union (IoU), precision, recall, and mean average precision (mAP).
- **Instance segmentation:** fine-tune YOLO segmentation on binary masks and compare box-level and mask-level evaluation.
- **SAM3 promptable segmentation:** use text prompts such as `"fish"` or `"small crab"` to request masks, boxes, and scores, with cached outputs available for every runtime.

The default notebook behavior is:

- use a prebuilt compact data bundle,
- run live YOLO training only when a GPU is available,
- fall back to cached training curves and cached SAM3-style outputs when a GPU, model weights, or Hugging Face access are unavailable.

Vocabulary you will see:

- **YOLO** means "You Only Look Once"; in this notebook it refers to Ultralytics YOLO models for classification, detection, and segmentation.
- A **bounding box** is a rectangle around an object, usually represented by center point, width, and height.
- **Segmentation** means predicting pixels or regions, not just a class or rectangle.
- **Instance segmentation** means predicting a separate mask for each object instance.
- **COCO** means "Common Objects in Context"; here it mostly refers to a widely used JSON annotation format for images, categories, bounding boxes, and segmentations.
- **SAM3** means "Segment Anything Model 3"; in this notebook it refers to Meta's promptable segmentation model for detecting and segmenting objects from text prompts.
"""
        ),
        md(
            r"""
## Setup And Runtime Check

First locate the repository, optionally install dependencies, load the small bundle, and inspect the GPU.

GitHub repository for this tutorial:

https://github.com/Spiffical/fathomnet-underwater-vision-tutorial

The notebook works in two common modes:

- **Colab from GitHub:** the setup cell clones the repository into `/content/fathomnet_underwater_vision_tutorial`.
- **Local Jupyter:** start Jupyter from the cloned repository, or from a child directory inside it.

Data storage model for the session:

- The canonical data bundle is `data/fathomnet_underwater_tutorial_bundle.zip` in the repository.
- For Colab, the default `BUNDLE_URL` points to the zip in this GitHub repository.
- In Colab, you download your own copy into the temporary runtime, then the notebook extracts it under `data/fathomnet_underwater_tutorial_bundle`.
- Nothing is downloaded live from FathomNet during the session.
"""
        ),
        code(
            r"""
from pathlib import Path
import json
import os
import subprocess
import sys

GITHUB_REPO_URL = "https://github.com/Spiffical/fathomnet-underwater-vision-tutorial"
GITHUB_BRANCH = "main"
PROJECT_DIR_NAME = "fathomnet_underwater_vision_tutorial"

def _candidate_roots():
    cwd = Path.cwd().resolve()
    yield cwd
    yield from cwd.parents
    yield Path("/content") / PROJECT_DIR_NAME

REPO_ROOT = None
for candidate in _candidate_roots():
    if (candidate / "scripts").exists() and (candidate / "notebooks").exists():
        REPO_ROOT = candidate
        break

if REPO_ROOT is None and "google.colab" in sys.modules:
    clone_dir = Path("/content") / PROJECT_DIR_NAME
    if not clone_dir.exists():
        subprocess.check_call([
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            GITHUB_BRANCH,
            f"{GITHUB_REPO_URL}.git",
            str(clone_dir),
        ])
    REPO_ROOT = clone_dir

if REPO_ROOT is None:
    raise RuntimeError("Could not find the tutorial repo root. Start Jupyter from the cloned repo.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print(f"Repository root: {REPO_ROOT}")
"""
        ),
        md(
            r"""
### Dependencies And Runtime

Now check the Python dependencies, inspect the GPU/runtime, and set the random seed. In Colab this cell installs missing packages; locally it only checks what is already importable.
"""
        ),
        code(
            r"""
from scripts.tutorial_setup import (
    detect_runtime,
    download_tutorial_bundle,
    ensure_dependencies,
    print_runtime_summary,
    set_reproducible_seed,
)

# Local runs use the repo-local zip automatically. In Colab, this URL is the
# fallback if the zip is not already present after cloning.
BUNDLE_URL = "https://github.com/Spiffical/fathomnet-underwater-vision-tutorial/raw/main/data/fathomnet_underwater_tutorial_bundle.zip"
LOCAL_BUNDLE_ZIP = REPO_ROOT / "data" / "fathomnet_underwater_tutorial_bundle.zip"

IN_COLAB = "google.colab" in sys.modules
INSTALL_DEPENDENCIES = IN_COLAB
RUN_LIVE_TRAINING = False  # updated after the runtime check

ensure_dependencies(install=INSTALL_DEPENDENCIES, extra_pip_args=("--quiet",))
RUNTIME = print_runtime_summary(detect_runtime())
RUN_LIVE_TRAINING = bool(RUNTIME.get("has_cuda", False))
set_reproducible_seed(42)

print(f"RUN_LIVE_TRAINING = {RUN_LIVE_TRAINING}")
"""
        ),
        md(
            r"""
### Download Or Unpack The Tutorial Bundle

This cell prepares the compact FathomNet-derived data bundle. In Colab, each participant gets a temporary runtime copy. Locally, the notebook uses the zip already stored in the repository when available.
"""
        ),
        code(
            r"""
BUNDLE_ROOT = download_tutorial_bundle(
    bundle_url=BUNDLE_URL or None,
    bundle_zip_path=LOCAL_BUNDLE_ZIP if LOCAL_BUNDLE_ZIP.exists() else None,
    output_dir=REPO_ROOT / "data" / "fathomnet_underwater_tutorial_bundle",
)

print(f"Bundle root: {BUNDLE_ROOT}")
print((BUNDLE_ROOT / "manifest.json").read_text()[:1200])
"""
        ),
        md(
            r"""
## First Look At FathomNet Imagery

Before training anything, look at the data. These are full underwater images from the compact FathomNet-derived bundle. Some organisms are obvious, some are tiny, and some are visually ambiguous even for a human.

As you scan the grid, ask:

- What would count as an "object"?
- Which organisms are easy to crop and classify?
- Which organisms need a bounding box?
- Which organisms would really need a pixel mask?
"""
        ),
        code(
            r"""
from scripts.tutorial_data import get_task_paths
from scripts.tutorial_viz import show_image_grid

detect_paths = get_task_paths("detect", BUNDLE_ROOT)
segment_paths = get_task_paths("segment", BUNDLE_ROOT)
classification_paths = get_task_paths("classification", BUNDLE_ROOT)
coco_paths = get_task_paths("coco", BUNDLE_ROOT)

first_look_images = sorted((detect_paths["root"] / "images" / "val").glob("*.jpg"))[:8]
show_image_grid(
    first_look_images,
    titles=[path.stem[:8] for path in first_look_images],
    columns=4,
)
"""
        ),
        md(
            r"""
## YOLO Warm-Up: Predict Before Training

The official Ultralytics Colab tutorial starts with the shortest useful workflow: install/check the package, run `predict` on a normal image, then move to validation, training, and export. You can open that reference here:

https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb

This notebook uses the same idea, but keeps the warm-up small. First, load a pretrained YOLO model and ask it to predict objects in a familiar image. After that, move to underwater imagery where the pretrained vocabulary and the visual statistics are much less convenient.

The official tutorial currently uses newer YOLO model families. This workshop defaults to `yolo11n.pt` and `yolo11n-seg.pt` because they match the prepared underwater experiments and run quickly in free Colab. You can swap model names later.
"""
        ),
        code(
            r"""
YOLO_WARMUP_SOURCE = "https://ultralytics.com/images/zidane.jpg"
# `conf` is the confidence threshold: YOLO keeps detections whose score is at
# least this value. Raising it makes predictions more conservative.
YOLO_WARMUP_CONF = 0.25
YOLO_WARMUP_RESULTS = None

try:
    from ultralytics import YOLO
    import matplotlib.pyplot as plt

    warmup_model = YOLO("yolo11n.pt")
    YOLO_WARMUP_RESULTS = warmup_model.predict(
        source=YOLO_WARMUP_SOURCE,
        imgsz=640,
        conf=YOLO_WARMUP_CONF,
        project=REPO_ROOT / "runs" / "warmup",
        name="predict",
        exist_ok=True,
        verbose=False,
    )

    result = YOLO_WARMUP_RESULTS[0]
    print("CLI equivalent:")
    print(f"yolo predict model=yolo11n.pt source='{YOLO_WARMUP_SOURCE}' conf={YOLO_WARMUP_CONF}")
    print()
    print("Detected classes:", [result.names[int(cls)] for cls in result.boxes.cls.tolist()])

    annotated = result.plot()
    plt.figure(figsize=(8, 5))
    plt.imshow(annotated[..., ::-1])
    plt.axis("off")
    plt.show()

    # Ultralytics may cache downloaded URL sources in the working directory.
    for temporary_name in ["zidane.jpg", "bus.jpg"]:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()
except Exception as exc:
    print("YOLO warm-up prediction skipped. This can happen offline or before dependencies are installed.")
    print(type(exc).__name__, exc)
"""
        ),
        md(
            r"""
### YOLO Warm-Up Exercises

Beginner:

- Change `YOLO_WARMUP_CONF` from `0.25` to `0.6`.
- Which detections disappear first?

Intermediate:

- Change `YOLO_WARMUP_SOURCE` to `https://ultralytics.com/images/bus.jpg`.
- Read the printed class names and decide whether the model is solving classification, detection, or segmentation.

Advanced:

- Run the same pretrained model on one underwater validation image before fine-tuning.
- What does the model miss, and what does it hallucinate?
"""
        ),
        md(
            r"""
## Dataset Exploration

You have now seen the images. Next, inspect how the same underlying problem is represented for different machine learning tasks.

The bundle has four useful views:

- `classification_crops`: organism crops grouped by source-level FathomNet concept label.
- `yolo_detect_binary`: one-class object detection labels.
- `yolo_segment_binary`: one-class instance segmentation polygon labels.
- `sam3_cached_outputs`: SAM3-like prompt outputs for the fallback lab.

The full source COCO-style annotation file in the local YOLO project has `119,096` images, `280,118` annotations, and `1,897` categories. **COCO** means "Common Objects in Context"; the name comes from a benchmark dataset, but people also use "COCO format" to mean a JSON annotation schema with image records, category records, and annotation records. This tutorial uses a compact subset so you can work in Colab without spending the session on data transfer.

For the live detection and segmentation exercises, the YOLO labels deliberately omit extremely tiny annotations. That keeps the training signal about visible organisms rather than letting dense fragments dominate the classroom metrics. The original COCO subset is still included for discussion and after-session extensions.

The images and annotations are FathomNet-derived. The bundle includes `licenses/attribution.csv`; visual content is subject to contributor-selected Creative Commons terms and is used here for machine-learning training and development.
"""
        ),
        code(
            r"""
from collections import Counter

from scripts.tutorial_data import (
    classification_examples_by_class,
    get_task_paths,
    load_manifest,
    select_coco_category_examples,
    select_yolo_examples,
)
from scripts.tutorial_viz import draw_yolo_boxes, draw_yolo_masks, show_image_grid

detect_paths = get_task_paths("detect", BUNDLE_ROOT)
segment_paths = get_task_paths("segment", BUNDLE_ROOT)
classification_paths = get_task_paths("classification", BUNDLE_ROOT)
coco_paths = get_task_paths("coco", BUNDLE_ROOT)

manifest = load_manifest(BUNDLE_ROOT)
print(json.dumps({
    "name": manifest["name"],
    "classification_classes": manifest.get("classification_classes", []),
    "classification_source_concepts": manifest.get("classification_source_concepts", {}),
    "stats": manifest["stats"],
}, indent=2))
"""
        ),
        md(
            r"""
### View 1: Full Images With Truth Category Labels

This is a full-image, classification-style view of the COCO-format truth labels. The title above each image lists categories annotated as present in that image. This view hides boxes and masks on purpose, so you can separate the question "what is present?" from the question "where is it?"
"""
        ),
        code(
            r"""
coco_category_examples = select_coco_category_examples(
    coco_paths["json"],
    [detect_paths["root"] / "images" / "train", detect_paths["root"] / "images" / "val"],
    limit=6,
)

def compact_category_title(item, max_categories=2):
    '''Make a short title from ground-truth category names.'''

    from textwrap import wrap

    names = item["category_names"]
    visible = ", ".join(names[:max_categories]) if names else "no named categories"
    if len(names) > max_categories:
        visible += f", +{len(names) - max_categories} more"
    title_lines = wrap(visible, width=34, break_long_words=False) or [visible]
    title_lines.append(f"{item['annotation_count']} annotations")
    return "\n".join(title_lines)

show_image_grid(
    [item["image_path"] for item in coco_category_examples],
    titles=[compact_category_title(item) for item in coco_category_examples],
    columns=3,
    max_images=6,
    figsize=(14, 8),
)
"""
        ),
        md(
            r"""
### View 2: Classification Crops With Truth Labels

The training classifier uses organism crops grouped by source-level FathomNet concept labels. Each crop has one truth label from the folder name. There is still no geometry target here: no center point, no bounding box, no polygon.
"""
        ),
        code(
            r"""
classification_examples = classification_examples_by_class(
    classification_paths["root"],
    split="val",
    examples_per_class=2,
)

show_image_grid(
    [item["image_path"] for item in classification_examples],
    titles=[item["class_name"] for item in classification_examples],
    columns=4,
    max_images=12,
)
"""
        ),
        md(
            r"""
### View 3: YOLO Object Detection Truth Labels

Object detection asks for a class and a rectangle for each object. The YOLO detection rows here have the form

`class_id x_center y_center width height`

where the four geometry values are normalized to `[0, 1]` by the image width and height. The examples below are selected because they contain multiple labeled objects.
"""
        ),
        code(
            r"""
import matplotlib.pyplot as plt

detect_multi_examples = select_yolo_examples(
    detect_paths["root"],
    split="val",
    min_instances=2,
    limit=4,
)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, item in zip(axes.flat, detect_multi_examples):
    draw_yolo_boxes(item["image_path"], item["label_path"], class_names={0: "object"}, ax=ax)
    ax.set_title(f"{Path(item['image_path']).stem[:8]}: {item['instance_count']} objects")
for ax in axes.flat[len(detect_multi_examples):]:
    ax.axis("off")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            r"""
### View 4: YOLO Instance Segmentation Truth Labels

Instance segmentation asks for a separate region for each object. In YOLO segmentation format, each row starts with the class id and is followed by normalized polygon coordinates:

`class_id x1 y1 x2 y2 ... xK yK`

The target is more informative than a box, but it is also more expensive to annotate and usually harder to learn.
"""
        ),
        code(
            r"""
segment_multi_examples = select_yolo_examples(
    segment_paths["root"],
    split="val",
    min_instances=2,
    limit=4,
)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, item in zip(axes.flat, segment_multi_examples):
    draw_yolo_masks(item["image_path"], item["label_path"], class_names={0: "object"}, ax=ax)
    ax.set_title(f"{Path(item['image_path']).stem[:8]}: {item['instance_count']} masks")
for ax in axes.flat[len(segment_multi_examples):]:
    ax.axis("off")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            r"""
### Annotation Counts And Category Mix

You have now seen the truth labels visually. The next two cells summarize the same bundle numerically: first the COCO-style categories and annotation counts, then the object-size distribution used by the YOLO detection labels.
"""
        ),
        code(
            r"""
coco = json.loads(coco_paths["json"].read_text())
categories_by_id = {category["id"]: category["name"] for category in coco["categories"]}
annotations_by_image = Counter(annotation["image_id"] for annotation in coco["annotations"])
category_counts = Counter(categories_by_id.get(annotation["category_id"], annotation["category_id"]) for annotation in coco["annotations"])

print("COCO-style subset:")
print(f"  images:      {len(coco['images'])}")
print(f"  annotations: {len(coco['annotations'])}")
print(f"  categories:  {len(coco['categories'])}")
print()
print("Top annotated categories in this small subset:")
for name, count in category_counts.most_common(12):
    print(f"  {name:35s} {count:3d}")
print()
print("Annotations per image:")
print(f"  min={min(annotations_by_image.values())}, max={max(annotations_by_image.values())}, mean={sum(annotations_by_image.values()) / len(annotations_by_image):.2f}")
"""
        ),
        md(
            r"""
### Object Size Distribution

This histogram shows how large the detection boxes are as a fraction of image area. It matters because tiny objects are harder to localize, and the tutorial labels intentionally filter out extremely tiny boxes for the live training exercises.

The bundle uses a normalized bounding-box area threshold of `0.005`: an object is kept only if its box covers at least `0.5%` of the image area. For segmentation labels, the same idea is applied to the bounding box around the polygon.

This is a teaching choice, not a scientific law. It makes short training runs easier to interpret because the labels focus on visible organisms. The tradeoff is that the live task becomes "find larger, visually salient objects," not "find every organism in the scene."
"""
        ),
        code(
            r"""
import matplotlib.pyplot as plt

print("Tiny-object filter threshold:", manifest["stats"]["yolo_min_box_area"])

label_paths = sorted((detect_paths["root"] / "labels" / "train").glob("*.txt")) + sorted((detect_paths["root"] / "labels" / "val").glob("*.txt"))
box_area_fractions = []
instances_per_file = []

for label_path in label_paths:
    rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
    instances_per_file.append(len(rows))
    for row in rows:
        _, _, _, width, height = map(float, row)
        box_area_fractions.append(width * height)

print(f"YOLO detection files: {len(label_paths)}")
print(f"YOLO detection instances after tiny-object filtering: {len(box_area_fractions)}")
print(f"instances/image: min={min(instances_per_file)}, max={max(instances_per_file)}, mean={sum(instances_per_file) / len(instances_per_file):.2f}")
print(f"box area fraction: min={min(box_area_fractions):.4f}, median={sorted(box_area_fractions)[len(box_area_fractions)//2]:.4f}, max={max(box_area_fractions):.4f}")

plt.figure(figsize=(7, 3.5))
plt.hist(box_area_fractions, bins=18, edgecolor="black")
plt.xlabel("normalized box area")
plt.ylabel("count")
plt.title("Object sizes in the live YOLO detection labels")
plt.grid(alpha=0.25)
plt.show()
"""
        ),
        md(
            r"""
### Advanced Exercise: Audit The Train/Validation Split

A validation metric is only useful if the validation split is independent and reasonably representative. In this exercise, audit the YOLO detection split before trusting its mAP.

Implement `audit_yolo_split_exercise(...)` so it reports:

- image counts in `train` and `val`;
- whether any image stems appear in both splits;
- object instances per split;
- median box area per split.

Hints:

- labels live under `labels/train` and `labels/val`;
- image IDs can be compared by filename stem;
- YOLO detection rows store normalized `width` and `height` in columns 4 and 5;
- a suspicious audit result is a modeling problem, not just a data-loading problem.
"""
        ),
        code(
            r"""
def audit_yolo_split_exercise(dataset_root):
    '''Audit leakage and rough distribution shift for a YOLO detection dataset.

    Return a dictionary with split counts, overlap information, instance counts,
    and median normalized box areas. Leave the return value as `None` until you
    are ready to implement the exercise.
    '''

    # TODO:
    # 1. Collect image stems from images/train and images/val.
    # 2. Compare train and val stems to detect leakage.
    # 3. Read label rows from labels/train and labels/val.
    # 4. Count instances and compute each box area as width * height.
    # 5. Return a compact dictionary that you can print with json.dumps(...).
    return None


split_audit = audit_yolo_split_exercise(detect_paths["root"])
if split_audit is None:
    print("Exercise ready: implement audit_yolo_split_exercise(...) to audit the train/validation split.")
else:
    print(json.dumps(split_audit, indent=2))
"""
        ),
        md(
            r"""
### Dataset Exploration Questions

Beginner:

- Pick one image where the object is visually obvious and one where it is subtle.
- Which source of difficulty is most visible: contrast, scale, clutter, partial view, or taxonomy?

Intermediate:

- Compare the `coco/subset.json` category counts with the binary YOLO labels.
- What information is lost when many biological categories are collapsed into `object`?
- Compare the category-only full-image view with the box and mask views.
- What extra information do you gain when the truth labels include geometry?

Advanced:

- The live YOLO labels drop extremely tiny boxes. Predict how the histogram and mAP would change if those boxes were restored.
- The segmentation labels contain more geometry than boxes. Where would that extra information matter scientifically?
"""
        ),
        md(
            r"""
# Part 1: Classification From Organism Crops

Classification means predicting one label for an input. This is the easiest entry point because each organism crop has one label and no geometric target. The crop labels in this bundle are common FathomNet source concepts, so there are enough classes for top-5 accuracy to be meaningful instead of automatically perfect.

Here the image crop is a tensor

$$x \in [0,1]^{H \times W \times 3},$$

and the model produces one score, or logit, for each class. For `K` classes,

$$z=f_\theta(x)\in \mathbb{R}^K,\qquad y\in\{1,\dots,K\}.$$

The softmax turns logits into class probabilities,

$$p_k=\frac{\exp(z_k)}{\sum_j \exp(z_j)},$$

and cross-entropy penalizes the model when it gives low probability to the true class:

$$\ell(z,y)=-\log p_y.$$

The live exercise is to change one training knob, rerun the small model, and interpret whether validation accuracy moved in a meaningful direction.
"""
        ),
        md(
            r"""
### Train, Validation, And Test Splits

Before training, define the data splits. Most supervised machine-learning workflows keep separate sets:

- **Training set:** examples used to update model parameters by gradient descent.
- **Validation set:** examples used during development to choose hyperparameters, compare runs, tune thresholds, and select the best checkpoint.
- **Test set:** examples held back until the end for a final estimate of performance on data you did not optimize against.

In symbols, training chooses parameters

$$\hat{\theta}=\arg\min_\theta \frac{1}{n_{\mathrm{train}}}\sum_{i\in \mathrm{train}}\ell(f_\theta(x_i),y_i),$$

while validation estimates whether those parameters are useful away from the data that supplied the gradients. Ultralytics saves `weights/best.pt` as the checkpoint that performs best on the validation set during training. That is usually the checkpoint you evaluate or fine-tune from next, while `weights/last.pt` is simply the final epoch.

This compact tutorial bundle uses train/validation splits for live exercises. In a real project, keep a separate test split untouched until your modeling choices are fixed.
"""
        ),
        md(
            r"""
### Section Bootstrap: Load The Crop Dataset

Start this section by loading the classification paths and printing the split/class counts. If you skipped here from above, this bootstrap cell also restores the common setup variables.
"""
        ),
        code(
            section_bootstrap("classification")
        ),
        md(
            r"""
### Inspect The Classification Inputs

Before training, look at one crop from each class. These are the actual inputs for the classifier: one cropped image, one truth label.
"""
        ),
        code(
            r"""
class_examples = []
class_titles = []
for class_dir in sorted((CLASSIFY_ROOT / "train").iterdir()):
    images = sorted(class_dir.glob("*.jpg"))
    if images:
        class_examples.append(images[0])
        class_titles.append(class_dir.name)

show_image_grid(class_examples, titles=class_titles, columns=4)
"""
        ),
        md(
            r"""
### Visible Helper: Cross-Entropy

This is a small, editable version of the loss behind multi-class classification. It is not meant to replace PyTorch; it is here so the math is visible.
"""
        ),
        code(
            r"""
def softmax_cross_entropy_from_logits(logits, true_class):
    '''Compute -log softmax(logits)[true_class] in a numerically stable way.'''

    import math

    max_logit = max(logits)
    shifted = [value - max_logit for value in logits]
    normalizer = sum(math.exp(value) for value in shifted)
    log_probability = shifted[true_class] - math.log(normalizer)
    return -log_probability


example_logits = [1.2, -0.4, 0.7, 2.1, 0.0]
print("loss if the true class is index 3:", softmax_cross_entropy_from_logits(example_logits, 3))
print("loss if the true class is index 1:", softmax_cross_entropy_from_logits(example_logits, 1))
"""
        ),
        md(
            r"""
### Visible Helper: Training Arguments

You are encouraged to edit this function before running a training cell. It is deliberately small: the goal is to connect a few knobs to training behavior.
"""
        ),
        code(
            r"""
def build_train_args(
    *,
    n_epochs=10,
    imgsz=320,
    batch=8,
    lr0=0.001,
    optimizer="AdamW",
    patience=3,
    workers=0,
    project="runs/tutorial",
    name="experiment",
    seed=42,
):
    '''Collect the main training hyperparameters in one visible place.

    Think of this as choosing an optimizer trajectory through parameter space:

        theta_{t+1} = theta_t - eta * grad L(theta_t)

    where `lr0` controls the initial step size eta, `batch` controls how noisy
    the gradient estimate is, and `n_epochs` controls how many passes you make
    through the finite training sample. The optimizer is explicit so Ultralytics
    uses the learning rate you choose instead of replacing it with an automatic
    choice.

    Ultralytics expects this argument to be named `epochs`, so the returned
    dictionary maps the classroom-facing `n_epochs` name to `epochs`.
    '''

    return {
        "epochs": int(n_epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "lr0": float(lr0),
        "optimizer": str(optimizer),
        "patience": int(patience),
        "workers": int(workers),
        "project": str(project),
        "name": str(name),
        "seed": int(seed),
        # Keep checkpoint saving explicit. Ultralytics writes weights/best.pt
        # for the validation-best checkpoint and weights/last.pt for the final epoch.
        "save": True,
        "verbose": False,
    }
"""
        ),
        md(
            r"""
### Train Or Load A Small Classification Run

Now move from the hand-computed loss to an actual model. This cell trains `yolo11n-cls.pt` when a GPU is available. If not, it loads the cached classification curve so the interpretation exercise still works.

Focus on the training arguments: `n_epochs`, `imgsz`, `batch`, `optimizer`, and `lr0`. Those are the knobs you will modify in the exercises. When live training runs, Ultralytics saves the validation-best checkpoint at `weights/best.pt`.

The plot reports top-1 and top-5 accuracy when those columns are available. Top-1 asks whether the highest-probability class is correct. Top-5 asks whether the truth appears anywhere in the five highest-probability classes, which is useful here because the classifier has more than five possible labels.
"""
        ),
        code(
            r"""
CLASSIFY_N_EPOCHS = 10
CLASSIFY_ARGS = build_train_args(
    n_epochs=CLASSIFY_N_EPOCHS,
    imgsz=224,
    batch=16,
    lr0=0.001,
    project=REPO_ROOT / "runs" / "classification",
    name="classification_default",
)

# Documentation checkpoint:
# 1. Open https://docs.ultralytics.com/tasks/classify/
# 2. Find how a pretrained classification model is loaded.
# 3. Compare that with the general training pattern at https://docs.ultralytics.com/modes/train/
#
# Your live exercise is to modify the model name or train arguments below.
classification_training = None
classification_save_dir = None
classification_best_model_path = None
if RUN_LIVE_TRAINING:
    from ultralytics import YOLO

    classification_model = YOLO("yolo11n-cls.pt")
    classification_training = classification_model.train(data=str(CLASSIFY_ROOT), **CLASSIFY_ARGS)
    classification_save_dir = getattr(classification_training, "save_dir", None)
    if classification_save_dir is None and getattr(classification_model, "trainer", None) is not None:
        classification_save_dir = getattr(classification_model.trainer, "save_dir", None)
    print("Live classification training finished.")
    candidate_best = Path(classification_save_dir) / "weights" / "best.pt" if classification_save_dir is not None else None
    if candidate_best is not None and candidate_best.exists():
        classification_best_model_path = candidate_best
        print(f"Best validation checkpoint: {classification_best_model_path}")
else:
    print("No GPU detected. Using cached classification training curve.")

classification_results_csv = (
    Path(classification_save_dir) / "results.csv"
    if classification_save_dir is not None
    else CACHED_CLASSIFY_RESULTS
)
if not classification_results_csv.exists():
    classification_results_csv = CACHED_CLASSIFY_RESULTS

plot_training_curves(classification_results_csv, title="Classification learning curve")
"""
        ),
        md(
            r"""
### Mini-Lab: Learning Rate As An Optimization Knob

The learning rate controls the scale of each gradient step. In a small workshop run, you usually cannot prove one setting is globally best. You can still learn to recognize three useful patterns:

- too small: the curve barely moves;
- useful: validation accuracy improves without wild instability;
- too large: loss or validation accuracy jumps around or degrades.

Try one or more learning rates from `lr0 = 1e-4`, `1e-3`, and `1e-2`. The default mini-lab uses several epochs because one epoch is often too short to reveal the difference between slow, useful, and unstable optimization. The cached curves give you a baseline comparison, and a live GPU run lets you see how much the result changes from one run to the next.
"""
        ),
        code(
            r"""
RUN_CLASSIFICATION_LR_LAB = False
LR_VALUES = [1e-4, 1e-3, 1e-2]
LR_LAB_N_EPOCHS = 5

if RUN_CLASSIFICATION_LR_LAB and RUN_LIVE_TRAINING:
    lr_rows = [
        run_classification_lr_trial(
            CLASSIFY_ROOT,
            build_train_args,
            lr0=lr0,
            repo_root=REPO_ROOT,
            n_epochs=LR_LAB_N_EPOCHS,
        )
        for lr0 in LR_VALUES
    ]
    for row in lr_rows:
        print(row)
        plot_training_curves(row["results_csv"], title=f"classification lr0={row['lr0']}")
else:
    print("Learning-rate lab is ready.")
    print("Set RUN_CLASSIFICATION_LR_LAB = True on a GPU to run these trials live.")
    print("Suggested lr0 values:", LR_VALUES)
"""
        ),
        md(
            r"""
### Confusion Matrix Reading Practice

The next cell uses a small discussion matrix rather than requiring live predictions. Read it row by row: each row is a true class, and each column is the predicted class. The goal is to connect metric summaries to specific biological or visual confusions.
"""
        ),
        code(
            r"""
# A small confusion matrix for discussion. It uses the first few classes in the
# current bundle so the labels stay synchronized if the data bundle is rebuilt.
classes = [path.name for path in sorted((CLASSIFY_ROOT / "val").iterdir())[:6] if path.is_dir()]
toy_confusion = [
    [7, 1, 0, 0, 0, 0],
    [1, 5, 1, 0, 0, 0],
    [0, 1, 6, 1, 0, 0],
    [0, 0, 1, 5, 1, 0],
    [0, 0, 0, 1, 6, 1],
    [1, 0, 0, 0, 1, 5],
]
plot_confusion_matrix(toy_confusion, classes, normalize=True, title="Discussion matrix: normalized by true class")
"""
        ),
        md(
            r"""
### Advanced Exercise: Build A Confusion Matrix From Predictions

The toy matrix above is useful for reading practice. A real confusion matrix should come from model predictions on validation images.

Use the Ultralytics classification prediction docs and `sklearn.metrics.confusion_matrix` to implement `build_confusion_matrix_from_predictions(...)`.

Hints:

- validation crops are stored in `CLASSIFY_ROOT / "val" / class_name`;
- the true class comes from the folder name;
- an Ultralytics classification result has class probabilities in `result.probs`;
- use the trained `classification_best_model_path` if it exists, otherwise this exercise is best run after live training.
"""
        ),
        code(
            r"""
RUN_CLASSIFICATION_CONFUSION_ADVANCED = False

def build_confusion_matrix_from_predictions(model_or_path, val_root):
    '''Predict validation crops and return `(matrix, class_names)`.

    Leave the return value as `None` until you are ready to implement the
    advanced exercise.
    '''

    # TODO:
    # 1. Load a YOLO classification model.
    # 2. Walk through each class folder in `val_root`.
    # 3. Predict each image and record true/predicted class ids.
    # 4. Use sklearn.metrics.confusion_matrix(...).
    return None


if RUN_CLASSIFICATION_CONFUSION_ADVANCED:
    model_path = classification_best_model_path
    if model_path is None:
        print("Run live classification training first so `classification_best_model_path` exists.")
    else:
        confusion_result = build_confusion_matrix_from_predictions(model_path, CLASSIFY_ROOT / "val")
        if confusion_result is None:
            print("Exercise ready: implement build_confusion_matrix_from_predictions(...).")
        else:
            matrix, class_names = confusion_result
            plot_confusion_matrix(matrix, class_names, normalize=True, title="Validation confusion matrix")
else:
    print("Advanced confusion-matrix exercise is ready.")
    print("Set RUN_CLASSIFICATION_CONFUSION_ADVANCED = True after implementing the helper.")
"""
        ),
        md(
            r"""
### Classification Exercises

Beginner:

- Change `CLASSIFY_N_EPOCHS` from `10` to another value, then rerun the training cell.
- Run or reason through one learning-rate trial.
- Decide whether the validation curve is improving, noisy, or overfitting.

Intermediate:

- Use the Ultralytics classification docs to find the direct `YOLO(...).train(...)` pattern for classification.
- Compare one `lr0` result with another `lr0` result.
- Change `imgsz` from `224` to `320`, then decide whether the extra computation was worth it.
- Pick one class and inspect three crops that seem visually ambiguous.

Advanced:

- Design a fair pretrained-vs-random-initialization comparison. What would you keep fixed?
- Complete `build_confusion_matrix_from_predictions(...)` and replace the toy matrix with predictions from your live model.
- Which two classes are most confusable, and what visual ambiguity might explain it?
- Propose a better class grouping for this small crop dataset.
"""
        ),
        md(
            r"""
# Part 2: Binary Object Detection With YOLO

Detection asks for *where* the animals or biological structures are, not only what crop class they belong to. In this section the target is a **bounding box**, or a rectangle that encloses an object.

Instead of one label per crop, the output is a set of boxes:

$$\{(c_i,b_i,s_i)\}_{i=1}^m,\qquad b_i=(c_x,c_y,w,h).$$

Here `c_i` is the class, `b_i` is a normalized box, and `s_i` is the confidence score. YOLO stores `c_x`, `c_y`, `w`, and `h` in `[0,1]` relative to image width and height.

The workshop bundle uses a binary **object** target and filters very small boxes. This makes the first detector behave like a useful mathematical object during a short tutorial: you can see IoU, confidence thresholds, precision, recall, and AP move for interpretable reasons.

During training, the model parameters are chosen by minimizing empirical risk on the training split:

$$
\hat{\theta}\in\arg\min_\theta \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x_i), y_i).
$$

The validation split is the quick estimate of whether those parameters are useful outside the examples that supplied the gradients.

For one ground-truth box `A` and one predicted box `B`, the intersection-over-union is

$$\operatorname{IoU}(A,B)=\frac{|A \cap B|}{|A \cup B|}.$$

Precision and recall depend on a confidence threshold:

$$\mathrm{precision}=\frac{TP}{TP+FP}, \qquad \mathrm{recall}=\frac{TP}{TP+FN}.$$

Average precision, or **AP**, summarizes the precision-recall curve as that threshold changes. **mAP** means mean average precision. In YOLO reports, `mAP50` uses an IoU threshold of `0.50`, while `mAP50-95` averages over several IoU thresholds from `0.50` to `0.95`, making it a stricter localization metric.
"""
        ),
        md(
            r"""
### Mini-Lab: Matching Predictions To Labels

Detection metrics are built from a matching rule. A predicted box becomes a true positive only if it has enough overlap with an unmatched ground-truth box. Everything else is a false positive, and every unmatched ground-truth box is a false negative.

Use this toy example before looking at YOLO outputs. It is small enough to calculate by hand.
"""
        ),
        code(
            r"""
def box_iou_xyxy(box_a, box_b):
    '''IoU for boxes in [x_min, y_min, x_max, y_max] pixel coordinates.'''

    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_width = max(0, inter_x1 - inter_x0)
    inter_height = max(0, inter_y1 - inter_y0)
    intersection = inter_width * inter_height
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def one_object_precision_recall(predictions, ground_truth_box, *, conf_threshold=0.25, iou_threshold=0.5):
    '''Simplified precision/recall for one ground-truth object.

    `predictions` is a list of dictionaries with `box` and `score`.
    The highest-score prediction above the threshold that overlaps enough gets
    matched as the true positive. Other predictions above threshold are false positives.
    '''

    kept = [prediction for prediction in predictions if prediction["score"] >= conf_threshold]
    kept = sorted(kept, key=lambda prediction: prediction["score"], reverse=True)
    matched = False
    true_positives = 0
    false_positives = 0

    for prediction in kept:
        iou = box_iou_xyxy(prediction["box"], ground_truth_box)
        prediction["iou"] = iou
        if iou >= iou_threshold and not matched:
            true_positives += 1
            matched = True
        else:
            false_positives += 1

    false_negatives = 0 if matched else 1
    precision = true_positives / (true_positives + false_positives) if kept else 0.0
    recall = true_positives / (true_positives + false_negatives)
    return {
        "kept_predictions": kept,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
    }


toy_ground_truth = [10, 10, 50, 50]
toy_predictions = [
    {"box": [12, 12, 48, 48], "score": 0.92},
    {"box": [55, 12, 90, 45], "score": 0.70},
    {"box": [8, 8, 35, 35], "score": 0.35},
]

for threshold in [0.25, 0.5, 0.8]:
    summary = one_object_precision_recall(toy_predictions, toy_ground_truth, conf_threshold=threshold)
    print(f"conf >= {threshold}:", {key: summary[key] for key in ["tp", "fp", "fn", "precision", "recall"]})
"""
        ),
        md(
            r"""
### Section Bootstrap: Load The Detection Dataset

The toy example above shows the metric logic. Now switch to the real YOLO detection dataset. This bootstrap cell loads the dataset YAML, validates image/label pairing, and defines the cached result path used when live training is unavailable.
"""
        ),
        code(
            section_bootstrap("detection")
        ),
        md(
            r"""
### Detection Truth Example: Several Objects In One Image

This plot shows one validation image from the YOLO detection dataset. The green rectangles are **ground-truth labels**, not model predictions. Each row in the matching `.txt` label file says: class id, normalized box center, normalized box width, and normalized box height.

The example is selected to contain multiple labeled objects so the detection task is visually clear.
"""
        ),
        code(
            r"""
import matplotlib.pyplot as plt

PREFERRED_DETECT_EXAMPLE_STEM = "eb7661e6-ded7-49dc-b335-0e7dfbb5d775"
detect_candidates = select_yolo_examples(DETECT_ROOT, split="val", min_instances=2, limit=6)
detect_example = next(
    (
        item
        for item in detect_candidates
        if item["image_path"].stem == PREFERRED_DETECT_EXAMPLE_STEM
    ),
    detect_candidates[0],
)
first_detect_label = detect_example["label_path"]
first_detect_image = detect_example["image_path"]
DETECT_EXAMPLE_STEM = first_detect_image.stem

print(f"Showing {DETECT_EXAMPLE_STEM}: {detect_example['instance_count']} truth boxes")
ax = draw_yolo_boxes(first_detect_image, first_detect_label, class_names={0: "object"})
ax.set_title(f"Detection truth labels: {detect_example['instance_count']} objects")
plt.show()
"""
        ),
        md(
            r"""
### Train Or Load A Small Detection Run

Now we train a binary object detector. The model starts from `yolo11n.pt`, an Ultralytics YOLO detection checkpoint, and the dataset comes from `DETECT_YAML`.

If a GPU is available, this cell runs a short fine-tuning job. Otherwise, it loads a cached training curve. Either way, the output you should inspect is the same: precision, recall, `mAP50`, and `mAP50-95`. For a live run, the best validation checkpoint is saved as `weights/best.pt`.
"""
        ),
        code(
            r"""
DETECT_N_EPOCHS = 10
DETECT_ARGS = build_train_args(
    n_epochs=DETECT_N_EPOCHS,
    imgsz=320,
    batch=8,
    lr0=0.001,
    project=REPO_ROOT / "runs" / "detect",
    name="detect_default",
)

# Documentation checkpoint:
# 1. Open https://docs.ultralytics.com/modes/train/
# 2. Find the Python example that loads a pretrained model.
# 3. Find where `data=...`, `epochs=...`, and `imgsz=...` enter `model.train(...)`.
#
# Your live exercise is to understand and modify this small reference block.
# It uses direct Ultralytics code rather than a tutorial utility wrapper.
detection_training = None
detection_save_dir = None
detection_best_model_path = None
if RUN_LIVE_TRAINING:
    from ultralytics import YOLO

    detection_model = YOLO("yolo11n.pt")
    detection_training = detection_model.train(data=str(DETECT_YAML), **DETECT_ARGS)
    detection_save_dir = getattr(detection_training, "save_dir", None)
    if detection_save_dir is None and getattr(detection_model, "trainer", None) is not None:
        detection_save_dir = getattr(detection_model.trainer, "save_dir", None)
    print("Live detection training finished.")
    candidate_best = Path(detection_save_dir) / "weights" / "best.pt" if detection_save_dir is not None else None
    if candidate_best is not None and candidate_best.exists():
        detection_best_model_path = candidate_best
        print(f"Best validation checkpoint: {detection_best_model_path}")
else:
    print("No GPU detected. Using cached detection training curve.")

detection_results_csv = (
    Path(detection_save_dir) / "results.csv"
    if detection_save_dir is not None
    else CACHED_DETECT_RESULTS
)
if not detection_results_csv.exists():
    detection_results_csv = CACHED_DETECT_RESULTS

plot_training_curves(
    detection_results_csv,
    metric_columns=["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
    title="Detection metrics",
)
"""
        ),
        md(
            r"""
### Mini-Lab: Confidence Threshold Sweep

Training gives the model parameters. The confidence threshold is a decision rule you choose after the model scores candidate detections.

Sweep a few thresholds on the same underwater image. Your job is to describe the tradeoff in plain language before you look at the metric names:

- low threshold: more detections, more possible false positives;
- high threshold: fewer detections, more possible missed objects.
"""
        ),
        code(
            r"""
THRESHOLD_SWEEP_VALUES = [0.10, 0.25, 0.50, 0.80]
threshold_sweep_rows = []

try:
    from ultralytics import YOLO
    import matplotlib.pyplot as plt

    threshold_weights = (
        detection_best_model_path
        if detection_best_model_path is not None and detection_best_model_path.exists()
        else "yolo11n.pt"
    )
    threshold_model = YOLO(str(threshold_weights))
    threshold_image = first_detect_image
    threshold_sweep_rows = [
        prediction_count_at_threshold(threshold_model, threshold_image, conf=value, repo_root=REPO_ROOT)
        for value in THRESHOLD_SWEEP_VALUES
    ]

    for row in threshold_sweep_rows:
        print(f"conf={row['conf']:.2f}: detections={row['detections']}, mean score={row['mean_score']:.3f}")

    fig, axes = plt.subplots(1, len(threshold_sweep_rows), figsize=(4 * len(threshold_sweep_rows), 4))
    for ax, row in zip(axes, threshold_sweep_rows):
        ax.imshow(row["result"].plot()[..., ::-1])
        ax.set_title(f"conf >= {row['conf']}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
except Exception as exc:
    print("Threshold sweep skipped. You can run it after Ultralytics is available.")
    print(type(exc).__name__, exc)
"""
        ),
        md(
            r"""
### Mini-Lab: Overfit A Tiny Dataset

A powerful debugging move is to ask whether the model can memorize a tiny training set. If it cannot overfit 8-12 examples, something may be wrong with the labels, model wiring, optimization settings, or data path.

This cell prepares the tiny dataset for you. Training is off by default because it is a live GPU exercise.
"""
        ),
        code(
            r"""
RUN_TINY_OVERFIT_LAB = False
tiny_detect_yaml = make_tiny_detection_dataset(
    DETECT_ROOT,
    REPO_ROOT / "tmp" / "tiny_detection_overfit",
    train_images=8,
    val_images=8,
)
print(f"Tiny overfit dataset: {tiny_detect_yaml}")

if RUN_TINY_OVERFIT_LAB and RUN_LIVE_TRAINING:
    from ultralytics import YOLO

    TINY_OVERFIT_N_EPOCHS = 10
    tiny_args = build_train_args(
        n_epochs=TINY_OVERFIT_N_EPOCHS,
        imgsz=320,
        batch=4,
        lr0=0.003,
        patience=20,
        project=REPO_ROOT / "runs" / "tiny_overfit",
        name="detect_8_images",
    )
    tiny_model = YOLO("yolo11n.pt")
    tiny_result = tiny_model.train(data=str(tiny_detect_yaml), **tiny_args)
    tiny_save_dir = getattr(tiny_result, "save_dir", None) or getattr(tiny_model.trainer, "save_dir", None)
    tiny_best_model_path = Path(tiny_save_dir) / "weights" / "best.pt"
    if tiny_best_model_path.exists():
        print(f"Best validation checkpoint: {tiny_best_model_path}")
    plot_training_curves(
        Path(tiny_save_dir) / "results.csv",
        metric_columns=["train/box_loss", "val/box_loss", "metrics/mAP50(B)"],
        title="Tiny-dataset overfit check",
    )
else:
    print("Tiny overfit lab is ready. Set RUN_TINY_OVERFIT_LAB = True on a GPU.")
    print("Success criterion: training loss drops strongly; validation may or may not improve.")
"""
        ),
        md(
            r"""
### Advanced Exercise: Start From FathomNet Megalodon

The detector above starts from a general YOLO checkpoint. A stronger underwater baseline is [FathomNet Megalodon](https://huggingface.co/FathomNet/megalodon), a YOLOv8x object detector fine-tuned by MBARI/FathomNet to detect one class: `object`. The model card reports that it was trained from publicly available FathomNet localizations and is meant for post-processing underwater images and video.

This is an advanced path because the checkpoint is larger than `yolo11n.pt`, and fine-tuning it can be slow on a small free GPU. The key question is worth the trouble:

> Does a domain-specific FathomNet checkpoint give you a better starting point than a generic image checkpoint?

Useful references:

- FathomNet Megalodon model card: https://huggingface.co/FathomNet/megalodon
- Hugging Face Hub download docs: https://huggingface.co/docs/huggingface_hub/guides/download
- Ultralytics prediction mode: https://docs.ultralytics.com/modes/predict/
- Ultralytics training and fine-tuning mode: https://docs.ultralytics.com/modes/train/

You will not get the full solution here. The cells below give constants, checks, and hints. Your job is to use the documentation to fill in the missing pieces:

1. download the checkpoint from Hugging Face;
2. load it as a YOLO model;
3. run prediction on `first_detect_image`;
4. optionally fine-tune it on `DETECT_YAML`.
"""
        ),
        code(
            r"""
RUN_MEGALODON_ADVANCED = False

MEGALODON_REPO_ID = "FathomNet/megalodon"
MEGALODON_FILENAME = "mbari-megalodon-yolov8x.pt"
MEGALODON_MODEL_PATH = None

if RUN_MEGALODON_ADVANCED:
    # TODO:
    # 1. Import the Hugging Face download helper.
    # 2. Download `MEGALODON_FILENAME` from `MEGALODON_REPO_ID`.
    # 3. Store the returned local path in `MEGALODON_MODEL_PATH`.
    #
    # Hints:
    # - The helper is called `hf_hub_download`.
    # - Use `cache_dir=REPO_ROOT / ".cache" / "huggingface"` if you want the
    #   checkpoint to stay inside the tutorial repo.
    print("TODO: download the Megalodon checkpoint and set MEGALODON_MODEL_PATH.")
else:
    print("Advanced Megalodon path is off by default.")
    print("Set RUN_MEGALODON_ADVANCED = True to download the FathomNet checkpoint.")
"""
        ),
        md(
            r"""
#### Prediction Task

After the checkpoint path exists, use the Ultralytics prediction documentation to run Megalodon on `first_detect_image`. Compare the predicted boxes with the ground-truth boxes from the earlier detection-truth cell.
"""
        ),
        code(
            r"""
RUN_MEGALODON_PREDICT = False

if RUN_MEGALODON_PREDICT and MEGALODON_MODEL_PATH is not None:
    # TODO:
    # 1. Import YOLO from ultralytics.
    # 2. Load `MEGALODON_MODEL_PATH`.
    # 3. Run `.predict(...)` on `first_detect_image`.
    # 4. Plot the first result with `result.plot()`.
    #
    # Hints:
    # - Start with the Ultralytics prediction docs.
    # - Use a moderate image size first, for example `imgsz=640`.
    # - Compare the output with the truth boxes shown above.
    print("TODO: load Megalodon with YOLO(...) and run prediction.")
else:
    print("Prediction exercise ready.")
    print("Set RUN_MEGALODON_PREDICT = True after you have downloaded the checkpoint.")
"""
        ),
        md(
            r"""
#### Optional Fine-Tuning Scaffold

Fine-tuning means continuing optimization from an existing checkpoint instead of starting from generic weights. For this small tutorial dataset, use a small learning rate and very few epochs first. Your goal is not to produce a publishable model in one run; your goal is to learn whether the domain-specific starting point changes the early training behavior.

Before running the cell, read the Ultralytics training examples and identify:

- where the checkpoint path enters `YOLO(...)`;
- where the dataset YAML enters `model.train(data=...)`;
- which arguments control image size, batch size, learning rate, and epoch count.
"""
        ),
        code(
            r"""
RUN_MEGALODON_FINE_TUNE = False

if RUN_MEGALODON_ADVANCED and RUN_MEGALODON_FINE_TUNE and RUN_LIVE_TRAINING and MEGALODON_MODEL_PATH is not None:
    # TODO:
    # 1. Build a small training-argument dictionary.
    # 2. Load the Megalodon checkpoint with YOLO(...).
    # 3. Call `.train(data=str(DETECT_YAML), ...)`.
    # 4. Plot the resulting `results.csv` with `plot_training_curves`.
    #
    # Hints:
    # - Keep this small at first: a modest `n_epochs`, batch 1-2, and a small `lr0`.
    # - The checkpoint is larger than YOLO11n, so memory is the limiting resource.
    # - Use the generic detection training cell above as a pattern, but do not
    #   copy it blindly; check each argument against the Ultralytics docs.
    print("TODO: fine-tune Megalodon on DETECT_YAML and plot the metrics.")
else:
    print("Fine-tuning scaffold is ready.")
    print("Set RUN_MEGALODON_ADVANCED = True, RUN_MEGALODON_FINE_TUNE = True, and use a GPU.")
"""
        ),
        md(
            r"""
### Exercise: COCO Boxes To YOLO Boxes

The bundle already includes YOLO labels so you can spend most of the session on modeling. The real conversion step is deliberately left as an exercise.

COCO stores a bounding box as

```text
[x_min, y_min, width, height]
```

in pixel coordinates. YOLO detection stores one row as

```text
class_id center_x center_y width height
```

with all four geometry values normalized to `[0,1]`.
"""
        ),
        code(
            r"""
def coco_bbox_to_yolo_exercise(coco_bbox, image_width, image_height):
    '''Convert one COCO [x_min, y_min, width, height] box to YOLO geometry.

    Return `(center_x, center_y, width, height)`, normalized by image size.
    Leave the return value as `None` until you are ready to test your answer.
    '''

    # TODO: fill this in during the exercise.
    return None


example_coco_bbox = [20, 10, 40, 30]
candidate_box = coco_bbox_to_yolo_exercise(example_coco_bbox, image_width=100, image_height=100)

if candidate_box is None:
    print("Exercise ready: implement coco_bbox_to_yolo_exercise(...) when you reach the detection lab.")
else:
    print("Your YOLO box:", candidate_box)
    assert len(candidate_box) == 4
    assert all(0 <= value <= 1 for value in candidate_box)
"""
        ),
        md(
            r"""
### Mini-Lab: Error Taxonomy

Metrics tell you *how much* error there is. A failure taxonomy tells you *what kind* of error it is. For underwater imagery, common categories are:

- good prediction;
- missed small object;
- poor localization;
- duplicate detection;
- object-like background texture;
- ambiguous or incomplete label.

Look at a few predictions and assign one category to each. This is how you turn a metric into a modeling decision.
"""
        ),
        code(
            r"""
ERROR_CATEGORIES = [
    "good prediction",
    "missed small object",
    "poor localization",
    "duplicate detection",
    "object-like background texture",
    "ambiguous or incomplete label",
]

print("Use these categories while inspecting predictions:")
for index, category in enumerate(ERROR_CATEGORIES, start=1):
    print(f"{index}. {category}")

print()
print("Suggested workflow:")
print("1. Pick two images from the threshold sweep.")
print("2. Compare predictions against the green ground-truth boxes.")
print("3. Assign one error category and write one sentence explaining the likely cause.")
"""
        ),
        md(
            r"""
### Detection Exercises

Beginner:

- Use the Ultralytics training guide to identify the two lines that load a model and start training.
- Run the confidence threshold sweep and describe what changes from `0.10` to `0.80`.
- Complete the toy matching exercise: what is TP, FP, and FN at each confidence threshold?

Intermediate:

- Change `imgsz` from `320` to `416`.
- Change `DETECT_N_EPOCHS`, `lr0`, or `batch`, then compare `mAP50` and recall.
- Complete `coco_bbox_to_yolo_exercise(...)` above.
- Use the error taxonomy on two predicted images.

Advanced:

- Turn on `RUN_TINY_OVERFIT_LAB` and check whether the model can memorize 8 training images.
- Turn on the Megalodon advanced path, compare its predictions with `yolo11n.pt`, and decide whether the domain-specific checkpoint changes the failure modes.
- Fine-tune Megalodon for one or two epochs, then compare its early metrics with the generic checkpoint run.
- Explain how the precision-recall curve would move if the model became more conservative.
- Sketch how you would convert the whole `coco/subset.json` file into YOLO detection labels.
"""
        ),
        md(
            r"""
# Part 3: Instance Segmentation With YOLO

Segmentation means predicting regions or pixels. **Instance segmentation** upgrades a rectangle to a separate shape for each object instance. A mask can be viewed as a set of pixels

$$M \subseteq \Omega,\qquad \Omega=\{1,\dots,H\}\times\{1,\dots,W\}.$$

For multiple objects, the model predicts a set of class-mask-score triples:

$$\{(c_i,M_i,s_i)\}_{i=1}^m.$$

As in the detection section, the live labels focus on larger visible instances. The omitted tiny polygons are a useful reminder that label design is part of the modeling problem, not a boring preprocessing footnote.

The same IoU idea applies:

$$\operatorname{maskIoU}(M,N)=\frac{|M \cap N|}{|M \cup N|}.$$

In the YOLO segmentation text format, one row is

```text
class x1 y1 x2 y2 ... xn yn
```

where polygon coordinates are normalized to `[0,1]`.
"""
        ),
        md(
            r"""
### Section Bootstrap: Load The Segmentation Dataset

Start the segmentation section by loading the YOLO segmentation YAML and validating its labels. This is the same pattern as detection, but each label row now stores a polygon instead of a rectangle.
"""
        ),
        code(
            section_bootstrap("segmentation")
        ),
        md(
            r"""
### Segmentation Truth Example: Same Image, More Geometry

This plot shows **ground-truth polygon labels** from the YOLO segmentation dataset. When possible, it uses the same image as the detection example above so you can compare boxes with masks directly. The colored regions are annotation polygons, not model predictions.
"""
        ),
        code(
            r"""
import matplotlib.pyplot as plt

preferred_stem = globals().get("DETECT_EXAMPLE_STEM")
if preferred_stem is not None:
    candidate_label = SEGMENT_ROOT / "labels" / "val" / f"{preferred_stem}.txt"
    candidate_image = SEGMENT_ROOT / "images" / "val" / f"{preferred_stem}.jpg"
else:
    candidate_label = None
    candidate_image = None

if candidate_label is not None and candidate_label.exists() and candidate_image.exists():
    first_segment_label = candidate_label
    first_segment_image = candidate_image
    segment_instance_count = yolo_label_instance_count(first_segment_label)
    print(f"Showing the same image as detection: {preferred_stem}")
else:
    segment_example = select_yolo_examples(SEGMENT_ROOT, split="val", min_instances=2, limit=1)[0]
    first_segment_label = segment_example["label_path"]
    first_segment_image = segment_example["image_path"]
    segment_instance_count = segment_example["instance_count"]
    print(f"Showing {first_segment_image.stem}: {segment_instance_count} truth masks")

ax = draw_yolo_masks(first_segment_image, first_segment_label, class_names={0: "object"})
ax.set_title(f"Segmentation truth labels: {segment_instance_count} masks")
plt.show()
"""
        ),
        md(
            r"""
### Visible Helper: Mask IoU

This helper is intentionally simple. It assumes binary masks with equal shape. You can modify it to handle soft masks, ignored pixels, or batches.
"""
        ),
        code(
            r"""
def mask_iou(mask_a, mask_b):
    '''Compute IoU for two binary masks represented as nested lists or arrays.'''

    intersection = 0
    union = 0
    for row_a, row_b in zip(mask_a, mask_b):
        for value_a, value_b in zip(row_a, row_b):
            a = bool(value_a)
            b = bool(value_b)
            intersection += int(a and b)
            union += int(a or b)
    return intersection / union if union else 1.0


mask_a = [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
mask_b = [[1, 0, 0], [0, 1, 1], [0, 0, 0]]
print("mask IoU:", mask_iou(mask_a, mask_b))
"""
        ),
        md(
            r"""
### Advanced Exercise: Rasterize A YOLO Polygon

YOLO segmentation labels store polygons, but mask IoU is defined on pixels. To connect the two, rasterize one normalized polygon row into a binary mask.

Implement `yolo_polygon_row_to_mask(...)`.

Hints:

- a YOLO segmentation row is `class_id x1 y1 x2 y2 ...`;
- multiply normalized x coordinates by image width and y coordinates by image height;
- `PIL.ImageDraw.Draw(...).polygon(...)` can fill a polygon;
- the output can be a NumPy boolean array or nested list of `0/1` values.
"""
        ),
        code(
            r"""
def yolo_polygon_row_to_mask(row, image_width, image_height):
    '''Rasterize one YOLO segmentation polygon row into a binary mask.

    Leave the return value as `None` until you are ready to implement the
    advanced exercise.
    '''

    # TODO:
    # 1. Parse row[1:] as normalized x/y coordinate pairs.
    # 2. Convert normalized coordinates to pixel coordinates.
    # 3. Fill the polygon into a binary image.
    # 4. Return the mask.
    return None


example_segment_row = [float(part) for part in first_segment_label.read_text().splitlines()[0].split()]
from PIL import Image

with Image.open(first_segment_image) as example_segment_image:
    example_width, example_height = example_segment_image.size

candidate_mask = yolo_polygon_row_to_mask(example_segment_row, example_width, example_height)
if candidate_mask is None:
    print("Exercise ready: implement yolo_polygon_row_to_mask(...) to rasterize a polygon label.")
else:
    import numpy as np
    import matplotlib.pyplot as plt

    candidate_mask = np.asarray(candidate_mask).astype(bool)
    print("mask shape:", candidate_mask.shape, "mask pixels:", int(candidate_mask.sum()))
    plt.figure(figsize=(6, 4))
    plt.imshow(candidate_mask, cmap="gray")
    plt.axis("off")
    plt.title("Rasterized YOLO polygon")
    plt.show()
"""
        ),
        md(
            r"""
### Train Or Load A Small Segmentation Run

Now train the segmentation model, or load cached curves if live training is unavailable. The important difference from detection is that the report contains both box metrics and mask metrics. Compare them: a model can learn reasonable boxes before it learns precise masks. For a live run, the best validation checkpoint is saved as `weights/best.pt`.
"""
        ),
        code(
            r"""
SEGMENT_N_EPOCHS = 10
SEGMENT_ARGS = build_train_args(
    n_epochs=SEGMENT_N_EPOCHS,
    imgsz=320,
    batch=4,
    lr0=0.001,
    project=REPO_ROOT / "runs" / "segment",
    name="segment_default",
)

# Documentation checkpoint:
# 1. Open https://docs.ultralytics.com/datasets/segment/
# 2. Confirm the polygon-row format and the dataset YAML structure.
# 3. Open https://docs.ultralytics.com/modes/train/ for the same model-loading pattern.
#
# Your live exercise is to connect the segmentation dataset YAML to a pretrained
# segmentation model and then interpret both box mAP and mask mAP.
segmentation_training = None
segmentation_save_dir = None
segmentation_best_model_path = None
if RUN_LIVE_TRAINING:
    from ultralytics import YOLO

    segmentation_model = YOLO("yolo11n-seg.pt")
    segmentation_training = segmentation_model.train(data=str(SEGMENT_YAML), **SEGMENT_ARGS)
    segmentation_save_dir = getattr(segmentation_training, "save_dir", None)
    if segmentation_save_dir is None and getattr(segmentation_model, "trainer", None) is not None:
        segmentation_save_dir = getattr(segmentation_model.trainer, "save_dir", None)
    print("Live segmentation training finished.")
    candidate_best = Path(segmentation_save_dir) / "weights" / "best.pt" if segmentation_save_dir is not None else None
    if candidate_best is not None and candidate_best.exists():
        segmentation_best_model_path = candidate_best
        print(f"Best validation checkpoint: {segmentation_best_model_path}")
else:
    print("No GPU detected. Using cached segmentation training curve.")

segmentation_results_csv = (
    Path(segmentation_save_dir) / "results.csv"
    if segmentation_save_dir is not None
    else CACHED_SEGMENT_RESULTS
)
if not segmentation_results_csv.exists():
    segmentation_results_csv = CACHED_SEGMENT_RESULTS

plot_training_curves(
    segmentation_results_csv,
    metric_columns=["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/mAP50(M)", "metrics/mAP50-95(M)"],
    title="Segmentation: box mAP vs mask mAP",
)
"""
        ),
        md(
            r"""
### Exercise: COCO Polygons To YOLO Segmentation Rows

COCO segmentation annotations often store polygon vertices as one flat list:

```text
[x1, y1, x2, y2, ..., xn, yn]
```

YOLO segmentation wants the same geometry normalized by image width and height, preceded by the class id.

You do not get the full converter here. The point is to reason through the coordinate map first, then decide what edge cases a production converter needs to handle: multiple polygons, holes, tiny regions, invalid polygons, and categories you might merge.
"""
        ),
        code(
            r"""
def coco_polygon_to_yolo_row_exercise(class_id, polygon, image_width, image_height):
    '''Convert one COCO polygon list to one YOLO segmentation row.

    Return a list like `[class_id, x1, y1, x2, y2, ...]`.
    Leave the return value as `None` until you are ready to test your answer.
    '''

    # TODO: fill this in during the exercise.
    return None


example_polygon = [10, 10, 40, 10, 40, 30, 10, 30]
candidate_row = coco_polygon_to_yolo_row_exercise(0, example_polygon, image_width=100, image_height=100)

if candidate_row is None:
    print("Exercise ready: implement coco_polygon_to_yolo_row_exercise(...) in the segmentation lab.")
else:
    print("Your YOLO segmentation row:", candidate_row)
    assert len(candidate_row) >= 7
    assert len(candidate_row[1:]) % 2 == 0
    assert all(0 <= value <= 1 for value in candidate_row[1:])
"""
        ),
        md(
            r"""
### Segmentation Exercises

Beginner:

- Inspect two masks: one clean and one messy.
- Why can box mAP be higher than mask mAP?
- Use the Ultralytics segmentation docs to identify the minimum number of polygon points per object.

Intermediate:

- Change the confidence threshold and inspect the visual result.
- Complete `mask_iou(...)` for one extra edge case: two empty masks, two disjoint masks, or two identical masks.
- Complete `coco_polygon_to_yolo_row_exercise(...)` above.

Advanced:

- Complete `yolo_polygon_row_to_mask(...)` and compare the rasterized mask area with the polygon's bounding-box area.
- Use the coarse biological label plan from the source YOLO repo as an after-session extension.
- Ask whether binary "object" segmentation is a scientifically useful target, or only a stepping stone.
- Propose a rule for dropping or keeping tiny polygons, then predict how that rule will affect recall and mAP.
"""
        ),
        md(
            r"""
# Part 4: SAM3 Text-Prompt Segmentation Lab

SAM3 stands for **Segment Anything Model 3**. It is Meta's promptable segmentation foundation model for images and videos. "Promptable" means the model does not only take an image as input; it also takes a prompt that tells the model what kind of object or region you want. SAM3 can use text prompts, visual prompts such as points and boxes, and exemplar prompts. In this notebook you will focus on text prompts.

SAM3 changes the interface compared with the supervised YOLO models above. Instead of training a new model for every label set, you can ask for a concept:

```text
image + "fish" -> masks, boxes, scores
```

Mathematically, the text prompt becomes part of the input:

$$g_\phi(x,q)\rightarrow \{(M_i,b_i,s_i)\}_{i=1}^m.$$

That changes the main question from "how do you train a new supervised head?" to "which prompt and threshold define the object concept you actually want?"

This is useful for rapid exploration and for proposing pseudo-labels, but it is not magic. A prompt like `"fish"` may miss small or unusual fish; a broad prompt like `"marine organism"` may include objects that are not useful for your scientific question. You will inspect those failures directly.

The default path below uses cached SAM3-style outputs so everyone can complete the lab. If your runtime passes the live SAM3 checks and you have checkpoint access, install SAM3, enter a Hugging Face token, and let the notebook set `USE_LIVE_SAM3 = True`.
"""
        ),
        md(
            r"""
### Section Bootstrap: Load SAM3 Cache And Runtime Checks

This cell checks whether live SAM3 is possible and lists the cached prompt outputs available in the tutorial bundle. The rest of the section works even when live SAM3 is unavailable.
"""
        ),
        code(
            section_bootstrap("sam3")
        ),
        md(
            r"""
### Optional: Install SAM3 For Live Inference

The cached SAM3 path works without installing SAM3. Live SAM3 needs the official SAM3 package, a compatible Python/CUDA/PyTorch runtime, and checkpoint access. The bootstrap status above tells you which checks are failing. If `sam3_importable` is `false`, SAM3 is not installed in the current runtime.

Set `INSTALL_SAM3_FOR_LIVE = True` only if you want to try live SAM3. This can take several minutes in Colab. The install source is the official SAM3 GitHub repository: https://github.com/facebookresearch/sam3
"""
        ),
        code(
            r"""
INSTALL_SAM3_FOR_LIVE = False

if INSTALL_SAM3_FOR_LIVE:
    install_status = install_sam3_package()
    print(json.dumps(install_status, indent=2))
else:
    print("SAM3 install is off. Set INSTALL_SAM3_FOR_LIVE = True to install the official SAM3 package.")

SAM3_STATUS = sam3_can_run_live()
USE_LIVE_SAM3 = bool(SAM3_STATUS["can_run"])

print(json.dumps(SAM3_STATUS, indent=2))
print(f"USE_LIVE_SAM3 = {USE_LIVE_SAM3}")
"""
        ),
        md(
            r"""
### Optional: Hugging Face Token For Live SAM3

Live SAM3 checkpoint loading may require a Hugging Face account, checkpoint access, and a token. The cached path above does not need a token.

To create a token:

1. Sign in to Hugging Face.
2. Open https://huggingface.co/settings/tokens.
3. Click **New token**.
4. Choose a `read` token, or a fine-grained token with read access to the SAM3 model/checkpoint you plan to use.
5. Copy the token and paste it into the hidden input when you run the cell below.

Official Hugging Face token documentation: https://huggingface.co/docs/hub/security-tokens

Do not paste a real token into a saved notebook cell, markdown cell, chat, or GitHub issue. The cell below uses hidden input and stores the token only in this runtime's environment variables.
"""
        ),
        code(
            r"""
ENTER_HF_TOKEN_FOR_LIVE_SAM3 = False

if ENTER_HF_TOKEN_FOR_LIVE_SAM3:
    token_status = configure_huggingface_token()
    print(json.dumps(token_status, indent=2))
else:
    print("Token entry is off. Set ENTER_HF_TOKEN_FOR_LIVE_SAM3 = True to enter a token.")

SAM3_STATUS = sam3_can_run_live()
USE_LIVE_SAM3 = bool(SAM3_STATUS["can_run"])

print(json.dumps(SAM3_STATUS, indent=2))
print(f"USE_LIVE_SAM3 = {USE_LIVE_SAM3}")
if not USE_LIVE_SAM3:
    print("Using cached SAM3-style outputs for the cells below.")
"""
        ),
        md(
            r"""
### Visible Helper: Try A SAM3 Prompt

Edit `prompt` and `confidence_threshold`. The same function works with cached fallback outputs or live SAM3.
"""
        ),
        code(
            r"""
def try_sam3_prompt(image_id, prompt, confidence_threshold=0.5):
    '''Run or load SAM3-style text-prompt segmentation for one image.'''

    cached = load_cached_sam3_result(BUNDLE_ROOT, image_id=image_id, prompt=prompt)
    image_path = cached["image_path"]

    if USE_LIVE_SAM3:
        live = run_sam3_text_prompt(
            image_path,
            prompt,
            confidence_threshold=confidence_threshold,
        )
        live["image_path"] = image_path
        return live

    cached["confidence_threshold"] = confidence_threshold
    return cached


cached_prompt_table = available_cached_prompts(BUNDLE_ROOT)
SAM3_IMAGE_ID = next(iter(cached_prompt_table))
PROMPT = "marine organism"
CONFIDENCE_THRESHOLD = 0.5

sam3_result = try_sam3_prompt(SAM3_IMAGE_ID, PROMPT, CONFIDENCE_THRESHOLD)
plot_sam3_result(
    sam3_result["image_path"],
    sam3_result,
    score_threshold=CONFIDENCE_THRESHOLD,
    title=f"{SAM3_IMAGE_ID}: {PROMPT}",
)
"""
        ),
        md(
            r"""
### Compare Several Prompts

After one prompt works, try several prompts on the same cached image. This cell does not train anything; it only changes the text query and reports how many masks were returned for each concept.
"""
        ),
        code(
            r"""
# Try several prompts on the same image. Some may correctly return nothing.
for prompt in ["fish", "sponge", "gelatinous animal", "small crab", "echinoderm"]:
    result = try_sam3_prompt(SAM3_IMAGE_ID, prompt, confidence_threshold=0.5)
    print(prompt, "detections:", len(result.get("boxes", [])), "source:", result.get("source"))
"""
        ),
        md(
            r"""
### Advanced Exercise: Export SAM3 Pseudo-Labels

Promptable segmentation can be used to create **pseudo-labels**: labels produced by a model instead of a human. This can speed up annotation, but it can also copy the prompt model's mistakes into your supervised training set.

Implement `sam3_result_to_yolo_pseudo_labels(...)` so it converts cached SAM3-style polygons into YOLO segmentation rows.

Hints:

- `sam3_result["polygons"]` is a list of polygons;
- `sam3_result["scores"]` is aligned with those polygons;
- the cached polygons are already normalized to `[0, 1]`;
- filter out masks below `score_threshold`;
- each output row should look like `[class_id, x1, y1, x2, y2, ...]`.
"""
        ),
        code(
            r"""
def sam3_result_to_yolo_pseudo_labels(result, class_id=0, score_threshold=0.5):
    '''Convert SAM3-style polygons into YOLO segmentation pseudo-label rows.

    Leave the return value as `None` until you are ready to implement the
    advanced exercise.
    '''

    # TODO:
    # 1. Iterate over polygons and scores together.
    # 2. Skip polygons with scores below `score_threshold`.
    # 3. Flatten each polygon into a YOLO segmentation row.
    # 4. Return the list of rows.
    return None


pseudo_label_rows = sam3_result_to_yolo_pseudo_labels(
    sam3_result,
    class_id=0,
    score_threshold=CONFIDENCE_THRESHOLD,
)

if pseudo_label_rows is None:
    print("Exercise ready: implement sam3_result_to_yolo_pseudo_labels(...).")
else:
    print(f"pseudo-label rows: {len(pseudo_label_rows)}")
    for row in pseudo_label_rows[:3]:
        print(row)
"""
        ),
        md(
            r"""
### SAM3 Exercises

Beginner:

- Try `PROMPT = "fish"`, `"sponge"`, `"gelatinous animal"`, and `"small crab"`.
- Increase `CONFIDENCE_THRESHOLD` from `0.5` to `0.8`.
- Which prompts are too broad? Which are too specific?

Advanced:

- If live SAM3 works, compare cached fallback with live masks.
- Complete `sam3_result_to_yolo_pseudo_labels(...)`, then reason about when pseudo-label noise helps or hurts supervised training.
"""
        ),
        md(
            r"""
# Wrap-Up

You moved through four levels of supervision and geometry:

- classification: one image crop, one class;
- detection: objects as normalized boxes;
- segmentation: objects as masks or polygons;
- SAM3 prompting: segmentation conditioned on text.

Good after-session projects:

- train on the 500-image or full dataset after this notebook,
- convert the coarse biological label plan into a multiclass detection task,
- compare YOLO training against FathomNet pretrained checkpoints,
- use SAM3 prompt outputs to propose labels for active learning,
- extend from images to video tracking.

The main lesson is not that one model wins everywhere. The real modeling choice is the pair:

$$\text{scientific question} \quad + \quad \text{label geometry}.$$

Once that pair is clear, the model choice becomes much easier to reason about.
"""
        ),
    ]

    for index, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{index:03d}")

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    out_path = Path("notebooks/fathomnet_underwater_vision_tutorial.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(build_notebook(), handle, indent=1)
        handle.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
