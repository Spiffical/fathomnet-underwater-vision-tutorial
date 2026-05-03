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
    section_bootstrap = r"""
# Common skip-ahead bootstrap. This block lets a section run even if you did
# not execute the setup cells above.
from pathlib import Path
import json
import subprocess
import sys

GITHUB_REPO_URL = "https://github.com/Spiffical/fathomnet-underwater-vision-tutorial"
GITHUB_BRANCH = "main"
PROJECT_DIR_NAME = "fathomnet_underwater_vision_tutorial"
BUNDLE_URL = "https://github.com/Spiffical/fathomnet-underwater-vision-tutorial/raw/main/data/fathomnet_underwater_tutorial_bundle.zip"

if "REPO_ROOT" not in globals():
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
            subprocess.check_call(["git", "clone", "--depth", "1", "--branch", GITHUB_BRANCH, f"{GITHUB_REPO_URL}.git", str(clone_dir)])
        REPO_ROOT = clone_dir
    if REPO_ROOT is None:
        raise RuntimeError("Could not find the tutorial repo root. Start Jupyter from the cloned repo.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tutorial_setup import detect_runtime, download_tutorial_bundle, ensure_dependencies

if "LOCAL_BUNDLE_ZIP" not in globals():
    LOCAL_BUNDLE_ZIP = REPO_ROOT / "data" / "fathomnet_underwater_tutorial_bundle.zip"

ensure_dependencies(install=("google.colab" in sys.modules), extra_pip_args=("--quiet",))

if "BUNDLE_ROOT" not in globals():
    BUNDLE_ROOT = download_tutorial_bundle(
        bundle_url=BUNDLE_URL or None,
        bundle_zip_path=LOCAL_BUNDLE_ZIP if LOCAL_BUNDLE_ZIP.exists() else None,
        output_dir=REPO_ROOT / "data" / "fathomnet_underwater_tutorial_bundle",
    )

if "RUN_LIVE_TRAINING" not in globals():
    RUN_LIVE_TRAINING = bool(detect_runtime().get("has_cuda", False))
if "SMOKE_EPOCHS" not in globals():
    SMOKE_EPOCHS = 1

if "build_train_args" not in globals():
    def build_train_args(
        *,
        epochs=1,
        imgsz=320,
        batch=8,
        lr0=0.001,
        patience=3,
        workers=0,
        project="runs/tutorial",
        name="experiment",
        seed=42,
    ):
        '''Compact fallback copy of the visible helper from the setup section.'''

        return {
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "lr0": float(lr0),
            "patience": int(patience),
            "workers": int(workers),
            "project": str(project),
            "name": str(name),
            "seed": int(seed),
            "verbose": False,
        }

"""
    cells = [
        md(
            r"""
# Underwater Computer Vision With FathomNet

This notebook is a 3-hour, Colab-first tutorial for training and comparing image recognition, object detection, instance segmentation, and promptable segmentation workflows on underwater imagery.

You only need a mathematical foundation to start. You can follow the guided path from classification to segmentation, or jump into the section that matches your comfort level.

Main source anchors:

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
## How To Use This Notebook

Suggested live timing:

1. `0-15 min`: setup, runtime check, and skip map.
2. `15-30 min`: YOLO warm-up on a familiar image.
3. `30-50 min`: FathomNet context and dataset exploration.
4. `50-75 min`: classification from organism crops.
5. `75-115 min`: binary object detection with YOLO.
6. `115-150 min`: instance segmentation with YOLO.
7. `150-175 min`: SAM3 text-prompt segmentation.
8. `175-180 min`: wrap-up and extension ideas.

The sections are intentionally independent. If you skip ahead, run the short **section bootstrap** cell at the top of that section.

The default notebook behavior is:

- use a prebuilt compact data bundle,
- run live YOLO training only when a GPU is available,
- fall back to cached training curves and cached SAM3-style outputs when a GPU, model weights, or Hugging Face access are unavailable.
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

- The canonical workshop bundle is `data/fathomnet_underwater_tutorial_bundle.zip` in the instructor/project copy.
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
SMOKE_EPOCHS = 1

ensure_dependencies(install=INSTALL_DEPENDENCIES, extra_pip_args=("--quiet",))
RUNTIME = print_runtime_summary(detect_runtime())
RUN_LIVE_TRAINING = bool(RUNTIME.get("has_cuda", False))
set_reproducible_seed(42)

print(f"RUN_LIVE_TRAINING = {RUN_LIVE_TRAINING}")
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
### Visible Helper: Training Arguments

You are encouraged to edit this function. It is deliberately small: the goal is to connect a few knobs to training behavior.
"""
        ),
        code(
            r"""
def build_train_args(
    *,
    epochs=1,
    imgsz=320,
    batch=8,
    lr0=0.001,
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
    the gradient estimate is, and `epochs` controls how many passes you make
    through the finite training sample.
    '''

    return {
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "lr0": float(lr0),
        "patience": int(patience),
        "workers": int(workers),
        "project": str(project),
        "name": str(name),
        "seed": int(seed),
        "verbose": False,
    }
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
## FathomNet And Underwater Imagery

Now switch domains. The [FathomNet Database](https://database.fathomnet.org/fathomnet/#/) is the source you should open when you want to see the broader image archive, taxonomy, and annotation context behind this tutorial. The FathomNet site describes the database as expert-annotated, machine-learning-ready underwater visual data for marine science and computer vision.

Underwater imagery is not just "ImageNet, but blue." Common modeling complications include:

- changing illumination from vehicle lights and water-column attenuation;
- scale changes from vehicle altitude, camera angle, and organism distance;
- motion blur, turbidity, backscatter, and low contrast;
- partial animals, occlusion, and organisms with non-rigid transparent bodies;
- sparse or incomplete labels, where unlabeled organisms may still be present;
- a long tail of rare taxa.

That is why this workshop starts with a compact teaching bundle rather than the full archive. You can focus on model behavior first, then scale up after the session.
"""
        ),
        md(
            r"""
## Dataset Exploration

The bundle has four useful views of the same general problem:

- `classification_crops`: organism crops grouped by coarse class.
- `yolo_detect_binary`: one-class object detection labels.
- `yolo_segment_binary`: one-class instance segmentation polygon labels.
- `sam3_cached_outputs`: SAM3-like prompt outputs for the fallback lab.

The full source COCO-style annotation file in the local YOLO project has `119,096` images, `280,118` annotations, and `1,897` categories. This tutorial uses a compact subset so you can work in Colab without spending the session on data transfer.

For the live detection and segmentation exercises, the YOLO labels deliberately omit extremely tiny annotations. That keeps the training signal about visible organisms rather than letting dense fragments dominate the classroom metrics. The original COCO subset is still included for discussion and after-session extensions.

The images and annotations are FathomNet-derived. The bundle includes `licenses/attribution.csv`; visual content is subject to contributor-selected Creative Commons terms and is used here for machine-learning training and development.
"""
        ),
        code(
            r"""
from collections import Counter

from scripts.tutorial_data import (
    get_task_paths,
    load_manifest,
    summarize_classification_dataset,
    summarize_dataset,
    validate_yolo_dataset,
)
from scripts.tutorial_viz import draw_yolo_boxes, show_image_grid

manifest = load_manifest(BUNDLE_ROOT)
print(json.dumps({
    "name": manifest["name"],
    "workshop_classes": manifest["workshop_classes"],
    "stats": manifest["stats"],
}, indent=2))
"""
        ),
        code(
            r"""
detect_paths = get_task_paths("detect", BUNDLE_ROOT)
segment_paths = get_task_paths("segment", BUNDLE_ROOT)
classification_paths = get_task_paths("classification", BUNDLE_ROOT)
coco_paths = get_task_paths("coco", BUNDLE_ROOT)

underwater_images = sorted((detect_paths["root"] / "images" / "val").glob("*.jpg"))[:8]
show_image_grid(
    underwater_images,
    titles=[path.stem[:8] for path in underwater_images],
    columns=4,
)
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
        code(
            r"""
import matplotlib.pyplot as plt

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
        code(
            r"""
example_labels = sorted((detect_paths["root"] / "labels" / "val").glob("*.txt"))[:4]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, label_path in zip(axes.flat, example_labels):
    image_path = detect_paths["root"] / "images" / "val" / f"{label_path.stem}.jpg"
    draw_yolo_boxes(image_path, label_path, class_names={0: "object"}, ax=ax)
    ax.set_title(label_path.stem[:8])
plt.tight_layout()
plt.show()
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

Advanced:

- The live YOLO labels drop extremely tiny boxes. Predict how the histogram and mAP would change if those boxes were restored.
"""
        ),
        md(
            r"""
# Part 1: Classification From Organism Crops

This is the easiest entry point because each example has one label. It is still not trivial: underwater crops can be low contrast, partially occluded, visually ambiguous, and taxonomically long-tailed.

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
        code(
            section_bootstrap + r"""
# Section bootstrap: classification
from scripts.tutorial_data import get_task_paths, summarize_classification_dataset
from scripts.tutorial_models import run_classification_lr_trial
from scripts.tutorial_viz import plot_confusion_matrix, plot_training_curves, show_image_grid

CLASSIFY_PATHS = get_task_paths("classification", BUNDLE_ROOT)
CLASSIFY_ROOT = CLASSIFY_PATHS["root"]
CACHED_CLASSIFY_RESULTS = BUNDLE_ROOT / "cached_training" / "classification" / "results.csv"

print(json.dumps(summarize_classification_dataset(CLASSIFY_ROOT), indent=2))
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

show_image_grid(class_examples, titles=class_titles, columns=5)
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
        code(
            r"""
CLASSIFY_ARGS = build_train_args(
    epochs=SMOKE_EPOCHS,
    imgsz=224,
    batch=16,
    lr0=0.001,
    project=REPO_ROOT / "runs" / "classification",
    name="classification_smoke",
)

# Documentation checkpoint:
# 1. Open https://docs.ultralytics.com/tasks/classify/
# 2. Find how a pretrained classification model is loaded.
# 3. Compare that with the general training pattern at https://docs.ultralytics.com/modes/train/
#
# Your live exercise is to modify the model name or train arguments below.
classification_training = None
classification_save_dir = None
if RUN_LIVE_TRAINING:
    from ultralytics import YOLO

    classification_model = YOLO("yolo11n-cls.pt")
    classification_training = classification_model.train(data=str(CLASSIFY_ROOT), **CLASSIFY_ARGS)
    classification_save_dir = getattr(classification_training, "save_dir", None)
    if classification_save_dir is None and getattr(classification_model, "trainer", None) is not None:
        classification_save_dir = getattr(classification_model.trainer, "save_dir", None)
    print("Live classification training finished.")
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

For live groups, split the room across `lr0 = 1e-4`, `1e-3`, and `1e-2`. Each group runs one setting, then you compare results on the board.
"""
        ),
        code(
            r"""
RUN_CLASSIFICATION_LR_LAB = False
LR_VALUES = [1e-4, 1e-3, 1e-2]

if RUN_CLASSIFICATION_LR_LAB and RUN_LIVE_TRAINING:
    lr_rows = [
        run_classification_lr_trial(
            CLASSIFY_ROOT,
            build_train_args,
            lr0=lr0,
            repo_root=REPO_ROOT,
            epochs=1,
        )
        for lr0 in LR_VALUES
    ]
    for row in lr_rows:
        print(row)
        plot_training_curves(row["results_csv"], title=f"classification lr0={row['lr0']}")
else:
    print("Learning-rate lab is ready.")
    print("Set RUN_CLASSIFICATION_LR_LAB = True on a GPU, or assign one lr0 value to each group.")
    print("Suggested lr0 values:", LR_VALUES)
"""
        ),
        code(
            r"""
# A small confusion matrix for discussion. Replace this with predictions from
# your live run if you want to make the exercise more advanced.
classes = ["echinoderm", "fish", "gelatinous", "crustacean", "sponge_coral"]
toy_confusion = [
    [8, 0, 1, 0, 1],
    [0, 6, 1, 1, 0],
    [1, 0, 7, 0, 2],
    [0, 2, 0, 5, 1],
    [1, 0, 2, 0, 8],
]
plot_confusion_matrix(toy_confusion, classes, normalize=True, title="Discussion matrix: normalized by true class")
"""
        ),
        md(
            r"""
### Classification Exercises

Beginner:

- Change `epochs` from `1` to `2`.
- Run or reason through one learning-rate trial.
- Decide whether the validation curve is improving, noisy, or overfitting.

Intermediate:

- Use the Ultralytics classification docs to find the direct `YOLO(...).train(...)` pattern for classification.
- Compare your group's `lr0` result with another group's result.
- Change `imgsz` from `224` to `320`, then decide whether the extra computation was worth it.
- Pick one class and inspect three crops that seem visually ambiguous.

Advanced:

- Design a fair pretrained-vs-random-initialization comparison. What would you keep fixed?
- Replace the toy confusion matrix with predictions from your live model.
- Which two classes are most confusable, and what visual ambiguity might explain it?
- Propose a better class grouping for this small crop dataset.
"""
        ),
        md(
            r"""
# Part 2: Binary Object Detection With YOLO

Detection asks for *where* the animals or biological structures are, not only what crop class they belong to.

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

Average precision summarizes the precision-recall curve as that threshold changes.
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
        code(
            section_bootstrap + r"""
# Section bootstrap: detection
from scripts.tutorial_data import get_task_paths, make_tiny_detection_dataset, validate_yolo_dataset
from scripts.tutorial_models import prediction_count_at_threshold
from scripts.tutorial_viz import draw_yolo_boxes, plot_training_curves

DETECT_PATHS = get_task_paths("detect", BUNDLE_ROOT)
DETECT_ROOT = DETECT_PATHS["root"]
DETECT_YAML = DETECT_PATHS["yaml"]
CACHED_DETECT_RESULTS = BUNDLE_ROOT / "cached_training" / "detection" / "results.csv"

print(json.dumps(validate_yolo_dataset(DETECT_YAML, task="detect"), indent=2)[:3000])
"""
        ),
        code(
            r"""
first_detect_label = next((DETECT_ROOT / "labels" / "val").glob("*.txt"))
first_detect_image = DETECT_ROOT / "images" / "val" / f"{first_detect_label.stem}.jpg"
draw_yolo_boxes(first_detect_image, first_detect_label, class_names={0: "object"})
"""
        ),
        code(
            r"""
DETECT_ARGS = build_train_args(
    epochs=SMOKE_EPOCHS,
    imgsz=320,
    batch=8,
    lr0=0.001,
    project=REPO_ROOT / "runs" / "detect",
    name="detect_smoke",
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
if RUN_LIVE_TRAINING:
    from ultralytics import YOLO

    detection_model = YOLO("yolo11n.pt")
    detection_training = detection_model.train(data=str(DETECT_YAML), **DETECT_ARGS)
    detection_save_dir = getattr(detection_training, "save_dir", None)
    if detection_save_dir is None and getattr(detection_model, "trainer", None) is not None:
        detection_save_dir = getattr(detection_model.trainer, "save_dir", None)
    print("Live detection training finished.")
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
        Path(detection_save_dir) / "weights" / "best.pt"
        if detection_save_dir is not None and (Path(detection_save_dir) / "weights" / "best.pt").exists()
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

    tiny_args = build_train_args(
        epochs=12,
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
- Change `epochs`, `lr0`, or `batch`, then compare `mAP50` and recall.
- Complete `coco_bbox_to_yolo_exercise(...)` above.
- Use the error taxonomy on two predicted images.

Advanced:

- Turn on `RUN_TINY_OVERFIT_LAB` and check whether the model can memorize 8 training images.
- Compare a generic YOLO checkpoint with a FathomNet detector such as Megalodon if you have downloaded the weights.
- Explain how the precision-recall curve would move if the model became more conservative.
- Sketch how you would convert the whole `coco/subset.json` file into YOLO detection labels.
"""
        ),
        md(
            r"""
# Part 3: Instance Segmentation With YOLO

Instance segmentation upgrades a rectangle to a shape. A mask can be viewed as a set of pixels

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
        code(
            section_bootstrap + r"""
# Section bootstrap: segmentation
from scripts.tutorial_data import get_task_paths, validate_yolo_dataset
from scripts.tutorial_viz import draw_yolo_masks, plot_training_curves

SEGMENT_PATHS = get_task_paths("segment", BUNDLE_ROOT)
SEGMENT_ROOT = SEGMENT_PATHS["root"]
SEGMENT_YAML = SEGMENT_PATHS["yaml"]
CACHED_SEGMENT_RESULTS = BUNDLE_ROOT / "cached_training" / "segmentation" / "results.csv"

print(json.dumps(validate_yolo_dataset(SEGMENT_YAML, task="segment"), indent=2)[:3000])
"""
        ),
        code(
            r"""
first_segment_label = next((SEGMENT_ROOT / "labels" / "val").glob("*.txt"))
first_segment_image = SEGMENT_ROOT / "images" / "val" / f"{first_segment_label.stem}.jpg"
draw_yolo_masks(first_segment_image, first_segment_label, class_names={0: "object"})
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
        code(
            r"""
SEGMENT_ARGS = build_train_args(
    epochs=SMOKE_EPOCHS,
    imgsz=320,
    batch=4,
    lr0=0.001,
    project=REPO_ROOT / "runs" / "segment",
    name="segment_smoke",
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
if RUN_LIVE_TRAINING:
    from ultralytics import YOLO

    segmentation_model = YOLO("yolo11n-seg.pt")
    segmentation_training = segmentation_model.train(data=str(SEGMENT_YAML), **SEGMENT_ARGS)
    segmentation_save_dir = getattr(segmentation_training, "save_dir", None)
    if segmentation_save_dir is None and getattr(segmentation_model, "trainer", None) is not None:
        segmentation_save_dir = getattr(segmentation_model.trainer, "save_dir", None)
    print("Live segmentation training finished.")
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

- Use the coarse biological label plan from the source YOLO repo as an after-session extension.
- Ask whether binary "object" segmentation is a scientifically useful target, or only a stepping stone.
- Propose a rule for dropping or keeping tiny polygons, then predict how that rule will affect recall and mAP.
"""
        ),
        md(
            r"""
# Part 4: SAM3 Text-Prompt Segmentation Lab

SAM3 changes the interface. Instead of training a new model for every label set, you can ask for a concept:

```text
image + "fish" -> masks, boxes, scores
```

Mathematically, the text prompt becomes part of the input:

$$g_\phi(x,q)\rightarrow \{(M_i,b_i,s_i)\}_{i=1}^m.$$

That changes the main question from "how do you train a new supervised head?" to "which prompt and threshold define the object concept you actually want?"

The default path below uses cached SAM3-like outputs so everyone can complete the lab. If your runtime passes the live SAM3 checks and you have checkpoint access, set `USE_LIVE_SAM3 = True`.
"""
        ),
        code(
            section_bootstrap + r"""
# Section bootstrap: SAM3
from scripts.tutorial_sam3 import (
    available_cached_prompts,
    load_cached_sam3_result,
    run_sam3_text_prompt,
    sam3_can_run_live,
)
from scripts.tutorial_viz import plot_sam3_result

SAM3_STATUS = sam3_can_run_live()
USE_LIVE_SAM3 = False

print(json.dumps(SAM3_STATUS, indent=2))
print(json.dumps(available_cached_prompts(BUNDLE_ROOT), indent=2)[:3000])
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
### SAM3 Exercises

Beginner:

- Try `PROMPT = "fish"`, `"sponge"`, `"gelatinous animal"`, and `"small crab"`.
- Increase `CONFIDENCE_THRESHOLD` from `0.5` to `0.8`.
- Which prompts are too broad? Which are too specific?

Advanced:

- If live SAM3 works, compare cached fallback with live masks.
- Try using SAM3 outputs as pseudo-labels, then reason about when pseudo-label noise helps or hurts supervised training.
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

- train on the 500-image or full dataset outside the 3-hour tutorial,
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
