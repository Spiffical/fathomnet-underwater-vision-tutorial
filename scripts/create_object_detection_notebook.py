"""Generate the bounding-box-first FathomNet tutorial notebook."""

from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def build_notebook() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    cells = [
        md(
            r"""
# Underwater Object Detection with FathomNet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Spiffical/fathomnet-underwater-vision-tutorial/blob/main/notebooks/fathomnet_underwater_object_detection.ipynb)

This self-contained tutorial teaches **object detection**: finding each object in an image and drawing a rectangular **bounding box** around it.

![An underwater scene shown first without annotations and then with all seven available bounding boxes](https://raw.githubusercontent.com/Spiffical/fathomnet-underwater-vision-tutorial/main/docs/images/object_detection_overview.png)

*The same FathomNet-derived scene is shown raw and with reference annotations. Each rectangle localizes one object instance; repeated class names are expected when several sponges are present.*

The images and annotations are derived from the [FathomNet Database](https://database.fathomnet.org/fathomnet/#/), a global, expert-annotated collection created to support marine science and underwater machine learning. Underwater imagery is unusually challenging because water changes colour and contrast, artificial lights create uneven illumination, animals may be small or partly hidden, and the seafloor contains complex textures.

The central investigation is **domain shift**, meaning that the images used to develop a model differ from the images on which we want to use it:

1. an ordinary pretrained **You Only Look Once (YOLO)** detector works well on a familiar ground-level photograph;
2. the unchanged model misses or misnames underwater organisms;
3. we inspect YOLO bounding-box annotations and **fine-tune** the same model—continue its training—on 32 positive underwater images plus three reviewed background frames;
4. we download and compare a larger FathomNet-trained YOLO model called Megalodon; and
5. we finish with **Segment Anything Model 3 (SAM3)**, which can use text prompts to request object masks.

The appendices extend the same workflow to class-specific detection and conventional instance segmentation. Review [FathomNet data use](https://www.fathomnet.org/datause) and the bundle's `licenses/attribution.csv` before redistributing imagery.

Reference links:

- FathomNet Database: https://database.fathomnet.org/fathomnet/#/
- FathomNet data use policy: https://www.fathomnet.org/datause
- Ultralytics prediction guide: https://docs.ultralytics.com/modes/predict/
- Ultralytics training guide: https://docs.ultralytics.com/modes/train/
- Ultralytics detection format: https://docs.ultralytics.com/datasets/detect/
- Ultralytics segmentation format: https://docs.ultralytics.com/datasets/segment/
- Megalodon 2024 YOLO11 checkpoint: https://huggingface.co/FathomNet/megalodon-2024-yolov11
- SAM3 repository: https://github.com/facebookresearch/sam3
- SAM3 image predictor example: https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb
"""
        ),
        md(
            r"""
## Learning goals

By the end, you should be able to:

- distinguish one-class detection, multi-class detection, and instance-segmentation targets;
- read a YOLO detection row: `class_id x_center y_center width height`;
- explain why a detector trained on general photographs can fail under domain shift;
- fine-tune vanilla `yolo11n.pt` on a small underwater dataset and interpret the result cautiously;
- download and use a recent FathomNet YOLO checkpoint (a saved model file); and
- describe what is new about text-prompted SAM3 segmentation.

Prerequisites: comfort with Python, Git, and Jupyter or Google Colab. No prior computer-vision experience is assumed.

Run the notebook from top to bottom because later cells reuse variables created earlier. Questions are followed immediately by answer cells for independent study. The confidence-threshold exercise and both appendices provide optional extensions to the main detection workflow.
"""
        ),
        md(
            r"""
## Essential vocabulary

- A **model** is a function with adjustable numbers that maps an input, such as an image, to predictions. Those learned numbers are called **parameters** or **weights**.
- **Training** adjusts the weights using labelled examples. **Inference** uses fixed weights to make predictions on new images.
- A **pretrained model** has already learned from another dataset. **Fine-tuning** continues training it on a new dataset or task. **Vanilla YOLO** means the ordinary general-image checkpoint before any underwater fine-tuning.
- A **class** is a category the model can predict. A **label** or **annotation** is the recorded answer for one training example. **Ground truth** means the annotations used as the reference when evaluating predictions. Ground truth can still be incomplete or filtered, so its scope must be stated.
- **Object detection** predicts a class and bounding box for each object. **Segmentation** predicts pixels or regions; **instance segmentation** predicts a separate region, called a **mask**, for each individual object.
- **YOLO** stands for **You Only Look Once**. In this notebook it refers to Ultralytics YOLO models, which predict many object boxes in one pass through an image.
- **COCO** stands for **Common Objects in Context**, a standard dataset used to train and compare vision models. It contains 80 everyday object classes such as person, tie, and bird. The vanilla YOLO checkpoint used here was trained on those classes.
- A **confidence score** is the model's strength of belief in one prediction. A **confidence threshold** hides predictions below a chosen score; it does not turn the remaining predictions into guaranteed facts.
- A **checkpoint** is a file containing saved model weights. PyTorch checkpoints commonly end in `.pt`.

Terms specific to data splits, optimisation, and evaluation are introduced immediately before they are used.
"""
        ),
        md(
            r"""
## 1. Setup

The first cell locates the project root. In Colab it clones the repository into `/content` when needed; locally, launch Jupyter from the repository or one of its subdirectories.
"""
        ),
        md(
            r"""
### In Google Colab: select a GPU

Before running any setup code:

1. Choose **Runtime → Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** or another available NVIDIA GPU.
3. Click **Save**, then run the notebook from the top.

![Google Colab Change runtime type dialog with T4 GPU selected](https://raw.githubusercontent.com/Spiffical/fathomnet-underwater-vision-tutorial/main/docs/images/colab_gpu_runtime.png)

The setup report should identify a CUDA device and print `Live fine-tuning: True`. GPU availability depends on your Colab account and current capacity. If no GPU is available, the notebook still runs using its clearly labelled saved training results.
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

# Search upward so local launches work from either the repository root or a subdirectory.
def candidate_roots():
    cwd = Path.cwd().resolve()
    yield cwd
    yield from cwd.parents
    yield Path("/content") / PROJECT_DIR_NAME

REPO_ROOT = next(
    (path for path in candidate_roots() if (path / "scripts").exists() and (path / "notebooks").exists()),
    None,
)

# A fresh Colab runtime does not contain the repository, so clone it only when needed.
if REPO_ROOT is None and "google.colab" in sys.modules:
    REPO_ROOT = Path("/content") / PROJECT_DIR_NAME
    if not REPO_ROOT.exists():
        subprocess.check_call([
            "git", "clone", "--depth", "1", "--branch", GITHUB_BRANCH,
            f"{GITHUB_REPO_URL}.git", str(REPO_ROOT),
        ])

if REPO_ROOT is None:
    raise RuntimeError("Repository not found. Start Jupyter inside the cloned tutorial repository.")

# Put this repository on Python's import path so the tutorial helper modules are available.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print(f"Repository: {REPO_ROOT}")
"""
        ),
        md(
            r"""
### Dependencies, runtime, and data

A **runtime** is the Python process and hardware executing the notebook. The main software packages are:

- **PyTorch:** the machine-learning framework that stores the model and computes weight updates;
- **Torchvision:** image utilities designed to work with PyTorch; and
- **Ultralytics:** the Python package that provides the YOLO model, training, prediction, and evaluation interface used below.

Colab installs the tested, **pinned** versions—fixed versions chosen for reproducibility—of PyTorch 2.11, Torchvision 0.26, and Ultralytics 8.4.41. Local users should first run `pip install -r requirements.txt` in a Python 3.12 environment.

Training is fastest on a **graphics processing unit (GPU)**. NVIDIA GPUs use **Compute Unified Device Architecture (CUDA)**; Apple Silicon uses **Metal Performance Shaders (MPS)**. The setup chooses CUDA first, then MPS, then a **central processing unit (CPU)**.

CPU training is optional because its speed varies widely. The complete 35-epoch recipe took about 1.5 minutes on one recent MacBook CPU, but other computers may take substantially longer. CPU-only runs therefore use the saved reference result by default. To try live CPU fine-tuning, change `RUN_CPU_FINE_TUNING` to `True` in the setup cell below. GPU and MPS runtimes train live automatically.

The setup fixes the random seed at `42` to reduce run-to-run variation, although small numerical differences can still occur across hardware.
"""
        ),
        code(
            r"""
from scripts.tutorial_setup import (
    dependency_versions,
    download_tutorial_bundle,
    ensure_dependencies,
    set_reproducible_seed,
)

IN_COLAB = "google.colab" in sys.modules
# Colab installs the pinned requirements; local environments are prepared from requirements.txt.
DEPENDENCY_STATUS = ensure_dependencies(install=IN_COLAB, extra_pip_args=("--quiet",))
print(json.dumps(dependency_versions(), indent=2))

import torch
from ultralytics import YOLO

RUN_CPU_FINE_TUNING = False  # CPU students: change to True to run the 35-epoch fine-tune.

# Prefer the fastest supported accelerator while keeping one DEVICE variable for later calls.
if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_LABEL = torch.cuda.get_device_name(0)
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
    DEVICE_LABEL = "Apple Metal Performance Shaders (MPS)"
else:
    DEVICE = "cpu"
    DEVICE_LABEL = "CPU"

RUN_LIVE_FINE_TUNING = DEVICE != "cpu" or RUN_CPU_FINE_TUNING
# Fix the sampling and initialization seed so reruns are easier to compare.
set_reproducible_seed(42)

# Reuse a verified local bundle when present; otherwise download and extract it once.
BUNDLE_ROOT = download_tutorial_bundle(
    bundle_url=(
        "https://github.com/Spiffical/fathomnet-underwater-vision-tutorial/raw/main/"
        "data/fathomnet_underwater_tutorial_bundle.zip"
    ),
    bundle_zip_path=REPO_ROOT / "data" / "fathomnet_underwater_tutorial_bundle.zip",
    output_dir=REPO_ROOT / "data" / "fathomnet_underwater_tutorial_bundle",
)

print(f"Device: {DEVICE_LABEL}")
print(f"CPU fine-tuning opt-in: {RUN_CPU_FINE_TUNING}")
print(f"Live fine-tuning: {RUN_LIVE_FINE_TUNING}")
print(f"Data bundle: {BUNDLE_ROOT}")
"""
        ),
        md(
            r"""
### Setup success checks and common fixes

The cell is ready when it prints a device, the CPU opt-in state, `Live fine-tuning: True` or `False`, and a path to the data bundle.

- The first run needs internet access for model checkpoints and, in Colab, the repository.
- If package installation requests a Colab restart, restart the session and rerun from the top.
- On CPU, YOLO inference remains live. Fine-tuning uses the labelled saved reference path unless `RUN_CPU_FINE_TUNING = True`.
"""
        ),
        md(
            r"""
## 2. Two underwater scenes

We will reuse two evaluation scenes throughout the lesson so changes are easy to see. Neither scene is included in the 35-image fine-tuning set.

- The first has a fish plus several structurally different **benthic** (seafloor-associated) organisms, including sponge-like forms.
- The second has two visibly distinct crustaceans: a crab and a smaller isopod-like animal. Isopods are related to crabs and shrimp.

We chose these scenes because each contains several visibly different creatures. That makes misses, duplicate boxes, incorrect labels, and differences between text prompts easier to spot.
"""
        ),
        code(
            r"""
from scripts.tutorial_data import get_task_paths
from scripts.tutorial_viz import show_image_grid

DETECT_ROOT = get_task_paths("detect", BUNDLE_ROOT)["root"]
# Keep IDs, images, and labels in matching order for the side-by-side comparisons below.
SCENE_IDS = [
    "74e42388-95e0-4be4-8bd0-230bca26e682",
    "e7371b25-f074-48ac-a6ef-b16ea241a4aa",
]
SCENE_IMAGES = [DETECT_ROOT / "images" / "val" / f"{scene_id}.jpg" for scene_id in SCENE_IDS]
SCENE_LABELS = [DETECT_ROOT / "labels" / "val" / f"{scene_id}.txt" for scene_id in SCENE_IDS]

show_image_grid(
    SCENE_IMAGES,
    titles=["fish, sponges, and other benthos", "crab and isopod-like animal"],
    columns=2,
    figsize=(13, 5),
)
"""
        ),
        md(
            r"""
## 3. Cat-café warm-up: what vanilla YOLO already knows

`yolo11n.pt` is the filename of the starting checkpoint: `11` identifies the YOLO model generation, `n` means the small **nano** model size, and `.pt` identifies a PyTorch checkpoint. It is pretrained on the 80 COCO object classes. This café scene contains several familiar examples: two cats, multiple chairs, and a dining table. A detector can return several instances of the same class, so each visible cat or chair can receive its own box.

This cell uses the detection checkpoint, which predicts boxes. The related `yolo11n-seg.pt` checkpoint can also predict a pixel mask for each supported object instance. We begin with boxes because they are the focus of the main workflow.

In the prediction call, `conf=0.25` sets the confidence threshold, `imgsz=768` sets the working inference size, and `device=DEVICE` selects the available hardware. Predictions below the confidence threshold are hidden. The slightly larger inference size helps preserve the smaller cat and background chairs while remaining quick on a CPU.

Image credit: [“Cat cafe vilnius 01”](https://commons.wikimedia.org/wiki/File:Cat_cafe_vilnius_01.jpg) by Sh.aliaksei, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
"""
        ),
        code(
            r"""
from urllib.request import Request, urlopen
import matplotlib.pyplot as plt

ASSET_DIR = REPO_ROOT / "tmp" / "object_detection_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)
WARMUP_IMAGE = ASSET_DIR / "cat_cafe_vilnius.jpg"
# Cache the external image so rerunning the cell does not download it again.
if not WARMUP_IMAGE.exists():
    warmup_image_request = Request(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/"
        "Cat_cafe_vilnius_01.jpg/1280px-Cat_cafe_vilnius_01.jpg",
        headers={"User-Agent": "FathomNetUnderwaterVisionTutorial/1.0"},
    )
    with urlopen(warmup_image_request) as response, WARMUP_IMAGE.open("wb") as destination:
        destination.write(response.read())

vanilla_model = YOLO("yolo11n.pt")
# predict() returns one Results object per input image; [0] selects our single image.
warmup_result = vanilla_model.predict(
    WARMUP_IMAGE, imgsz=768, conf=0.25, device=DEVICE, verbose=False
)[0]

print([
    (warmup_result.names[int(class_id)], round(float(score), 2))
    for class_id, score in zip(warmup_result.boxes.cls, warmup_result.boxes.conf)
])
plt.figure(figsize=(8, 8))
# Ultralytics returns blue-green-red (BGR) channel order; Matplotlib expects
# red-green-blue (RGB), so [..., ::-1] reverses the three colour channels.
plt.imshow(warmup_result.plot()[..., ::-1])
plt.axis("off")
plt.title("Vanilla YOLO11n: familiar COCO objects in a cat café")
plt.show()
"""
        ),
        md(
            r"""
### Question

YOLO returns more than one `cat` box and more than one `chair` box. What does that tell you about object detection? Does its success here imply that it should also find a fish, crab, sponge, or sea star?
"""
        ),
        md(
            r"""
### Answer

Each box represents a separately localized object instance, so several boxes may share the same class name. Success in this scene does not imply underwater competence. Pretraining supplies useful **features**—reusable visual patterns such as edges, textures, and shapes—but the COCO class list and image distribution are mostly terrestrial. An image **distribution** describes which colours, viewpoints, backgrounds, object sizes, and categories commonly occur. Underwater colour, illumination, texture, object scale, and target categories are different. This change between development data and intended-use data is domain shift.
"""
        ),
        md(
            r"""
### Run the same untouched model underwater

We lower the threshold to `0.20` so weak and wrong guesses remain visible. A failure can be a **miss** (a labelled object receives no suitable box), a **false positive** (a box is predicted where the reference labels do not support one), or a wrong class name. The class names below still come from COCO; we have not fine-tuned anything yet. `max_det=10` caps the displayed predictions at ten per image so a low threshold cannot make the figure unreadable.
"""
        ),
        code(
            r"""
UNDERWATER_CONF = 0.20
# Passing a list of paths returns a matching list of Results objects.
vanilla_underwater_results = vanilla_model.predict(
    SCENE_IMAGES,
    imgsz=640,
    conf=UNDERWATER_CONF,
    max_det=10,
    device=DEVICE,
    verbose=False,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, result, scene_id in zip(axes, vanilla_underwater_results, SCENE_IDS):
    # Ultralytics plots in BGR channel order; Matplotlib displays RGB.
    ax.imshow(result.plot()[..., ::-1])
    ax.axis("off")
    names = [result.names[int(class_id)] for class_id in result.boxes.cls]
    ax.set_title(f"{scene_id[:8]} — vanilla labels: {names or ['none']}")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            r"""
### Expected observation

On the tested stack, vanilla YOLO returns no boxes for the first scene and calls the crab/isopod objects **birds** in the second. Exact scores can vary slightly by hardware, but the qualitative domain mismatch should be the same.

Next, we will fine-tune this same model using underwater annotations and compare its predictions before and after training.
"""
        ),
        md(
            r"""
## 4. What supervision does object detection need?

During training, the labels provide the model's **supervision**. For object detection, each label row gives the class and bounding box of one object:

Each non-empty YOLO detection label row is:

`class_id x_center y_center width height`

- `class_id` is an integer identifying the target class;
- `x_center` and `y_center` locate the box centre; and
- `width` and `height` specify the box size.

The four geometry values are **normalised**: horizontal measurements are divided by image width and vertical measurements by image height, so every value lies in `[0, 1]` regardless of pixel resolution.

We merge every available biological category into one class, **underwater organism** (class `0`). The model can learn where organisms are, but not their biological identities. This is a **one-class** or **binary object-detection** task. The narrower class name matters: equipment, sediment, debris, and open water can then be valid background rather than contradictory examples of an “underwater object.”

**Important:** the detection labels use every available bounding-box annotation in this tutorial dataset; no object-size cutoff is applied. The first scene therefore shows all seven available boxes: one fish, five sponges, and one sea-star relative. Some are small and easy to overlook.

“All available annotations” does not mean that every visible organism was annotated. The labels record what annotators chose to mark, so an apparently unlabelled organism may still be present. The metrics below measure agreement with these available annotations, not whether the scene has been exhaustively catalogued.
"""
        ),
        code(
            r"""
from scripts.tutorial_viz import draw_yolo_boxes

# Each text file contains one normalized YOLO row for every labelled object instance.
for label_path in SCENE_LABELS:
    print(label_path.name)
    print(label_path.read_text().strip() or "<empty label>")
    print()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, image_path, label_path in zip(axes, SCENE_IMAGES, SCENE_LABELS):
    # The main lesson deliberately maps every organism to the single class ID 0.
    draw_yolo_boxes(image_path, label_path, class_names={0: "underwater organism"}, ax=ax)
    ax.set_title(f"all available COCO labels — {label_path.stem[:8]}")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            r"""
### Question

A YOLO row contains `x_center=0.75` and `width=0.20`. What percentage of the image width does the box span, and where are its normalised left and right edges?
"""
        ),
        md(
            r"""
### Answer

The box spans **20% of the image width** because `width=0.20`. That value describes the box's size; `x_center=0.75` describes its horizontal position.

Half the box width lies on each side of the centre, so the normalised edges are:

- left: `0.75 - 0.20/2 = 0.65`
- right: `0.75 + 0.20/2 = 0.85`

For a 1,000-pixel-wide image, the box would run from pixel `650` to pixel `850` and would be `200` pixels wide. This illustrates why normalised annotations work across different image resolutions: multiply each normalised value by the relevant image dimension to recover pixels.
"""
        ),
        md(
            r"""
## 5. Build a deliberately small fine-tuning set

Supervised machine learning normally separates data by role:

- the **training set** supplies examples used to update model weights;
- the **validation set** is kept out of weight updates and is used to compare settings and estimate **generalisation**, meaning performance on new data; and
- a **test set**, when available, is reserved for a final evaluation after all modelling choices are fixed.

The tutorial dataset contains **83 training images** and **21 validation images**, but no separate test split. It includes four manually reviewed **negative frames**, whose empty YOLO label files explicitly mean "no clearly visible target organism." Three negatives expose the model to open water, sediment, suspended particles, blur, and ROV equipment during training; the fourth checks for false alarms on held-out open water.

The helper also writes a YAML dataset configuration containing the training and validation paths and class names. `FINE_TUNE_YAML` is the path to that file.

For the live demonstration we select **32 positive images** and all **three negative frames** from the training split. The selection is **deterministic**, meaning the same inputs produce the same selected subset each time. All 21 validation images remain held out from weight updates.

Negative labels need the same care as positive boxes. If a target organism is overlooked in a supposedly negative frame, training will incorrectly treat it as background. These four frames were inspected at full resolution, and their sources and review notes are recorded in `data/background_negatives/manifest.csv`. Even reviewed labels should be revisited by a domain expert before scientific use.

This tiny subset is enough for a live demonstration, but not for a scientific evaluation. A scientific study would use more varied data and a separate test set whose image origins have been checked to prevent overlap with training data.
"""
        ),
        code(
            r"""
from scripts.tutorial_data import (
    make_detection_finetune_dataset,
    validate_yolo_dataset,
)

FINE_TUNE_ROOT = REPO_ROOT / "tmp" / "detection_finetune_subset"
# Select 32 positive images and all three reviewed training negatives.
FINE_TUNE_YAML = make_detection_finetune_dataset(
    DETECT_ROOT,
    FINE_TUNE_ROOT,
    positive_train_images=32,
    negative_train_images=3,
    # None keeps the complete held-out validation split rather than subsampling it.
    val_images=None,
)
subset_report = validate_yolo_dataset(FINE_TUNE_YAML, task="detect")
print(json.dumps(subset_report, indent=2))
"""
        ),
        code(
            r"""
# Empty labels identify the reviewed training frames for which the correct target is no box.
negative_train_labels = [
    label_path
    for label_path in sorted((FINE_TUNE_ROOT / "labels" / "train").glob("*.txt"))
    if not label_path.read_text().strip()
]
negative_train_images = [
    FINE_TUNE_ROOT / "images" / "train" / f"{label_path.stem}.jpg"
    for label_path in negative_train_labels
]
assert len(negative_train_images) == 3
show_image_grid(
    negative_train_images,
    titles=["reviewed negative: no target organism"] * len(negative_train_images),
    columns=3,
    figsize=(15, 4),
)
"""
        ),
        md(
            r"""
## 6. Fine-tune the vanilla detector

The important line is `YOLO("yolo11n.pt")`: this starts from the ordinary COCO-pretrained weights used in the warm-up, not from a FathomNet checkpoint. This is **transfer learning**: reuse knowledge learned on one dataset, then adapt it to a related task.

Training repeatedly compares predictions with annotations using a numerical **loss function**, which assigns a larger penalty to worse predictions. A **gradient** describes how a small weight change would change the loss. An **optimizer** uses those gradients to update the weights in a direction that should reduce the loss. The main settings are:

- `epochs=35`: an **epoch** is one complete pass through the training set;
- `batch=8`: a **batch** is the group of images used for one weight-update step, so 35 images create five batches per epoch, with a smaller final batch;
- `imgsz=416`: the working image size, chosen as a speed/detail compromise;
- `lr0=0.001`: the initial **learning rate**, which controls the size of weight updates; and
- `optimizer="AdamW"`: the specific update rule used to apply those changes.

Each epoch contains only five batches. Per-epoch validation is disabled to keep the demonstration short; one validation pass runs at the end. The run saves `last.pt` for the final epoch and `best.pt` for the checkpoint Ultralytics selects for evaluation.

Exact timing depends on the hardware. CPU-only execution skips training and uses the saved results below unless you enabled the CPU opt-in during setup. The tested full CPU recipe completed in about 1.5 minutes on one recent MacBook, but that timing should not be assumed for other machines.

Ultralytics applies **non-maximum suppression (NMS)** after raw prediction to remove redundant, highly overlapping boxes. Some MPS runs print an `NMS time limit exceeded` warning during validation. If the cell still completes and prints final metrics, this is a speed warning in that after-prediction step rather than a failed training run.
"""
        ),
        md(
            r"""
### Detection metrics used below

A predicted box must first be matched to a ground-truth box. Box overlap is measured with **intersection over union (IoU)**:

$$\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.$$

The **intersection** is the area covered by both regions. The **union** is the total area covered by either region, counting their shared area only once. IoU therefore measures agreement relative to everything that the prediction and ground truth claim as part of the object. An IoU of `0` means no overlap; `1` means identical regions.

The figure below uses rectangles because this section evaluates boxes. Mask IoU uses the same formula but counts foreground pixels instead of rectangular area. Evaluations normally compare box with box or mask with mask; comparing a predicted mask directly with a ground-truth box would mix two different targets.
"""
        ),
        code(
            r"""
from matplotlib.patches import Patch, Rectangle


def rectangle_iou(box_a, box_b):
    '''Return intersection area, union area, and IoU for two (x, y, width, height) boxes.'''
    ax0, ay0, aw, ah = box_a
    bx0, by0, bw, bh = box_b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    # Clamping at zero makes disjoint boxes have zero intersection rather than a negative size.
    intersection_width = max(0, ix1 - ix0)
    intersection_height = max(0, iy1 - iy0)
    intersection = intersection_width * intersection_height
    # The overlap belongs to both box areas, so subtract it once from their sum.
    union = aw * ah + bw * bh - intersection
    return intersection, union, intersection / union, (ix0, iy0, intersection_width, intersection_height)


truth_box = (3, 3, 4, 4)  # area = 16 square units
# These cases show why containment alone does not guarantee a high IoU.
iou_examples = [
    ("perfect match", (3, 3, 4, 4)),
    ("prediction contains truth", (1, 1, 8, 8)),
    ("prediction inside truth", (4, 4, 2, 2)),
    ("same size, shifted", (5, 3, 4, 4)),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharex=True, sharey=True)
for ax, (title, prediction_box) in zip(axes, iou_examples):
    intersection, union, iou, intersection_box = rectangle_iou(truth_box, prediction_box)
    tx, ty, tw, th = truth_box
    px, py, pw, ph = prediction_box
    ix, iy, iw, ih = intersection_box

    ax.add_patch(Rectangle((tx, ty), tw, th, facecolor="#2a9d8f", alpha=0.18,
                           edgecolor="#16766c", linewidth=3, linestyle="--"))
    ax.add_patch(Rectangle((px, py), pw, ph, facecolor="#f4a261", alpha=0.18,
                           edgecolor="#d66b1f", linewidth=3))
    if iw > 0 and ih > 0:
        ax.add_patch(Rectangle((ix, iy), iw, ih, facecolor="#7b2cbf", alpha=0.28,
                               edgecolor="none"))

    ax.set_title(title, fontsize=12)
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(0.5, 9.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5, -0.09,
        f"intersection = {intersection:g}   union = {union:g}\n"
        f"IoU = {intersection:g}/{union:g} = {iou:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

fig.legend(
    handles=[
        Patch(facecolor="#2a9d8f", alpha=0.25, edgecolor="#16766c", label="ground truth"),
        Patch(facecolor="#f4a261", alpha=0.25, edgecolor="#d66b1f", label="prediction"),
        Patch(facecolor="#7b2cbf", alpha=0.35, label="intersection"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.91),
    ncol=3,
    frameon=False,
)
fig.suptitle("Intersection over union: overlap relative to total claimed area", y=0.99, fontsize=15)
plt.tight_layout(rect=(0, 0.03, 1, 0.82))
plt.show()
"""
        ),
        md(
            r"""
The two containment examples are deliberately symmetric. When one region completely contains the other, the intersection is the smaller area and the union is the larger area:

$$\mathrm{IoU}=\frac{\text{smaller area}}{\text{larger area}}.$$

The oversized prediction covers four times the ground-truth area, so its IoU is `16/64 = 0.25`. The undersized prediction covers only one quarter of the ground-truth area, so its IoU is `4/16 = 0.25`. IoU penalizes both excessive background and incomplete object coverage. At an IoU matching threshold of `0.50`, neither prediction would count as a correct match even though one contains the other completely.

For masks, replace “area” with the number of foreground pixels. A predicted mask that contains the entire ground-truth mask but covers four times as many pixels also has a mask IoU of `0.25`.

At a chosen confidence and IoU threshold:

- a **true positive (TP)** is a prediction matched to a ground-truth object;
- a **false positive (FP)** is an unmatched prediction; and
- a **false negative (FN)** is a ground-truth object the model missed.

$$\mathrm{precision}=\frac{TP}{TP+FP}, \qquad
\mathrm{recall}=\frac{TP}{TP+FN}.$$

**Precision** asks, “Of the boxes predicted, how many were correct?” **Recall** asks, “Of the labelled objects, how many were found?”

**Average precision (AP)** summarises the precision–recall trade-off as the confidence threshold changes. **Mean average precision (mAP)** averages AP across classes. Because this notebook has one class, that class is the only contributor to the mean. `mAP50` uses IoU `0.50`; `mAP50-95` averages results from IoU `0.50` through `0.95` and therefore demands more precise box placement.
"""
        ),
        code(
            r"""
FINE_TUNE_EPOCHS = 35
FINE_TUNE_IMGSZ = 416
FINE_TUNE_BATCH = 8
FINE_TUNE_SUMMARY = None
FINE_TUNE_HISTORY_CSV = None
fine_tuned_model = None

if RUN_LIVE_FINE_TUNING:
    # Start from the same general-image checkpoint used in the warm-up, not an underwater model.
    fine_tuned_model = YOLO("yolo11n.pt")
    train_result = fine_tuned_model.train(
        data=str(FINE_TUNE_YAML),
        epochs=FINE_TUNE_EPOCHS,
        imgsz=FINE_TUNE_IMGSZ,
        batch=FINE_TUNE_BATCH,
        lr0=0.001,
        optimizer="AdamW",
        workers=0,
        device=DEVICE,
        project=str(REPO_ROOT / "tmp" / "detection_training"),
        name="vanilla_yolo11n_underwater_35",
        exist_ok=True,
        save=True,
        # Validate once after training to keep this short live run focused and predictable.
        val=False,
        cache="disk",
        plots=False,
        verbose=False,
        seed=42,
    )
    save_dir = Path(fine_tuned_model.trainer.save_dir)
    FINE_TUNE_HISTORY_CSV = save_dir / "results.csv"
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    # With per-epoch validation disabled, use best.pt when available and otherwise the final epoch.
    checkpoint_path = best_path if best_path.exists() else last_path
    fine_tuned_model = YOLO(checkpoint_path)
    validation = fine_tuned_model.val(
        data=str(FINE_TUNE_YAML),
        imgsz=FINE_TUNE_IMGSZ,
        batch=FINE_TUNE_BATCH,
        workers=0,
        device=DEVICE,
        plots=False,
        verbose=False,
    )
    FINE_TUNE_SUMMARY = {
        "mode": "live",
        "checkpoint": str(checkpoint_path),
        "precision": round(float(validation.box.mp), 3),
        "recall": round(float(validation.box.mr), 3),
        "mAP50": round(float(validation.box.map50), 3),
        "mAP50-95": round(float(validation.box.map), 3),
    }
else:
    # The saved history keeps the lesson inspectable when CPU training is not selected.
    FINE_TUNE_HISTORY_CSV = (
        REPO_ROOT / "data" / "reference_training" / "underwater_detection_results.csv"
    )
    FINE_TUNE_SUMMARY = {
        "mode": "saved results from a previous run (exact notebook recipe)",
        "precision": 0.325,
        "recall": 0.200,
        "mAP50": 0.188,
        "mAP50-95": 0.120,
    }

print(json.dumps(FINE_TUNE_SUMMARY, indent=2))
"""
        ),
        md(
            r"""
### What changed during training?

The left plot follows three parts of the training loss across the 35 epochs:

- **box loss** measures errors in the predicted box position and size;
- **classification loss** measures errors in the predicted object confidence and class; and
- **distribution focal loss (DFL)** helps YOLO place box boundaries more precisely.

The three losses have different meanings, so compare each curve with its own earlier values rather than comparing their absolute heights. A downward trend shows that the optimizer is fitting the training annotations. It does **not** prove that the detector generalizes to new images, which is why the right plot reports the final held-out validation scores.

With only five batches per epoch and random image augmentation, these curves will be noisy. Look for the overall direction rather than expecting every epoch to improve.

When live fine-tuning runs, the plot reads the CSV produced by that run. When training is skipped, it uses the saved 35-epoch history from a previous run of the same recipe.
"""
        ),
        code(
            r"""
from scripts.tutorial_viz import plot_detection_training_summary

history_mode = "live training history" if RUN_LIVE_FINE_TUNING else "saved history from the same 35-epoch recipe"
print(f"Plot source: {history_mode}")
# The same plotting function accepts either the new results.csv or the bundled reference file.
_ = plot_detection_training_summary(
    FINE_TUNE_HISTORY_CSV,
    FINE_TUNE_SUMMARY,
    title="Vanilla YOLO11n fine-tuned on 32 positive + 3 negative images",
)
"""
        ),
        md(
            r"""
### If training does not fit the available memory

An “out of memory” error means the accelerator cannot hold the current model, images, and batch simultaneously. First reduce `FINE_TUNE_BATCH` from `8` to `4` or `2`. If necessary, reduce `FINE_TUNE_IMGSZ` from `416` to `320`; smaller images use less memory but can make small organisms harder to detect. Do not move validation images into training to improve the metric—that would invalidate the held-out evaluation.

Exact scores can differ slightly across CUDA and MPS because some numerical operations are implemented differently. The expected qualitative behaviour is more stable: the terrestrial class names disappear, boxes begin to overlap underwater organisms, and errors remain because the training set is tiny.
"""
        ),
        md(
            r"""
### Before, after, and held-out COCO labels

Start with the images rather than the summary metric. Do the new boxes land on real objects in the held-out scenes? The comparison uses a low `0.20` display threshold because this tiny fine-tune has deliberately limited data and training.

Green boxes show every available COCO annotation for these images. They are the evaluation references described in Section 4, but they may still omit visible organisms that were never annotated. Prediction labels change from terrestrial COCO categories such as `bear` or `giraffe` to the one fine-tuning class, `underwater organism`.
"""
        ),
        code(
            r"""
from matplotlib import patches
from PIL import Image

# Saved, tested reference detections are used only on CPU-only runtimes.
# They were generated one image at a time so the confidence-threshold exercise
# below can filter the same predictions consistently.
# Coordinates use [x_min, y_min, x_max, y_max] pixels in the original image.
REFERENCE_FINE_TUNE_DETECTIONS = {
    SCENE_IDS[0]: {
        "boxes": [
            [481.6, 525.4, 627.8, 696.6],
            [0.0, 320.9, 477.9, 1048.9],
            [1476.2, 450.6, 1787.3, 862.9],
            [0.0, 897.1, 196.5, 1061.9],
            [617.4, 20.6, 1492.8, 1080.0],
            [573.7, 618.0, 1020.6, 1080.0],
            [1457.0, 247.4, 1891.9, 862.8],
            [701.7, 621.6, 959.5, 1080.0],
            [6.3, 453.5, 492.3, 850.4],
            [1758.5, 123.1, 1919.1, 275.9],
        ],
        "scores": [0.593, 0.535, 0.348, 0.239, 0.202, 0.195, 0.101, 0.095, 0.090, 0.086],
    },
    SCENE_IDS[1]: {
        "boxes": [
            [295.4, 118.6, 442.4, 201.0],
            [466.2, 136.9, 525.5, 183.5],
            [585.1, 141.7, 601.5, 156.9],
            [291.1, 119.2, 429.6, 168.5],
            [569.4, 72.1, 586.1, 85.1],
            [378.6, 159.5, 416.0, 194.7],
            [381.0, 174.6, 412.7, 193.5],
            [297.2, 58.1, 443.5, 202.4],
            [298.0, 166.1, 319.6, 186.5],
            [101.2, 197.7, 122.0, 219.9],
        ],
        "scores": [0.273, 0.160, 0.053, 0.039, 0.036, 0.030, 0.027, 0.024, 0.021, 0.021],
    },
}

if fine_tuned_model is not None:
    fine_tuned_results = fine_tuned_model.predict(
        SCENE_IMAGES,
        imgsz=FINE_TUNE_IMGSZ,
        conf=UNDERWATER_CONF,
        max_det=10,
        device=DEVICE,
        verbose=False,
    )
else:
    fine_tuned_results = None

def show_reference_prediction(
    ax,
    image_path,
    reference,
    *,
    class_names=None,
    color="deepskyblue",
):
    ax.imshow(Image.open(image_path).convert("RGB"))
    class_ids = reference.get("class_ids", [0] * len(reference["boxes"]))
    for box, score, class_id in zip(reference["boxes"], reference["scores"], class_ids):
        x0, y0, x1, y1 = box
        ax.add_patch(patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor=color, linewidth=2,
        ))
        if class_names is None:
            label = "underwater organism"
        elif isinstance(class_names, dict):
            label = class_names.get(int(class_id), str(class_id))
        else:
            label = class_names[int(class_id)]
        ax.text(x0, y0, f"{label} {score:.2f}", color="black", backgroundcolor=color)
    ax.axis("off")

def filter_reference_prediction(reference, threshold):
    # Apply the same confidence rule to saved predictions that predict(conf=...) applies live.
    class_ids = reference.get("class_ids", [0] * len(reference["boxes"]))
    retained = [
        (box, score, class_id)
        for box, score, class_id in zip(reference["boxes"], reference["scores"], class_ids)
        if score >= threshold
    ]
    return {
        "boxes": [box for box, _, _ in retained],
        "scores": [score for _, score, _ in retained],
        "class_ids": [class_id for _, _, class_id in retained],
    }

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
# Each row aligns the before, after, and COCO-label views for one held-out scene.
for row, (scene_id, image_path, label_path, vanilla_result) in enumerate(zip(
    SCENE_IDS, SCENE_IMAGES, SCENE_LABELS, vanilla_underwater_results
)):
    axes[row, 0].imshow(vanilla_result.plot()[..., ::-1])
    axes[row, 0].axis("off")
    axes[row, 0].set_title("before: vanilla COCO model")

    if fine_tuned_results is not None:
        axes[row, 1].imshow(fine_tuned_results[row].plot()[..., ::-1])
        axes[row, 1].axis("off")
    else:
        reference_at_display_threshold = filter_reference_prediction(
            REFERENCE_FINE_TUNE_DETECTIONS[scene_id], UNDERWATER_CONF
        )
        show_reference_prediction(axes[row, 1], image_path, reference_at_display_threshold)
    axes[row, 1].set_title("after: 32 positive + 3 negative images")

    draw_yolo_boxes(image_path, label_path, class_names={0: "COCO target"}, ax=axes[row, 2], color="lime")
    axes[row, 2].set_title("held-out COCO labels")

plt.tight_layout()
plt.show()
"""
        ),
        md(
            r"""
### How to read the result

The saved results come from an earlier run of this exact recipe, including the reviewed background frames. That run reached about `0.19` mAP@0.5 against the available held-out labels. In the crab scene, vanilla YOLO calls both animals `bird`; after fine-tuning, the crab receives the generic `underwater organism` label, but the smaller animal is missed at the displayed threshold. The model has adapted to the new domain without becoming a strong detector.

This is still a weak detector: it misses organisms, produces low-confidence predictions, and retains some false positives. Thirty-five training images are enough to demonstrate adaptation, but not enough to support scientific use.
"""
        ),
        md(
            r"""
### Question

If the number of detections increases, has the detector necessarily improved?
"""
        ),
        md(
            r"""
### Answer

No. Extra boxes may add true positives, false positives, or both. Detection counts must be interpreted together with IoU matching, precision, recall, and AP. The visual comparison is also important here: it reveals *which* objects were found and where boxes overlap poorly. A prediction without a matching green box may cover an organism that the source annotations did not include rather than true background.
"""
        ),
        md(
            r"""
### Self-study exercise: confidence thresholds

The **confidence threshold** is a display and decision rule: predictions below it are removed. Before running the next cell, predict what will change as the threshold rises from `0.05` to `0.25`:

1. How many boxes will remain?
2. Which real objects or false alarms might disappear?
3. Which way will precision and recall usually move?

The final panel shows the same available COCO labels used for evaluation, so you can judge *which annotated targets* lost predictions rather than looking only at the count. It cannot tell you whether a prediction overlaps an organism omitted from the source annotations.
"""
        ),
        code(
            r"""
# Student control: edit or extend this list, then rerun the cell.
THRESHOLDS = [0.05, 0.10, 0.25]
threshold_rows = []
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, threshold in zip(axes[:3], THRESHOLDS):
    if fine_tuned_model is not None:
        result = fine_tuned_model.predict(
            SCENE_IMAGES[0], imgsz=FINE_TUNE_IMGSZ, conf=threshold,
            max_det=10, device=DEVICE, verbose=False,
        )[0]
        # Convert the device tensor into ordinary Python numbers for the table below.
        scores = result.boxes.conf.detach().cpu().tolist()
        ax.imshow(result.plot()[..., ::-1])
        source_label = "live predictions"
    else:
        reference = REFERENCE_FINE_TUNE_DETECTIONS[SCENE_IDS[0]]
        filtered_reference = filter_reference_prediction(reference, threshold)
        scores = filtered_reference["scores"]
        show_reference_prediction(ax, SCENE_IMAGES[0], filtered_reference)
        source_label = "saved reference predictions"

    score_range = f"{min(scores):.2f}–{max(scores):.2f}" if scores else "none"
    threshold_rows.append((threshold, len(scores), score_range))
    ax.axis("off")
    ax.set_title(f"confidence ≥ {threshold:.2f}\n{len(scores)} boxes shown")

draw_yolo_boxes(
    SCENE_IMAGES[0],
    SCENE_LABELS[0],
    class_names={0: "COCO target"},
    ax=axes[3],
    color="lime",
)
axes[3].set_title("held-out COCO labels")
plt.suptitle(f"Confidence-threshold comparison — {source_label}")
plt.tight_layout()
plt.show()

print(f"{'threshold':>10} | {'boxes shown':>11} | {'shown score range':>17}")
print("-" * 46)
for threshold, count, score_range in threshold_rows:
    print(f"{threshold:>10.2f} | {count:>11} | {score_range:>17}")
"""
        ),
        md(
            r"""
### Exercise answer

Increasing the threshold can only keep the same number of predictions or remove some; it cannot reveal a new low-confidence box. Precision often rises because weak false positives disappear, while recall often falls because some real objects also had weak scores. The direction is not guaranteed on one image or a tiny sample.

The plots make the trade-off concrete. A removed box helps measured precision if it was unmatched, but hurts measured recall if it covered an annotated COCO target. Detection count alone cannot tell us which happened. Dataset-level precision and recall require matching predictions to the chosen reference annotations with an IoU rule, as described above.
"""
        ),
        md(
            r"""
## 7. Download and use a FathomNet-trained model

Next, instead of fine-tuning YOLO ourselves, we will download [Megalodon 2024 YOLO11](https://huggingface.co/FathomNet/megalodon-2024-yolov11), a detector already trained on public FathomNet bounding-box annotations.

The checkpoint is hosted on the **Hugging Face Hub**. We download it with `hf_hub_download(...)` and pin a specific revision so that every run uses the same weights.

The filename `mbari-megalodon-yolo11x.pt` begins with **MBARI**, the **Monterey Bay Aquarium Research Institute**, and contains `x`, meaning the extra-large YOLO11 size. Megalodon has one generic class, `item`, so it can locate prominent underwater objects but cannot identify them biologically.
"""
        ),
        code(
            r"""
from huggingface_hub import hf_hub_download

MEGALODON_REPO = "FathomNet/megalodon-2024-yolov11"
MEGALODON_FILE = "mbari-megalodon-yolo11x.pt"
# Pin an exact Hub commit so a future model update cannot silently change the lesson.
MEGALODON_REVISION = "403704d2d4dba9903ffbdd215648c3fc8ae6f377"

megalodon_path = hf_hub_download(
    repo_id=MEGALODON_REPO,
    filename=MEGALODON_FILE,
    revision=MEGALODON_REVISION,
)
megalodon_model = YOLO(megalodon_path)
print(f"Megalodon checkpoint: {megalodon_path}")
"""
        ),
        md(
            r"""
### Inference playground

Change one variable at a time, predict what will happen, and rerun the cell:

- `PLAY_IMAGE_INDEX`: choose underwater scene `0` or `1`;
- `PLAY_CONFIDENCE`: try `0.05`, `0.25`, `0.50`, or `0.70`; and
- `PLAY_IMAGE_SIZE`: try `320`, `416`, or `640` pixels.

Lowering the confidence threshold reveals weaker candidates but may add false positives. A larger inference size preserves more image detail and may help with small organisms, but it requires more computation. Use the held-out COCO-label panel to judge *which* predictions changed, remembering that even the available source annotations may not cover every visible organism. Neither detection count nor confidence alone tells you whether the result improved.

The reported wall time measures only the prediction call. The first prediction in a runtime may be slower because the software performs one-time model setup, so repeated runs are more useful for comparing settings.
"""
        ),
        code(
            r"""
from time import perf_counter

# Student controls: change these three values and rerun only this cell.
PLAY_IMAGE_INDEX = 0
PLAY_CONFIDENCE = 0.25
PLAY_IMAGE_SIZE = 416

assert PLAY_IMAGE_INDEX in range(len(SCENE_IMAGES)), "Choose image index 0 or 1."
assert 0 < PLAY_CONFIDENCE <= 1, "Confidence must be in (0, 1]."
assert PLAY_IMAGE_SIZE > 0, "Image size must be positive."

play_image = SCENE_IMAGES[PLAY_IMAGE_INDEX]
play_label = SCENE_LABELS[PLAY_IMAGE_INDEX]
play_scene_id = SCENE_IDS[PLAY_IMAGE_INDEX]

# Time only inference; model download and initialization already happened above.
prediction_start = perf_counter()
megalodon_result = megalodon_model.predict(
    play_image,
    imgsz=PLAY_IMAGE_SIZE,
    conf=PLAY_CONFIDENCE,
    max_det=20,
    device=DEVICE,
    verbose=False,
)[0]
prediction_seconds = perf_counter() - prediction_start
scores = [round(float(score), 3) for score in megalodon_result.boxes.conf]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(megalodon_result.plot()[..., ::-1])
axes[0].axis("off")
axes[0].set_title(
    f"Megalodon prediction — {len(scores)} boxes\n"
    f"confidence ≥ {PLAY_CONFIDENCE:.2f}, image size {PLAY_IMAGE_SIZE}"
)
draw_yolo_boxes(
    play_image,
    play_label,
    class_names={0: "COCO target"},
    ax=axes[1],
    color="lime",
)
axes[1].set_title(f"held-out COCO labels — {play_scene_id[:8]}")
plt.tight_layout()
plt.show()

print(json.dumps({
    "image_index": PLAY_IMAGE_INDEX,
    "confidence_threshold": PLAY_CONFIDENCE,
    "inference_image_size": PLAY_IMAGE_SIZE,
    "boxes_shown": len(scores),
    "confidence_scores": scores,
    "prediction_wall_seconds": round(prediction_seconds, 3),
}, indent=2))
"""
        ),
        md(
            r"""
### Question

Is “Megalodon finds more underwater objects than vanilla YOLO11n” a fair scientific benchmark of training data alone?
"""
        ),
        md(
            r"""
### Answer and caveats

No. At least three factors are mixed together:

- **domain data:** Megalodon was trained on public FathomNet bounding-box annotations;
- **model capacity:** it is YOLO11x, with many more learned parameters than the YOLO11n model used for the live fine-tune; and
- **possible data relationship:** these tutorial images are FathomNet-derived, so they are not guaranteed to be independent of Megalodon's training collection.

Use this comparison to see how the models behave, not to claim that one training dataset is better. A fair benchmark would keep model size, training resources, annotation rules, and test data consistent.
"""
        ),
        md(
            r"""
## 8. Finale: text-prompted segmentation with SAM3

Unlike the YOLO detectors above, **Segment Anything Model 3 (SAM3)** takes a prompt at inference time. A **text prompt** such as `"fish"` or `"sponge"` tells the model what to segment.

For each matching object instance, SAM3 can return a confidence score, a box, and a **pixel mask**: an array indicating which image pixels belong to that object. Change the prompt and you can ask for a different kind of object without retraining the model.

Live SAM3 has heavier requirements than the YOLO sections: Python 3.12 or newer, a recent CUDA-enabled build of PyTorch, the official `sam3` Python package, permission to access the model checkpoint, and a Hugging Face **access token**. A token is a private credential used to authenticate the download. Never paste a real token into a saved code or markdown cell; the helper below uses hidden input and stores it only in the current runtime. Live SAM3 is unavailable on CPU and MPS runtimes.

To keep the notebook executable everywhere:

- a prepared CUDA runtime with SAM3 installed and the `HF_TOKEN` environment variable configured runs real inference automatically; and
- other runtimes show a **label-derived teaching reference**, constructed from existing annotations and clearly marked as *not SAM3 output*.

On unsupported hardware, the next cells load a label-derived example so the plotting code still works. It is not SAM3 output and should not be interpreted as model performance. The reference is **cached**, meaning it is already saved in the data bundle. To prepare a new compatible runtime, change `INSTALL_AND_AUTHENTICATE_SAM3` to `True`, run the next cell, enter a Hugging Face read token in the hidden prompt if requested, and rerun the cell once installation is complete. In the printed status report, `sam3_importable` means that Python can locate and load the installed package.
"""
        ),
        code(
            r"""
from scripts.tutorial_sam3 import (
    configure_huggingface_token,
    install_sam3_package,
    load_cached_sam3_result,
    run_sam3_text_prompt,
    sam3_can_run_live,
)
from scripts.tutorial_viz import plot_sam3_result

# For a fresh CUDA/Colab runtime, deliberately opt in once, run this cell,
# then rerun it after installation/authentication completes.
INSTALL_AND_AUTHENTICATE_SAM3 = False

if INSTALL_AND_AUTHENTICATE_SAM3:
    status = sam3_can_run_live()
    if not status["sam3_importable"]:
        install_sam3_package()
    if not status["hf_token_present"]:
        configure_huggingface_token(prompt_if_missing=True)

SAM3_STATUS = sam3_can_run_live()
# This combines package, hardware, checkpoint-access, and authentication checks.
USE_LIVE_SAM3 = bool(SAM3_STATUS["can_run"])
print(json.dumps(SAM3_STATUS, indent=2))
print(f"Live SAM3 inference: {USE_LIVE_SAM3}")
"""
        ),
        code(
            r"""
SAM3_IMAGE_ID = SCENE_IDS[0]
SAM3_PROMPTS = ["fish", "sponge"]

for prompt in SAM3_PROMPTS:
    if USE_LIVE_SAM3:
        sam3_result = run_sam3_text_prompt(
            SCENE_IMAGES[0], prompt, confidence_threshold=0.35
        )
        title = f"LIVE SAM3 — text prompt: {prompt!r}"
    else:
        # The cached object uses the same plotting schema but is explicitly not SAM3 output.
        sam3_result = load_cached_sam3_result(BUNDLE_ROOT, SAM3_IMAGE_ID, prompt)
        title = f"REFERENCE LABEL PREVIEW (NOT SAM3) — prompt: {prompt!r}"

    print({
        "prompt": prompt,
        "source": sam3_result.get("source"),
        "instances": len(sam3_result.get("boxes", [])),
    })
    plot_sam3_result(
        SCENE_IMAGES[0],
        sam3_result,
        score_threshold=0.35,
        title=title,
    )
"""
        ),
        md(
            r"""
### Question

If the prompt `"echinoderm"` returns no mask, does that prove no echinoderm is present?
"""
        ),
        md(
            r"""
### Answer

No. A negative prompt result can reflect the wording, confidence threshold, object scale, **occlusion** (part of the object being hidden), domain shift, or model failure. Try synonyms and more concrete descriptions, and inspect multiple thresholds. A prompt result is a model prediction, not proof. Compare it with expert annotations before drawing conclusions.
"""
        ),
        md(
            r"""
## Takeaways

- A successful cat-café prediction verifies model loading, inference, and plotting while showing that a detector can localize several object classes and instances; it does not establish underwater competence.
- Bounding-box labels provide the new supervision that changes a vanilla detector's behaviour.
- A 35-image fine-tune—32 positive frames plus three reviewed negatives—can create a visible domain-adaptation signal, but it is not enough to justify using the model in a real scientific workflow.
- Downloadable FathomNet models provide a strong underwater starting point, with provenance and comparison caveats.
- SAM3's text-prompt interface is genuinely different and powerful, but every prompt still needs evaluation against trustworthy labels.

For a real project, check for **data leakage**—the same image, video sequence, or near-duplicate appearing in both training and evaluation data. Retain meaningful biological classes rather than one generic object class, collect more difficult negative frames, compare model designs at similar sizes, and evaluate once on a provenance-controlled test set.
"""
        ),
        md(
            r"""
---

## Appendix A — Fine-tune YOLO to predict class names

The main lesson treats every annotated organism as the same class, `underwater organism`. That makes domain adaptation easy to see, but it leaves out an important part of object detection: a detector can predict both *where* an organism is and *what class* it belongs to.

Multi-class detection uses the same five-value YOLO row as one-class detection:

```text
class_id x_center y_center width height
```

Only the meaning of `class_id` changes. In the main lesson it is always `0`; here it can range from `0` to `4`.

The tutorial's COCO annotations contain 96 detailed FathomNet concept names, many represented by only one or two examples. That is too sparse for a useful short exercise, so this appendix maps selected concepts into five broad visual groups: `fish`, `crustacean`, `echinoderm`, `gelatinous`, and `sponge/cnidarian`. Sponges and cnidarians are biologically distinct; they share one visual workshop label here only because the dataset is small. These groups are not a biological taxonomy. A scientific project should define its classes with domain experts and retain the detailed concept provenance.
"""
        ),
        code(
            r"""
import importlib

from scripts import tutorial_data

# Reload the helper module so this cell also works after scripts/tutorial_data.py
# changes during an active Jupyter session.
tutorial_data = importlib.reload(tutorial_data)

COCO_JSON = get_task_paths("coco", BUNDLE_ROOT)["json"]
MULTICLASS_ROOT = REPO_ROOT / "tmp" / "appendix_multiclass_detection"
# Write disposable five-class labels under tmp/; the bundled data remains unchanged.
MULTICLASS_INFO = tutorial_data.make_coarse_multiclass_detection_dataset(
    COCO_JSON,
    DETECT_ROOT,
    MULTICLASS_ROOT,
)
MULTICLASS_YAML = Path(MULTICLASS_INFO["yaml_path"])
MULTICLASS_NAMES = MULTICLASS_INFO["class_names"]
multiclass_validation = validate_yolo_dataset(MULTICLASS_YAML, task="detect")

multiclass_summary = {
    "class_names": MULTICLASS_NAMES,
    "instances": MULTICLASS_INFO["instance_counts"],
    "images_with_class": MULTICLASS_INFO["image_counts"],
    "empty_label_files": MULTICLASS_INFO["empty_label_files"],
    "reviewed_negative_images": MULTICLASS_INFO["reviewed_negative_images"],
    "small_instances_omitted": MULTICLASS_INFO["filtered_small_instances"],
    "annotations_outside_the_five_groups": sum(
        MULTICLASS_INFO["unmapped_annotation_counts"].values()
    ),
}
print(json.dumps(multiclass_summary, indent=2))
# Catch malformed or out-of-range YOLO rows before optional training begins.
assert multiclass_validation["valid"], multiclass_validation["invalid_lines"]
"""
        ),
        md(
            r"""
The split remains the same as in the bundle: 83 training images and 21 validation images, including the four reviewed negatives. Images without one of the five retained classes stay in the dataset with an empty label file. The reviewed negatives are legitimate background for all five organism groups. Other empty files may instead mean that the image contains a biological category outside these five groups, so the model would be penalized if it predicted that organism as one of the retained classes. This is why an empty label must always be interpreted relative to a precisely defined target set.

The printed counts reveal a lesson that matters more than the score itself: an evaluation metric is an estimate, and an estimate needs enough observations to be trustworthy. The validation split contains only one retained fish instance and one retained crustacean instance. Missing either object can push that class's AP to zero; detecting it can make performance look much stronger than the evidence warrants.

This is a **low-number statistics** problem. When a class is represented by only one or two validation objects, the result depends heavily on which examples happened to enter the split. The dataset is still useful for learning the mechanics of multi-class training, but it is too small to support reliable claims about class-specific performance.
"""
        ),
        code(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Reusing the main held-out scenes makes the new class names directly comparable.
for ax, scene_id in zip(axes, SCENE_IDS):
    image_path = MULTICLASS_ROOT / "images" / "val" / f"{scene_id}.jpg"
    label_path = MULTICLASS_ROOT / "labels" / "val" / f"{scene_id}.txt"
    draw_yolo_boxes(
        image_path,
        label_path,
        class_names=MULTICLASS_NAMES,
        ax=ax,
        color="lime",
    )
    ax.set_title(f"five-class ground truth — {scene_id[:8]}")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            r"""
### Question

What happens during evaluation if a predicted box is in the right place but has the wrong class name?
"""
        ),
        md(
            r"""
### Answer

Its location may be good, but it cannot match the ground-truth object as the correct class. It usually counts as a false positive for the predicted class and a false negative for the true class. Multi-class detection therefore has two ways to fail: poor localization and incorrect classification.
"""
        ),
        md(
            r"""
### Optional multi-class fine-tuning

We again start from `yolo11n.pt`, the vanilla COCO detector. Ultralytics reads the five class names from `MULTICLASS_YAML` and adapts the detection head from its original 80 COCO outputs to five workshop classes. The rest of the network still begins with useful pretrained visual features.

Set `RUN_MULTICLASS_FINE_TUNING = True` to run the 30-epoch exercise. It is off by default so the appendix does not add several minutes to “Run All.” If training is skipped, the next two cells show saved metrics, loss curves, and predictions from a previous run of this exact recipe. Their titles identify them as saved results rather than live output.
"""
        ),
        code(
            r"""
from scripts.tutorial_viz import plot_detection_training_summary

# Keep appendix training opt-in so every reader can inspect the data and saved metrics quickly.
RUN_MULTICLASS_FINE_TUNING = False
MULTICLASS_EPOCHS = 30
MULTICLASS_IMGSZ = 416
MULTICLASS_BATCH = 8
multiclass_model = None

MULTICLASS_REFERENCE_SUMMARY = {
    "mode": "saved results from a previous 30-epoch appendix run",
    "precision": 0.698,
    "recall": 0.126,
    "mAP50": 0.126,
    "mAP50-95": 0.088,
    "mAP50_by_class": {
        "fish": 0.000,
        "crustacean": 0.025,
        "echinoderm": 0.004,
        "gelatinous": 0.370,
        "sponge/cnidarian": 0.231,
    },
}

if RUN_MULTICLASS_FINE_TUNING:
    multiclass_model = YOLO("yolo11n.pt")
    multiclass_model.train(
        data=str(MULTICLASS_YAML),
        epochs=MULTICLASS_EPOCHS,
        imgsz=MULTICLASS_IMGSZ,
        batch=MULTICLASS_BATCH,
        lr0=0.001,
        optimizer="AdamW",
        workers=0,
        device=DEVICE,
        project=str(REPO_ROOT / "tmp" / "appendix_training"),
        name="multiclass_yolo11n",
        exist_ok=True,
        save=True,
        val=False,
        cache="disk",
        plots=False,
        verbose=False,
        seed=42,
    )
    multiclass_save_dir = Path(multiclass_model.trainer.save_dir)
    multiclass_results_csv = multiclass_save_dir / "results.csv"
    multiclass_best = multiclass_save_dir / "weights" / "best.pt"
    multiclass_last = multiclass_save_dir / "weights" / "last.pt"
    # Fall back to the final epoch if this training configuration did not create best.pt.
    multiclass_checkpoint = multiclass_best if multiclass_best.exists() else multiclass_last
    multiclass_model = YOLO(multiclass_checkpoint)
    multiclass_metrics = multiclass_model.val(
        data=str(MULTICLASS_YAML),
        imgsz=MULTICLASS_IMGSZ,
        batch=MULTICLASS_BATCH,
        workers=0,
        device=DEVICE,
        plots=False,
        verbose=False,
    )
    # Ultralytics reports AP only for evaluated class IDs; map them back to readable names.
    evaluated_class_ids = [int(class_id) for class_id in multiclass_metrics.box.ap_class_index]
    per_class_map50 = {
        MULTICLASS_NAMES[class_id]: round(float(ap50), 3)
        for class_id, ap50 in zip(evaluated_class_ids, multiclass_metrics.box.ap50)
    }
    multiclass_summary = {
        "mode": "live",
        "checkpoint": str(multiclass_checkpoint),
        "precision": round(float(multiclass_metrics.box.mp), 3),
        "recall": round(float(multiclass_metrics.box.mr), 3),
        "mAP50": round(float(multiclass_metrics.box.map50), 3),
        "mAP50-95": round(float(multiclass_metrics.box.map), 3),
        "mAP50_by_class": per_class_map50,
    }
else:
    multiclass_summary = MULTICLASS_REFERENCE_SUMMARY
    multiclass_results_csv = (
        REPO_ROOT / "data" / "reference_training" / "multiclass_detection_results.csv"
    )

print(json.dumps(multiclass_summary, indent=2))
multiclass_history_mode = (
    "live training history"
    if RUN_MULTICLASS_FINE_TUNING
    else "saved history from the same 30-epoch recipe"
)
print(f"Plot source: {multiclass_history_mode}")
_ = plot_detection_training_summary(
    multiclass_results_csv,
    multiclass_summary,
    title="Vanilla YOLO11n fine-tuned for five underwater class groups",
)
"""
        ),
        code(
            r"""
MULTICLASS_DISPLAY_CONF = 0.05
# These saved detections came from the reference run summarized above. Coordinates
# use [x_min, y_min, x_max, y_max] pixels in the original images.
MULTICLASS_REFERENCE_PREDICTIONS = {
    SCENE_IDS[0]: {
        "boxes": [
            [481.7, 530.3, 629.2, 690.3],
            [285.3, 645.9, 421.1, 745.8],
            [761.4, 610.5, 942.7, 729.2],
            [1005.6, 577.3, 1105.8, 706.2],
            [0.0, 814.3, 180.5, 1059.6],
            [602.5, 746.7, 772.8, 907.3],
            [160.8, 460.5, 402.1, 611.9],
            [1.4, 335.9, 106.3, 494.7],
            [969.8, 33.6, 1392.2, 510.8],
            [305.2, 645.2, 413.8, 743.2],
        ],
        "scores": [0.5499, 0.1152, 0.0800, 0.0761, 0.0735, 0.0561, 0.0540, 0.0532, 0.0360, 0.0350],
        "class_ids": [4, 1, 4, 4, 4, 4, 4, 4, 4, 2],
    },
    SCENE_IDS[1]: {
        "boxes": [
            [298.6, 116.9, 432.5, 187.6],
            [381.8, 123.4, 444.0, 183.6],
            [298.6, 115.5, 453.0, 197.2],
            [463.3, 136.5, 525.0, 185.8],
            [294.9, 118.3, 435.8, 202.3],
            [379.5, 121.3, 443.9, 194.5],
            [297.8, 117.8, 448.7, 209.0],
            [296.8, 117.8, 451.4, 207.0],
            [460.5, 132.6, 527.6, 188.0],
            [375.4, 154.6, 426.0, 204.4],
        ],
        "scores": [0.0476, 0.0447, 0.0408, 0.0378, 0.0327, 0.0308, 0.0303, 0.0298, 0.0238, 0.0104],
        "class_ids": [3, 3, 0, 2, 2, 2, 3, 4, 3, 2],
    },
}

if multiclass_model is not None:
    multiclass_predictions = multiclass_model.predict(
        SCENE_IMAGES,
        imgsz=MULTICLASS_IMGSZ,
        conf=MULTICLASS_DISPLAY_CONF,
        max_det=10,
        device=DEVICE,
        verbose=False,
    )
else:
    multiclass_predictions = None

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for index, (ax, image_path, scene_id) in enumerate(zip(axes, SCENE_IMAGES, SCENE_IDS)):
    if multiclass_predictions is not None:
        result = multiclass_predictions[index]
        ax.imshow(result.plot()[..., ::-1])
        ax.axis("off")
        ax.set_title(f"live five-class prediction — {scene_id[:8]}")
    else:
        saved_prediction = filter_reference_prediction(
            MULTICLASS_REFERENCE_PREDICTIONS[scene_id],
            MULTICLASS_DISPLAY_CONF,
        )
        show_reference_prediction(
            ax,
            image_path,
            saved_prediction,
            class_names=MULTICLASS_NAMES,
            color="cyan",
        )
        ax.set_title(
            f"saved five-class prediction — {len(saved_prediction['boxes'])} boxes "
            f"at confidence ≥ {MULTICLASS_DISPLAY_CONF:.2f}"
        )
plt.tight_layout()
plt.show()
"""
        ),
        md(
            r"""
### Interpreting the saved result: when a metric is not enough

At the displayed confidence threshold of `0.05`, the saved predictions contain several overlapping boxes in the sponge-rich scene and no boxes in the crustacean scene. This is a useful visual warning: a model can produce many class-labelled boxes while still having weak recall and poor localization.

The saved run has an overall `mAP50` of `0.126`. The gelatinous and sponge/cnidarian groups score better than the other groups, while fish has an `mAP50` of `0.000`. That zero does **not** establish that YOLO cannot learn to detect fish. It means that the model did not correctly match the single retained fish instance in this particular validation split at IoU 0.50.

The reverse is equally important: a high score based on one or two objects would not be strong evidence either. Before attributing a result to the model, inspect how many independent images and objects contributed to it, look at the predictions, and ask whether the evaluation set represents the conditions in which the model will be used. A convincing assessment would use many more held-out examples from separate dives or video sequences and would report per-class support counts alongside AP.

### Question

What does the fish `mAP50` of `0.000` tell us, and what does it not tell us?

### Answer

It tells us that the detector did not produce a correct IoU-matched detection for the fish example available in this validation split. It does not give us a stable estimate of how the detector performs on fish in general. With only one example, the class metric is effectively decided by a single success or failure. The appropriate next step is to collect more independent fish examples for evaluation, not to draw a broad conclusion from another decimal place.
"""
        ),
        md(
            r"""
---

## Appendix B — Fine-tune YOLO for instance segmentation

Instance segmentation predicts a separate pixel region, called a **mask**, for every object. Compared with a box, a mask follows an object's outline and excludes more background, but masks take longer to annotate and usually require more computation to train.

A **polygon** is a closed shape defined by ordered corner points called **vertices**. YOLO segmentation rows contain `class_id` followed by normalised polygon vertices: `x1 y1 x2 y2 ...`. Connecting and filling the vertices creates the ground-truth instance mask.

The bundled segmentation labels use one generic class so we can focus on the change in geometry. To keep optional segmentation training compact, masks whose bounding boxes cover less than `0.5%` of the image are omitted. Detection keeps all available COCO boxes, so a small detection box may have no corresponding mask in the side-by-side figure below. Multi-class instance segmentation combines both appendices: keep the class-specific ID at the start of every polygon row, then train the segmentation model on those labels.
"""
        ),
        code(
            r"""
from scripts.tutorial_viz import draw_yolo_masks

SEGMENT_PATHS = get_task_paths("segment", BUNDLE_ROOT)
SEGMENT_ROOT = SEGMENT_PATHS["root"]
SEGMENT_YAML = SEGMENT_PATHS["yaml"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Each row uses the same scene; only the annotation representation changes.
for row, scene_id in enumerate(SCENE_IDS):
    detection_image = DETECT_ROOT / "images" / "val" / f"{scene_id}.jpg"
    detection_label = DETECT_ROOT / "labels" / "val" / f"{scene_id}.txt"
    segment_image = SEGMENT_ROOT / "images" / "val" / f"{scene_id}.jpg"
    segment_label = SEGMENT_ROOT / "labels" / "val" / f"{scene_id}.txt"

    draw_yolo_boxes(
        detection_image,
        detection_label,
        class_names={0: "object box"},
        ax=axes[row, 0],
        color="lime",
    )
    axes[row, 0].set_title("bounding-box ground truth")

    draw_yolo_masks(
        segment_image,
        segment_label,
        class_names={0: "object mask"},
        ax=axes[row, 1],
    )
    axes[row, 1].set_title("instance-mask ground truth")
plt.tight_layout()
plt.show()
"""
        ),
        code(
            r"""
polygon_label = next(
    path
    for path in sorted((SEGMENT_ROOT / "labels" / "val").glob("*.txt"))
    if path.read_text(encoding="utf-8").strip()
)
# A segmentation row starts with the class ID, followed by normalized x, y vertex pairs.
polygon_values = [float(value) for value in polygon_label.read_text(encoding="utf-8").splitlines()[0].split()]
polygon_points = list(zip(polygon_values[1::2], polygon_values[2::2]))

print({
    "label_file": polygon_label.name,
    "class_id": int(polygon_values[0]),
    "vertex_count": len(polygon_points),
    "first_three_normalized_vertices": polygon_points[:3],
})
"""
        ),
        md(
            r"""
### Question

Why can two segmentation rows have different numbers of values?
"""
        ),
        md(
            r"""
### Answer

After the class ID, each pair of values represents one polygon vertex. A simple outline may need only a few vertices; a complex outline may need many. The coordinates are normalized by image width and height, just like the box coordinates used earlier.
"""
        ),
        md(
            r"""
### Optional segmentation fine-tuning

`yolo11n-seg.pt` is the segmentation counterpart to the detector used in the main lesson. It predicts boxes and adds a mask head that produces a shape for each detected instance. Training therefore includes both box-related and mask-related losses, and evaluation reports two families of metrics:

- **box mAP** matches predictions using bounding-box IoU; and
- **mask mAP** matches predictions using pixel-mask IoU.

Set `RUN_SEGMENTATION_FINE_TUNING = True` to run this 30-epoch exercise. With the switch off, the next two cells show saved metrics, curves, and mask overlays from an earlier run of this exact recipe. The saved overlays use a confidence threshold of `0.20`. Their titles identify them as saved results, and the checkpoint itself is not included.
"""
        ),
        code(
            r"""
from scripts.tutorial_viz import plot_training_curves

# The -seg checkpoint adds a mask-prediction head to the usual YOLO box detector.
RUN_SEGMENTATION_FINE_TUNING = False
SEGMENTATION_EPOCHS = 30
SEGMENTATION_IMGSZ = 416
SEGMENTATION_BATCH = 8
segmentation_model = None

if RUN_SEGMENTATION_FINE_TUNING:
    segmentation_model = YOLO("yolo11n-seg.pt")
    segmentation_model.train(
        data=str(SEGMENT_YAML),
        epochs=SEGMENTATION_EPOCHS,
        imgsz=SEGMENTATION_IMGSZ,
        batch=SEGMENTATION_BATCH,
        lr0=0.001,
        optimizer="AdamW",
        workers=0,
        device=DEVICE,
        project=str(REPO_ROOT / "tmp" / "appendix_training"),
        name="binary_yolo11n_seg",
        exist_ok=True,
        save=True,
        cache="disk",
        plots=False,
        verbose=False,
        seed=42,
    )
    segmentation_save_dir = Path(segmentation_model.trainer.save_dir)
    segmentation_best = segmentation_save_dir / "weights" / "best.pt"
    segmentation_last = segmentation_save_dir / "weights" / "last.pt"
    segmentation_checkpoint = segmentation_best if segmentation_best.exists() else segmentation_last
    live_results_csv = segmentation_save_dir / "results.csv"
    segmentation_model = YOLO(segmentation_checkpoint)
    segmentation_metrics = segmentation_model.val(
        data=str(SEGMENT_YAML),
        imgsz=SEGMENTATION_IMGSZ,
        batch=SEGMENTATION_BATCH,
        workers=0,
        device=DEVICE,
        plots=False,
        verbose=False,
    )
    segmentation_summary = {
        "mode": "live",
        "checkpoint": str(segmentation_checkpoint),
        "box_mAP50": round(float(segmentation_metrics.box.map50), 3),
        "box_mAP50-95": round(float(segmentation_metrics.box.map), 3),
        "mask_mAP50": round(float(segmentation_metrics.seg.map50), 3),
        "mask_mAP50-95": round(float(segmentation_metrics.seg.map), 3),
    }
    segmentation_results_csv = live_results_csv
else:
    # Saved curves and summary keep the appendix useful without a second training run.
    segmentation_summary = {
        "mode": "saved results from a previous 30-epoch appendix run",
        "box_precision": 0.355,
        "box_recall": 0.338,
        "box_mAP50": 0.246,
        "box_mAP50-95": 0.166,
        "mask_precision": 0.355,
        "mask_recall": 0.338,
        "mask_mAP50": 0.238,
        "mask_mAP50-95": 0.130,
    }
    segmentation_results_csv = (
        REPO_ROOT / "data" / "reference_training" / "segmentation_results.csv"
    )

print(json.dumps(segmentation_summary, indent=2))
# Ultralytics uses (B) for box metrics and (M) for mask metrics in results.csv.
_ = plot_training_curves(
    segmentation_results_csv,
    metric_columns=[
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
    ],
    title="Box and mask validation metrics by epoch",
)
"""
        ),
        code(
            r"""
SEGMENTATION_DISPLAY_CONF = 0.20
if segmentation_model is not None:
    segmentation_predictions = segmentation_model.predict(
        SCENE_IMAGES,
        imgsz=SEGMENTATION_IMGSZ,
        conf=SEGMENTATION_DISPLAY_CONF,
        max_det=10,
        device=DEVICE,
        verbose=False,
    )
else:
    segmentation_predictions = None

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for index, (ax, scene_id) in enumerate(zip(axes, SCENE_IDS)):
    if segmentation_predictions is not None:
        result = segmentation_predictions[index]
        ax.imshow(result.plot()[..., ::-1])
        ax.set_title(f"live mask prediction — {scene_id[:8]}")
    else:
        reference_image = (
            REPO_ROOT
            / "data"
            / "reference_training"
            / f"appendix_segmentation_{scene_id}.jpg"
        )
        ax.imshow(Image.open(reference_image).convert("RGB"))
        ax.set_title(f"saved mask prediction — {scene_id[:8]}")
    ax.axis("off")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            r"""
### Question

When is a mask worth the added annotation cost?

### Answer

Use masks when the analysis depends on object area, contour, contact between organisms, partial occlusion, shape, or precise separation from the background. If approximate location or counting is enough, boxes are usually cheaper and easier to label consistently.

The same distinction applies to evaluation: good box mAP does not guarantee good mask mAP. A prediction can surround the right object while tracing its boundary poorly. The saved plot demonstrates both outcomes: the crustacean scene contains two localized masks, while the first scene contains one very large mask that spills across equipment, background, and several organisms. A mask is more detailed than a box, but detail does not make an incorrect prediction trustworthy.

### Putting the extensions together

- Multi-class detection changes the class ID while retaining box geometry.
- Instance segmentation changes the geometry from a box to a polygon.
- Multi-class instance segmentation makes both changes: each polygon begins with its biological class ID and is used to train `yolo11n-seg.pt`.

Before treating any of these models as scientific instruments, define the label set with experts, collect enough examples of rare classes, split data by video sequence or deployment to prevent leakage, audit unlabelled organisms, report per-class box and mask metrics, and reserve a provenance-controlled test set for final evaluation.

The [Ultralytics segmentation documentation](https://docs.ultralytics.com/tasks/segment/) provides the full prediction and training interface.
"""
        ),
    ]

    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"

    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "fathomnet_underwater_object_detection.ipynb"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "notebooks" / "fathomnet_underwater_object_detection.ipynb"
    output_path.write_text(json.dumps(build_notebook(), indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
