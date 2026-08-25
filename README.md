# Underwater Computer Vision With FathomNet

Local- and Colab-compatible tutorial notebooks on object detection, image classification, instance segmentation, and promptable segmentation for underwater imagery.

## Notebooks

- Recommended object-detection tutorial: `notebooks/fathomnet_underwater_object_detection.ipynb`
- Student notebook: `notebooks/fathomnet_underwater_vision_tutorial.ipynb`
- Instructor/master notebook: `notebooks/fathomnet_underwater_vision_tutorial_master.ipynb`

The object-detection notebook begins with a cat-café vanilla-YOLO warm-up, focuses the main lesson on bounding boxes and fine-tuning vanilla YOLO11n on 32 positive underwater images plus three reviewed background frames, downloads a FathomNet Megalodon checkpoint, and ends with SAM3. The appendices extend that workflow to class-specific detection and conventional instance segmentation.

Open the object-detection notebook in Colab:

https://colab.research.google.com/github/Spiffical/fathomnet-underwater-vision-tutorial/blob/main/notebooks/fathomnet_underwater_object_detection.ipynb

Open the student notebook in Colab:

https://colab.research.google.com/github/Spiffical/fathomnet-underwater-vision-tutorial/blob/main/notebooks/fathomnet_underwater_vision_tutorial.ipynb

The master notebook includes filled-in exercise answers and instructor notes.

Instructor exercise answer key:

- `docs/exercise_answer_key.md`

## Data

The workshop uses a compact prebuilt bundle:

- `data/fathomnet_underwater_tutorial_bundle.zip`
- `data/fathomnet_underwater_tutorial_bundle.zip.sha256`

The notebooks extract the zip into `data/fathomnet_underwater_tutorial_bundle/` at runtime. The extracted directory is ignored by git.

The binary detection task uses the class name `underwater organism`. Four manually reviewed frames with no clearly visible target organism provide three training negatives and one held-out validation negative. Their source URLs and review notes are recorded in `data/background_negatives/manifest.csv`.

The bundle contains FathomNet-derived imagery and annotations for teaching ML workflows. See:

- FathomNet Database: https://database.fathomnet.org/fathomnet/#/
- FathomNet data use: https://www.fathomnet.org/datause
- FathomNet terms: https://www.fathomnet.org/terms

## Local Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyterlab
jupyter lab
```

The tutorial pins the core vision stack to `torch==2.11.0`, `torchvision==0.26.0`, and `ultralytics==8.4.41`. The notebook setup cell applies the same pinning in Colab and prints a dependency report before training.

The object-detection notebook runs live YOLO fine-tuning with CUDA or Apple MPS. CPU-only execution uses a clearly labelled saved reference result by default, with an explicit opt-in for live CPU fine-tuning. Live SAM3 currently requires a supported CUDA environment, the official SAM3 package, model access, and a Hugging Face token; other runtimes show a label-derived teaching reference that is explicitly not presented as SAM3 inference.

## Regenerating The Notebooks

The notebooks are generated from scripts so the student and instructor versions stay aligned:

```bash
python scripts/create_tutorial_notebook.py
python scripts/create_master_notebook.py
python scripts/create_object_detection_notebook.py
```

Utility code lives in `scripts/`. The notebook keeps only the functions that are meant to be inspected, edited, or reasoned through during the tutorial.

## Validation

Before teaching, run the selected notebook top-to-bottom in the intended environment. The object-detection notebook has been tested locally on Apple Silicon with live MPS fine-tuning; preflight live SAM3 separately on the CUDA runtime that will be used.
