# Underwater Computer Vision With FathomNet

Colab-first tutorial notebooks on image classification, object detection, instance segmentation, and promptable segmentation for underwater imagery.

## Notebooks

- Student notebook: `notebooks/fathomnet_underwater_vision_tutorial.ipynb`
- Instructor/master notebook: `notebooks/fathomnet_underwater_vision_tutorial_master.ipynb`

Open the student notebook in Colab:

https://colab.research.google.com/github/Spiffical/fathomnet-underwater-vision-tutorial/blob/main/notebooks/fathomnet_underwater_vision_tutorial.ipynb

The master notebook includes filled-in exercise answers and instructor notes.

## Data

The workshop uses a compact prebuilt bundle:

- `data/fathomnet_underwater_tutorial_bundle.zip`
- `data/fathomnet_underwater_tutorial_bundle.zip.sha256`

The notebooks extract the zip into `data/fathomnet_underwater_tutorial_bundle/` at runtime. The extracted directory is ignored by git.

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
jupyter lab
```

The tutorial pins the core vision stack to `torch==2.11.0`, `torchvision==0.26.0`, and `ultralytics==8.4.41`. The notebook setup cell applies the same pinning in Colab and prints a dependency report before training.

The notebooks also run in CPU-only fallback mode. Live YOLO training is enabled only when CUDA is available.

## Regenerating The Notebooks

The notebooks are generated from scripts so the student and instructor versions stay aligned:

```bash
python scripts/create_tutorial_notebook.py
python scripts/create_master_notebook.py
```

Utility code lives in `scripts/`. The notebook keeps only the functions that are meant to be inspected, edited, or reasoned through during the tutorial.

## Validation

Before teaching, run both notebooks top-to-bottom in Colab or locally. A quick local smoke test can use CPU/fallback mode; a full pre-workshop check should also run live GPU training for the small YOLO sections.
