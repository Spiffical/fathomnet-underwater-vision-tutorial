"""Small wrappers around Ultralytics model workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _load_yolo(model_or_path):
    """Return an Ultralytics YOLO object from a model name, path, or object."""

    if hasattr(model_or_path, "train") and hasattr(model_or_path, "predict"):
        return model_or_path
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics before running YOLO training cells.") from exc
    return YOLO(str(model_or_path))


def train_yolo_task(
    task: str,
    data_yaml_or_dir: str | Path,
    model_name: str | Path,
    train_args: dict | None = None,
):
    """Train a YOLO classification, detection, or segmentation model.

    The `task` argument is mostly pedagogical: Ultralytics infers the actual
    task from the model weights, but keeping it explicit helps you reason
    about which loss and label format they are using.
    """

    train_args = dict(train_args or {})
    train_args.setdefault("data", str(data_yaml_or_dir))
    train_args.setdefault("project", f"runs/{task}")
    train_args.setdefault("exist_ok", True)
    model = _load_yolo(model_name)
    results = model.train(**train_args)
    save_dir = None
    trainer = getattr(model, "trainer", None)
    if trainer is not None and getattr(trainer, "save_dir", None) is not None:
        save_dir = Path(trainer.save_dir)
    return {"model": model, "results": results, "save_dir": save_dir}


def evaluate_yolo_model(model_or_path, data: str | Path | None = None, **kwargs):
    """Run Ultralytics validation for a trained or pretrained model."""

    model = _load_yolo(model_or_path)
    if data is not None:
        kwargs.setdefault("data", str(data))
    return model.val(**kwargs)


def predict_examples(
    model_or_path,
    sources: str | Path | Iterable[str | Path],
    **kwargs,
):
    """Run prediction on one image, a directory, or a list of images."""

    model = _load_yolo(model_or_path)
    if isinstance(sources, (str, Path)):
        source_arg = str(sources)
    else:
        source_arg = [str(source) for source in sources]
    kwargs.setdefault("save", True)
    kwargs.setdefault("conf", 0.25)
    return model.predict(source=source_arg, **kwargs)


def run_classification_lr_trial(
    class_root: str | Path,
    train_args_builder,
    *,
    lr0: float,
    repo_root: str | Path,
    n_epochs: int = 1,
    imgsz: int = 224,
    model_name: str = "yolo11n-cls.pt",
) -> dict[str, object]:
    """Run one small YOLO classification trial for a learning-rate lab.

    The notebook keeps the learning-rate values visible, while this helper
    handles the repeated Ultralytics boilerplate and result-path lookup.
    """

    model = _load_yolo(model_name)
    args = train_args_builder(
        n_epochs=n_epochs,
        imgsz=imgsz,
        batch=16,
        lr0=lr0,
        project=Path(repo_root) / "runs" / "classification_lr_lab",
        name=f"lr_{lr0:g}".replace(".", "p"),
    )
    result = model.train(data=str(class_root), **args)
    save_dir = getattr(result, "save_dir", None)
    trainer = getattr(model, "trainer", None)
    if save_dir is None and trainer is not None:
        save_dir = getattr(trainer, "save_dir", None)
    return {"lr0": lr0, "results_csv": Path(save_dir) / "results.csv" if save_dir else None}


def prediction_count_at_threshold(
    model_or_path,
    image_path: str | Path,
    *,
    conf: float,
    repo_root: str | Path,
    imgsz: int = 320,
) -> dict[str, object]:
    """Run YOLO prediction on one image and summarize detections at a threshold."""

    model = _load_yolo(model_or_path)
    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        project=Path(repo_root) / "runs" / "threshold_sweep",
        name=f"conf_{str(conf).replace('.', 'p')}",
        exist_ok=True,
        verbose=False,
    )[0]
    scores = result.boxes.conf.tolist() if result.boxes is not None else []
    return {
        "conf": conf,
        "detections": len(scores),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "result": result,
    }
