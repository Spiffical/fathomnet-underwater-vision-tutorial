"""SAM3 helpers with cached fallbacks for classroom use."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def sam3_can_run_live() -> dict[str, object]:
    """Check whether this runtime is likely able to run live SAM3 inference."""

    info: dict[str, object] = {
        "python_ok": sys.version_info >= (3, 12),
        "sam3_importable": importlib.util.find_spec("sam3") is not None,
        "torch_importable": importlib.util.find_spec("torch") is not None,
        "cuda_available": False,
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "can_run": False,
    }
    if info["torch_importable"]:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
    info["can_run"] = bool(
        info["python_ok"]
        and info["sam3_importable"]
        and info["cuda_available"]
        and info["hf_token_present"]
    )
    return info


def _load_index(bundle_root: str | Path) -> dict:
    index_path = Path(bundle_root) / "sam3_cached_outputs" / "index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def available_cached_prompts(bundle_root: str | Path, image_id: str | None = None) -> dict[str, list[str]]:
    """List cached prompts available in the tutorial bundle."""

    index = _load_index(bundle_root)
    if image_id is not None:
        return {image_id: sorted(index["images"].get(image_id, {}).get("prompts", {}).keys())}
    return {
        key: sorted(value.get("prompts", {}).keys())
        for key, value in index.get("images", {}).items()
    }


def load_cached_sam3_result(
    bundle_root: str | Path,
    image_id: str,
    prompt: str | None = None,
) -> dict:
    """Load one cached SAM3-like output from the bundle."""

    root = Path(bundle_root)
    index = _load_index(root)
    image_entry = index["images"][image_id]
    prompts = image_entry["prompts"]
    chosen_prompt = prompt or next(iter(prompts))
    if chosen_prompt not in prompts:
        raise KeyError(f"No cached prompt {chosen_prompt!r} for image {image_id!r}")
    result_path = root / "sam3_cached_outputs" / prompts[chosen_prompt]
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    result["image_path"] = str((root / image_entry["image"]).resolve())
    return result


def run_sam3_text_prompt(
    image_path: str | Path,
    prompt: str,
    *,
    confidence_threshold: float = 0.5,
) -> dict:
    """Run live SAM3 image inference for one text prompt.

    This intentionally follows the official image predictor example. It is not
    called by default in the workshop because checkpoint access and GPU support
    can vary across Colab sessions.
    """

    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    image = Image.open(image_path).convert("RGB")
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    def _tolist(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    return {
        "prompt": prompt,
        "image_path": str(image_path),
        "boxes": _tolist(output.get("boxes", [])),
        "scores": _tolist(output.get("scores", [])),
        "masks": _tolist(output.get("masks", [])),
        "source": "live_sam3",
    }
