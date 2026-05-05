"""
Local text-to-image via Hugging Face diffusers (no ComfyUI server).

Works on CPU (slow), CUDA, or Apple MPS. For laptops without a discrete GPU,
use a small fast model, e.g. stabilityai/sd-turbo with a few steps.

Env:
  SD_LOCAL_MODEL     HuggingFace model id (enables this backend when set)
  USE_LOCAL_SD       If 1/true, uses default model stabilityai/sd-turbo when SD_LOCAL_MODEL is empty
  SD_LOCAL_DEVICE    cpu | cuda | mps (default: auto-detect)
  SD_LOCAL_WIDTH     default 512
  SD_LOCAL_HEIGHT    default 512
  SD_LOCAL_STEPS     default: 4 for *turbo* models else 28
  SD_LOCAL_CFG       default: 0 for *turbo* else 7.5
  SD_LOCAL_NEGATIVE  optional negative prompt (ignored for many turbo setups)
  HF_TOKEN / HUGGING_FACE_HUB_TOKEN  for gated models
"""

from __future__ import annotations

import os
import random
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

_pipe: Any = None
_pipe_key: Optional[Tuple[str, str]] = None
_pipe_lock = threading.Lock()


def _truncate_prompt(text: str, max_words: int = 60, max_chars: int = 380) -> str:
    """Heuristic truncation to avoid CLIP 77-token overflows.

    Diffusers' CLIP text encoder commonly hard-limits to 77 tokens.
    We keep the front of the prompt (subject/framing) and trim tail style spam.
    """
    raw = (text or "").strip().replace("\n", " ")
    words = raw.split()
    compact = " ".join(words[:max_words])
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip(" ,.;:-")
    return compact


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def local_sd_configured() -> bool:
    if _env_str("SD_LOCAL_MODEL"):
        return True
    return _env_str("USE_LOCAL_SD", "").lower() in ("1", "true", "yes", "on")


def _model_id() -> str:
    m = _env_str("SD_LOCAL_MODEL")
    if m:
        return m
    return "stabilityai/sd-turbo"


def _pick_device(explicit: str) -> str:
    e = explicit.lower()
    if e in ("cpu", "cuda", "mps"):
        return e
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_steps_cfg(model_id: str) -> Tuple[int, float]:
    low = model_id.lower()
    if "turbo" in low:
        return _env_int("SD_LOCAL_STEPS", 4), _env_float("SD_LOCAL_CFG", 0.0)
    return _env_int("SD_LOCAL_STEPS", 28), _env_float("SD_LOCAL_CFG", 7.5)


def _get_pipeline(model_id: str, device: str):
    global _pipe, _pipe_key
    key = (model_id, device)
    if _pipe is not None and _pipe_key == key:
        return _pipe

    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    token = _env_str("HF_TOKEN") or _env_str("HUGGING_FACE_HUB_TOKEN")
    kwargs: dict = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if token:
        kwargs["token"] = token

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
    pipe.to(device)
    if device == "cpu":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    _pipe = pipe
    _pipe_key = key
    return pipe


def generate_local_sd_image(positive_prompt: str, dest_png: Path, seed: int | None = None) -> str:
    """Run txt2img and save PNG. Returns absolute path string."""
    import torch

    model_id = _model_id()
    device = _pick_device(_env_str("SD_LOCAL_DEVICE"))
    w = max(256, min(1024, _env_int("SD_LOCAL_WIDTH", 512)))
    h = max(256, min(1024, _env_int("SD_LOCAL_HEIGHT", 512)))
    steps, cfg = _default_steps_cfg(model_id)

    prompt = _truncate_prompt(positive_prompt)

    with _pipe_lock:
        pipe = _get_pipeline(model_id, device)
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        gen = torch.Generator(device=device).manual_seed(int(seed))

        call_kw: dict = {
            "prompt": prompt[:2000],
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "width": w,
            "height": h,
            "generator": gen,
        }
        neg = _env_str("SD_LOCAL_NEGATIVE")
        if neg and cfg > 0.01:
            call_kw["negative_prompt"] = neg[:2000]

        with torch.inference_mode():
            result = pipe(**call_kw)
        image = result.images[0]

    dest_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_png)

    sidecar = dest_png.parent / f"{dest_png.stem}_sd_meta.txt"
    sidecar.write_text(
        f"model={model_id}\ndevice={device}\nseed={seed}\nsteps={steps}\ncfg={cfg}\n"
        f"width={w}\nheight={h}\nprompt=\n{positive_prompt}\n",
        encoding="utf-8",
    )
    return str(dest_png.resolve())
