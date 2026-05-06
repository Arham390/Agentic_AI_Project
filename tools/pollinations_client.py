"""Pollinations.ai image-generation backend.

API: https://image.pollinations.ai/prompt/{url-encoded-prompt}?...

No API key required. Returns raw PNG/JPEG bytes which we save to the standard
outputs/image_assets directory.

Toggling: set IMAGE_BACKEND=pollinations (default) or IMAGE_BACKEND=local.
"""
from __future__ import annotations

import hashlib
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional


_BASE_URL = "https://image.pollinations.ai/prompt/"

# Allowed Pollinations models. flux is the highest-quality default; turbo is
# fastest. Override per-call via env if needed.
_DEFAULT_MODEL = "flux"

_NEGATIVE_DEFAULT = (
    "blurry, low quality, deformed, extra limbs, watermark, signature, text, logo, "
    "bad anatomy, distorted face, ugly, low resolution, jpeg artifacts"
)


def pollinations_configured() -> bool:
    """Pollinations is the default cloud backend — only disabled if user opts to local."""
    backend = (os.getenv("IMAGE_BACKEND") or "pollinations").strip().lower()
    return backend in ("pollinations", "cloud", "api")


def _sanitize_for_url(text: str, limit: int = 1500) -> str:
    """Trim and url-encode the prompt; Pollinations supports very long prompts but
    we cap at 1500 chars to avoid 414-style failures."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().replace("\n", " ").replace("\r", " ")
    if len(text) > limit:
        text = text[:limit]
    return urllib.parse.quote(text, safe="")


def _seed_from_prompt(prompt: str) -> int:
    """Deterministic seed so the same prompt produces the same image."""
    digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 999_999_999


def generate_pollinations_image(
    prompt: str,
    out_path: Path,
    *,
    width: int = 1024,
    height: int = 1024,
    model: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    timeout: float = 90.0,
    nologo: bool = True,
) -> str:
    """Fetch a Pollinations.ai image and write it to *out_path* (as PNG)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chosen_model = (model or os.getenv("POLLINATIONS_MODEL") or _DEFAULT_MODEL).strip()
    encoded = _sanitize_for_url(prompt)
    seed = _seed_from_prompt(prompt)

    params = {
        "model": chosen_model,
        "width": int(width),
        "height": int(height),
        "seed": seed,
        "enhance": "true",
    }
    if nologo:
        params["nologo"] = "true"
    if negative_prompt or os.getenv("POLLINATIONS_NEGATIVE"):
        params["negative_prompt"] = (negative_prompt or os.getenv("POLLINATIONS_NEGATIVE")
                                     or _NEGATIVE_DEFAULT)

    qs = urllib.parse.urlencode(params)
    url = f"{_BASE_URL}{encoded}?{qs}"

    # Pollinations occasionally rate-limits; one retry with backoff.
    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Agentic-AI-Project/1.0 (+pollinations-client)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data or len(data) < 1024:
                raise RuntimeError(f"Pollinations returned {len(data)} bytes (likely error)")
            out_path.write_bytes(data)
            return str(out_path.resolve())
        except Exception as exc:
            last_err = exc
            if attempt == 0:
                time.sleep(2.0)
                continue
            raise RuntimeError(f"Pollinations request failed: {exc}") from exc

    # Unreachable, but for type-checker.
    raise RuntimeError(f"Pollinations request failed: {last_err}")
