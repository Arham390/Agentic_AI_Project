import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.mcp_registry import invoke_tool
from tools.phase2_media import _audio_duration_seconds


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return slug[:64] or "scene"


def _iter_target_scenes(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    scene_payload = state.get("scene_payload")
    if isinstance(scene_payload, dict):
        return [scene_payload]

    manifest = state.get("scene_manifest_data")
    if isinstance(manifest, dict) and isinstance(manifest.get("scenes"), list):
        return list(manifest.get("scenes") or [])

    return []


def _scene_characters(scene: Dict[str, Any]) -> List[str]:
    dialogues = scene.get("dialogues", [])
    if not isinstance(dialogues, list):
        return []

    names: List[str] = []
    seen = set()
    for dialogue in dialogues:
        name = str(dialogue.get("character", "")).strip()
        if not name:
            continue
        low = name.lower()
        if low in seen:
            continue
        seen.add(low)
        names.append(name)
    return names


def _strict_production_mode() -> bool:
    raw = (os.getenv("PHASE2_STRICT_PRODUCTION") or "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _scene_backend_marker(frame_dir: Path) -> str:
    # Frame sequences may live in subfolders (e.g., _motion/ or _interpolated/).
    # The backend marker is written to the scene root directory.
    candidates = [frame_dir, frame_dir.parent]
    for cand in candidates:
        marker = cand / "_scene_backend.txt"
        if not marker.exists():
            continue
        for line in marker.read_text(encoding="utf-8").splitlines():
            if line.startswith("backend="):
                return line.split("=", 1)[1].strip().lower()
    return ""


def _frames_dir_from_render_result(render_result: Any, fallback_dir: Path) -> Path:
    if not isinstance(render_result, dict):
        return fallback_dir

    frame_paths = render_result.get("frame_paths")
    if isinstance(frame_paths, list) and frame_paths:
        first = str(frame_paths[0])
        if first:
            return Path(first).resolve().parent

    candidate = str(render_result.get("frame_sequence_dir", "") or "").strip()
    if candidate:
        return Path(candidate).resolve()
    return fallback_dir


def _primary_audio_path(state: Dict[str, Any], scene_id: str) -> Optional[Path]:
    sid = str(scene_id).strip()
    for entry in state.get("audio_tracks") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("scene_id", "")).strip() != sid:
            continue
        primary = str(entry.get("primary_audio", "")).strip()
        if primary:
            return Path(primary)
        clips = entry.get("clips", [])
        if isinstance(clips, list) and clips:
            first = clips[0]
            if isinstance(first, dict):
                ap = str(first.get("audio_path", "")).strip()
                if ap:
                    return Path(ap)
    return None


def _estimate_speech_duration_sec(scene: Dict[str, Any]) -> float:
    """Rough VO length when no WAV is available yet (e.g. parallel voice/video graph)."""
    secs = 0.0
    for d in scene.get("dialogues") or []:
        if not isinstance(d, dict):
            continue
        text = str(d.get("line", d.get("text", ""))).strip()
        if not text:
            continue
        words = max(1, len(text.split()))
        secs += max(0.35, words / 2.5)
    return max(2.0, secs) if secs > 0 else 4.0


def _target_frame_count(scene: Dict[str, Any], scene_id: str, state: Dict[str, Any]) -> int:
    """Return the number of frames needed for this scene.

    Frame count tracks the actual audio duration so that the visual length
    matches the speech exactly — no artificial padding or stretching.
    """
    dialogues = scene.get("dialogues", [])
    dialogue_count = len(dialogues) if isinstance(dialogues, list) else 0
    fps = max(6, int(os.getenv("PHASE2_VIDEO_TARGET_FPS", "24")))
    cap = max(60, int(os.getenv("PHASE2_MAX_SCENE_FRAMES", "720")))
    base = max(10, 8 + dialogue_count * 4)

    dur = 0.0
    ap = _primary_audio_path(state, scene_id)
    if ap is not None and ap.exists():
        dur = float(_audio_duration_seconds(ap))
    if dur <= 0.0:
        dur = _estimate_speech_duration_sec(scene)

    needed = int(math.ceil(dur * fps))
    return min(cap, max(base, needed))


def _write_generation_metrics(scene_id: str, attempts: List[Dict[str, Any]]) -> None:
    qa_dir = Path(__file__).resolve().parents[1] / "outputs" / "qa_metrics"
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / f"{_slugify(scene_id)}_video_gen_attempts.json"
    path.write_text(json.dumps({"scene_id": scene_id, "attempts": attempts}, indent=2), encoding="utf-8")


def _build_scene_video(scene: Dict[str, Any], idx: int, state: Dict[str, Any]) -> Dict[str, Any]:
    scene_id = str(scene.get("scene_id", f"scene_{idx:02d}"))
    heading = str(scene.get("heading", f"Scene {idx}"))

    frames_dir = Path(__file__).resolve().parents[1] / "outputs" / "intermediate_frames" / _slugify(scene_id)
    frames_dir.mkdir(parents=True, exist_ok=True)

    character_names = _scene_characters(scene)
    lead_name = character_names[0] if character_names else "Lead"

    # Scene-level style override always wins (set by edit_agent or scene manifest).
    scene_style = str(scene.get("style", "")).strip()

    if scene_style:
        # Blend the scene style with the character reference style for a richer prompt.
        try:
            ref = invoke_tool("query_stock_footage", {"character_name": lead_name})
            char_style = str(ref.get("reference_style", "")) if isinstance(ref, dict) else ""
        except Exception:
            char_style = ""
        style = f"{scene_style}, {char_style}".strip(", ") if char_style else scene_style
    else:
        try:
            ref = invoke_tool("query_stock_footage", {"character_name": lead_name})
            style = str(ref.get("reference_style", "cinematic framing")) if isinstance(ref, dict) else "cinematic framing"
        except Exception:
            style = "cinematic framing"

    frame_count = _target_frame_count(scene, scene_id, state)

    reference_image = ""
    images = scene.get("_images", [])
    if isinstance(images, list):
        for img in images:
            if str(img.get("character", "")).strip().lower() == lead_name.lower():
                reference_image = str(img.get("image_path", "")).strip()
                break
        if not reference_image and images:
            reference_image = str(images[0].get("image_path", "")).strip()

    render_result = invoke_tool(
        "render_frame_sequence",
        {
            "scene_id": scene_id,
            "heading": heading,
            "style": style,
            "frame_sequence_dir": str(frames_dir),
            "frame_count": frame_count,
            "reference_image": reference_image,
        },
    )
    effective_frames_dir = _frames_dir_from_render_result(render_result, frames_dir)
    rendered_count = (
        len(render_result.get("frame_paths") or [])
        if isinstance(render_result, dict) and isinstance(render_result.get("frame_paths"), list)
        else int(render_result.get("frame_count", frame_count))
        if isinstance(render_result, dict)
        else frame_count
    )

    return {
        "scene_id": scene_id,
        "heading": heading,
        # IMPORTANT: downstream expects a directory that actually contains frame_*.png.
        "frame_sequence_dir": str(effective_frames_dir.resolve()),
        "frame_count": rendered_count,
        "lead_character": lead_name,
        "reference_image": reference_image,
    }


def video_gen_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    scenes = _iter_target_scenes(state)
    images = state.get("images", [])
    if not isinstance(images, list):
        images = []

    outputs: List[Dict[str, Any]] = []
    retries = max(1, int(os.getenv("PHASE2_SCENE_RETRIES", "2")))
    strict = _strict_production_mode()

    for idx, scene in enumerate(scenes, start=1):
        scene_id = str(scene.get("scene_id", f"scene_{idx:02d}"))
        attempts: List[Dict[str, Any]] = []
        best: Dict[str, Any] | None = None

        for attempt in range(1, retries + 1):
            try:
                with_images = dict(scene)
                with_images["_images"] = images
                candidate = _build_scene_video(with_images, idx, state)
                backend = _scene_backend_marker(Path(candidate.get("frame_sequence_dir", "")))
                candidate["backend"] = backend
                candidate["attempt"] = attempt
                attempts.append({"attempt": attempt, "backend": backend, "frame_count": int(candidate.get("frame_count", 0))})

                if strict and not ("local-sd" in backend or "comfyui" in backend or "motion-synth" in backend):
                    raise RuntimeError(
                        f"Scene {scene_id} used non-production backend '{backend or 'unknown'}'"
                    )

                if best is None or int(candidate.get("frame_count", 0)) > int(best.get("frame_count", 0)):
                    best = candidate
                if best:
                    break
            except Exception as exc:
                attempts.append({"attempt": attempt, "error": str(exc)})
                if attempt >= retries and strict:
                    _write_generation_metrics(scene_id, attempts)
                    raise RuntimeError(f"Scene generation failed for {scene_id}: {exc}") from exc

        _write_generation_metrics(scene_id, attempts)
        if best is not None:
            outputs.append(best)
        else:
            outputs.append(
                {
                    "scene_id": scene_id,
                    "frame_sequence_dir": "",
                    "frame_count": 0,
                    "error": "scene generation failed",
                }
            )

    try:
        invoke_tool(
            "commit_memory",
            {
                "data": {
                    "agent": "video_gen",
                    "scene_video_count": len(outputs),
                }
            },
        )
    except Exception:
        pass

    return {"video_tracks": outputs}
