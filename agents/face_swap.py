import json
import os
from pathlib import Path
from typing import Any, Dict, List

import sys

from tools.mcp_registry import invoke_tool
from tools.phase2_media import phase2_face_swap_enabled


def _scene_index(scene_manifest_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    scenes = scene_manifest_data.get("scenes", []) if isinstance(scene_manifest_data, dict) else []
    if not isinstance(scenes, list):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for scene in scenes:
        scene_id = str(scene.get("scene_id", "")).strip()
        if scene_id:
            out[scene_id] = scene
    return out


def _dialogue_characters(scene: Dict[str, Any]) -> List[str]:
    dialogues = scene.get("dialogues", [])
    if not isinstance(dialogues, list):
        return []

    names: List[str] = []
    seen = set()
    for item in dialogues:
        name = str(item.get("character", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def _reference_image_for_scene(scene: Dict[str, Any], images: List[Dict[str, Any]]) -> str:
    chars = _dialogue_characters(scene)
    if not chars:
        chars = ["Lead"]

    for char_name in chars:
        for image in images:
            if str(image.get("character", "")).strip().lower() == char_name.lower():
                return str(image.get("image_path", ""))

    if images:
        return str(images[0].get("image_path", ""))

    return ""


def _strict_production_mode() -> bool:
    raw = (os.getenv("PHASE2_STRICT_PRODUCTION") or "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _frame_count(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    count = len(list(p.glob("frame_*.png")))
    if count > 0:
        return count
    count = len(list(p.glob("frame_*.jpg")))
    if count > 0:
        return count
    return len(list(p.glob("frame_*.jpeg")))


def _resolve_frames_dir(path: str) -> str:
    """Return a directory that actually contains frame_*.{png,jpg,jpeg}.

    Motion synthesis stores frames in subfolders like _motion/ or _interpolated/.
    """
    base = Path(path)
    if not path or not base.exists():
        return path

    if _frame_count(str(base)) > 0:
        return str(base)

    for sub in ("_motion", "_interpolated", "face_swapped"):
        cand = base / sub
        if _frame_count(str(cand)) > 0:
            return str(cand)

    # Last resort: shallow search.
    try:
        for child in base.iterdir():
            if child.is_dir() and _frame_count(str(child)) > 0:
                return str(child)
    except Exception:
        pass

    return str(base)


def _hitl_override_allowed(state: Dict[str, Any]) -> bool:
    if not bool(state.get("require_hitl", False)):
        return False
    try:
        return bool(sys.stdin.isatty())
    except Exception:
        return False


def face_swap_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    video_tracks = state.get("video_tracks", [])
    if not isinstance(video_tracks, list):
        video_tracks = []

    images = state.get("images", [])
    if not isinstance(images, list):
        images = []

    scene_lookup = _scene_index(state.get("scene_manifest_data", {}))

    if not phase2_face_swap_enabled():
        outputs_disabled: List[Dict[str, Any]] = []
        for video in video_tracks:
            scene_id = str(video.get("scene_id", "")).strip()
            if not scene_id:
                continue
            frame_dir = _resolve_frames_dir(str(video.get("frame_sequence_dir", "")).strip())
            scene = scene_lookup.get(scene_id, {})
            reference_image = _reference_image_for_scene(scene, images)
            expected_character = "Lead"
            chars = _dialogue_characters(scene)
            if chars:
                expected_character = chars[0]
            outputs_disabled.append(
                {
                    "scene_id": scene_id,
                    "face_swap_report_path": "",
                    "swapped_frame_sequence_dir": frame_dir,
                    "identity_validated": False,
                    "identity_confidence": 0.0,
                    "expected_character": expected_character,
                    "reference_image": reference_image,
                    "mapped_frames": 0,
                    "face_detect_rate": 0.0,
                    "face_swap_disabled": True,
                    "quality": {"passed": True},
                    "attempt_logs": [],
                }
            )
        try:
            invoke_tool(
                "commit_memory",
                {
                    "data": {
                        "agent": "face_swap",
                        "scene_count": len(outputs_disabled),
                        "face_swap_disabled": True,
                    }
                },
            )
        except Exception:
            pass
        return {"face_swaps": outputs_disabled}

    outputs: List[Dict[str, Any]] = []
    retries = max(1, int(os.getenv("PHASE2_SCENE_RETRIES", "2")))
    strict = _strict_production_mode()
    allow_hitl_override = _hitl_override_allowed(state)
    for video in video_tracks:
        scene_id = str(video.get("scene_id", "")).strip()
        if not scene_id:
            continue

        frame_dir = _resolve_frames_dir(str(video.get("frame_sequence_dir", "")).strip())
        scene = scene_lookup.get(scene_id, {})
        reference_image = _reference_image_for_scene(scene, images)

        expected_character = "Lead"
        chars = _dialogue_characters(scene)
        if chars:
            expected_character = chars[0]

        best_result: Dict[str, Any] | None = None
        best_score = -1.0
        attempt_logs: List[Dict[str, Any]] = []

        for attempt in range(1, retries + 1):
            try:
                report_path = invoke_tool(
                    "face_swapper",
                    {
                        "scene_id": scene_id,
                        "frame_sequence_dir": frame_dir,
                        "reference_image": reference_image,
                    },
                )

                swapped_frame_sequence_dir = frame_dir
                report_payload: Dict[str, Any] = {}
                try:
                    report_payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
                    swapped_candidate = str(report_payload.get("swapped_frame_sequence_dir", "")).strip()
                    if swapped_candidate:
                        swapped_frame_sequence_dir = swapped_candidate
                except Exception:
                    report_payload = {}

                try:
                    validation = invoke_tool(
                        "identity_validator",
                        {
                            "face_swap_report_path": report_path,
                            "expected_character": expected_character,
                        },
                    )
                except Exception:
                    validation = {"validated": False, "confidence": 0.0}

                mapped_frames = int(report_payload.get("mapped_frames", 0) or 0)
                swapped_frame_sequence_dir = _resolve_frames_dir(swapped_frame_sequence_dir)
                total_frames = max(1, _frame_count(swapped_frame_sequence_dir))
                face_detect_rate = float(mapped_frames) / float(total_frames)
                identity_confidence = float(validation.get("confidence", 0.0))

                qa = invoke_tool(
                    "assess_scene_quality",
                    {
                        "scene_id": scene_id,
                        "frame_sequence_dir": swapped_frame_sequence_dir,
                        "identity_confidence": identity_confidence,
                        "face_detect_rate": face_detect_rate,
                    },
                )

                score = (identity_confidence * 0.7) + (face_detect_rate * 0.3)
                current = {
                    "scene_id": scene_id,
                    "face_swap_report_path": report_path,
                    "swapped_frame_sequence_dir": swapped_frame_sequence_dir,
                    "identity_validated": bool(validation.get("validated", False)),
                    "identity_confidence": identity_confidence,
                    "expected_character": expected_character,
                    "reference_image": reference_image,
                    "mapped_frames": mapped_frames,
                    "face_detect_rate": face_detect_rate,
                    "quality": qa,
                    "attempt": attempt,
                }

                attempt_logs.append({
                    "attempt": attempt,
                    "identity_confidence": identity_confidence,
                    "face_detect_rate": face_detect_rate,
                    "passed": bool(qa.get("passed", False)),
                })

                if score > best_score:
                    best_score = score
                    best_result = current

                if bool(qa.get("passed", False)):
                    break
            except Exception as exc:
                attempt_logs.append({"attempt": attempt, "error": str(exc)})
                if attempt >= retries and strict:
                    if allow_hitl_override:
                        answer = input(
                            f"Face swap failed for {scene_id}: {exc}. "
                            "Continue WITHOUT face swap for this scene? [y/N]: "
                        ).strip().lower()
                        if answer in {"y", "yes"}:
                            identity_confidence = 0.0
                            face_detect_rate = 0.0
                            qa = invoke_tool(
                                "assess_scene_quality",
                                {
                                    "scene_id": scene_id,
                                    "frame_sequence_dir": frame_dir,
                                    "identity_confidence": identity_confidence,
                                    "face_detect_rate": face_detect_rate,
                                },
                            )
                            best_result = {
                                "scene_id": scene_id,
                                "face_swap_report_path": "",
                                "swapped_frame_sequence_dir": frame_dir,
                                "identity_validated": False,
                                "identity_confidence": identity_confidence,
                                "expected_character": expected_character,
                                "reference_image": reference_image,
                                "mapped_frames": 0,
                                "face_detect_rate": face_detect_rate,
                                "quality": qa,
                                "attempt": attempt,
                                "face_swap_skipped": True,
                                "quality_gate_overridden": True,
                            }
                            best_score = 0.0
                            break

                    raise RuntimeError(f"Face swap failed for {scene_id}: {exc}") from exc

        if best_result is None:
            outputs.append(
                {
                    "scene_id": scene_id,
                    "face_swap_report_path": "",
                    "identity_validated": False,
                    "identity_confidence": 0.0,
                    "error": "face swap failed",
                }
            )
            continue

        best_result["attempt_logs"] = attempt_logs
        if strict and not bool((best_result.get("quality") or {}).get("passed", False)):
            if bool(best_result.get("quality_gate_overridden", False)):
                outputs.append(best_result)
                continue
            if allow_hitl_override:
                q = best_result.get("quality") or {}
                msg = (
                    f"Face swap quality gate failed for {scene_id}. "
                    f"identity={best_result.get('identity_confidence', 0.0):.3f} (min {q.get('identity_min')}), "
                    f"face_rate={best_result.get('face_detect_rate', 0.0):.3f} (min {q.get('face_rate_min')}), "
                    f"motion={q.get('motion_score')} (min {q.get('motion_min')}). "
                    "Proceed anyway? [y/N]: "
                )
                answer = input(msg).strip().lower()
                if answer in {"y", "yes"}:
                    best_result["quality_gate_overridden"] = True
                else:
                    raise RuntimeError(f"Face swap quality gate failed for {scene_id}")
            else:
                raise RuntimeError(f"Face swap quality gate failed for {scene_id}")
        outputs.append(best_result)

    try:
        invoke_tool(
            "commit_memory",
            {
                "data": {
                    "agent": "face_swap",
                    "scene_count": len(outputs),
                }
            },
        )
    except Exception:
        pass

    return {"face_swaps": outputs}
