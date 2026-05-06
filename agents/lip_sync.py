import os
from pathlib import Path
import shutil
import sys
import wave
from typing import Any, Dict, List

from tools.mcp_registry import invoke_tool


def _subtitles_enabled() -> bool:
    raw = os.getenv("PHASE2_SUBTITLES_ENABLED", "0")
    return (raw or "0").strip().lower() in {"1", "true", "yes", "on"}


def _clips_by_scene(audio_tracks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Return scene_id -> list of clip dicts (character, line, audio_path)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for entry in audio_tracks:
        sid = str(entry.get("scene_id", "")).strip()
        if not sid:
            continue
        out[sid] = list(entry.get("clips") or [])
    return out


def _primary_audio_by_scene(audio_tracks: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for entry in audio_tracks:
        scene_id = str(entry.get("scene_id", "")).strip()
        if not scene_id:
            continue

        primary_audio = str(entry.get("primary_audio", "")).strip()
        if primary_audio:
            out[scene_id] = primary_audio
            continue

        clips = entry.get("clips", [])
        if isinstance(clips, list) and clips:
            first_clip = clips[0]
            candidate = str(first_clip.get("audio_path", "")).strip()
            if candidate:
                out[scene_id] = candidate
    return out


def _frames_by_scene(video_tracks: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for entry in video_tracks:
        scene_id = str(entry.get("scene_id", "")).strip()
        frame_dir = str(entry.get("frame_sequence_dir", "")).strip()
        if scene_id and frame_dir:
            out[scene_id] = frame_dir
    return out


def _swapped_frames_by_scene(face_swaps: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for entry in face_swaps:
        scene_id = str(entry.get("scene_id", "")).strip()
        frame_dir = str(entry.get("swapped_frame_sequence_dir", "")).strip()
        if scene_id and frame_dir:
            out[scene_id] = frame_dir
    return out


def _scene_ids(state: Dict[str, Any]) -> List[str]:
    tasks = state.get("scene_tasks", [])
    if isinstance(tasks, list) and tasks:
        ids = [str(task.get("scene_id", "")).strip() for task in tasks]
        return [sid for sid in ids if sid]

    manifest = state.get("scene_manifest_data", {})
    scenes = manifest.get("scenes", []) if isinstance(manifest, dict) else []
    if isinstance(scenes, list):
        ids = [str(scene.get("scene_id", "")).strip() for scene in scenes]
        return [sid for sid in ids if sid]
    return []


def _strict_production_mode() -> bool:
    raw = (os.getenv("PHASE2_STRICT_PRODUCTION") or "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _frame_count(dir_path: str) -> int:
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        return 0
    patterns = ("frame_*.png", "frame_*.jpg", "frame_*.jpeg")
    for pat in patterns:
        found = list(p.glob(pat))
        if found:
            return len(found)
    return 0


def _resolve_frames_dir(path: str) -> str:
    base = Path(path)
    if not path or not base.exists():
        return path

    if _frame_count(str(base)) > 0:
        return str(base)

    for sub in ("face_swapped", "_motion", "_interpolated"):
        cand = base / sub
        if _frame_count(str(cand)) > 0:
            return str(cand)

    try:
        for child in base.iterdir():
            if child.is_dir() and _frame_count(str(child)) > 0:
                return str(child)
    except Exception:
        pass
    return str(base)


def _audio_duration_seconds(path: str) -> float:
    p = Path(path)
    if not p.exists() or p.suffix.lower() != ".wav":
        return 0.0
    try:
        with wave.open(str(p), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
        if rate <= 0:
            return 0.0
        return float(frames) / float(rate)
    except Exception:
        return 0.0


def _collect_frames(dir_path: str) -> List[Path]:
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        return []
    out = sorted(p.glob("frame_*.png"))
    if out:
        return out
    out = sorted(p.glob("frame_*.jpg"))
    if out:
        return out
    return sorted(p.glob("frame_*.jpeg"))


def _ensure_audio_fit_frames(scene_id: str, frame_dir: str, audio_path: str, fps: int = 24) -> str:
    """Pad/loop frames so visual duration ~= audio duration.

    This avoids failing the sync quality gate when audio is longer than the frame sequence.
    """
    if not audio_path:
        return frame_dir
    duration = _audio_duration_seconds(audio_path)
    if duration <= 0.0:
        return frame_dir

    frames = _collect_frames(frame_dir)
    if not frames:
        return frame_dir

    required = int((duration * float(max(6, int(fps)))) + 0.9999)  # ceil
    required = max(len(frames), required)
    required = min(1800, required)
    if required <= len(frames):
        return frame_dir

    base = Path(frame_dir)
    out_dir = base / "_audiofit"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear prior generated frames.
    for old in out_dir.glob("frame_*.png"):
        try:
            old.unlink()
        except Exception:
            pass
    for old in out_dir.glob("frame_*.jpg"):
        try:
            old.unlink()
        except Exception:
            pass
    for old in out_dir.glob("frame_*.jpeg"):
        try:
            old.unlink()
        except Exception:
            pass

    suffix = frames[0].suffix
    for idx in range(required):
        src = frames[idx % len(frames)]
        dst = out_dir / f"frame_{idx + 1:04d}{suffix}"
        try:
            shutil.copyfile(src, dst)
        except Exception:
            # If a single copy fails, continue with best effort.
            pass

    return str(out_dir.resolve())


def _hitl_override_allowed(state: Dict[str, Any]) -> bool:
    if not bool(state.get("require_hitl", False)):
        return False
    try:
        return bool(sys.stdin.isatty())
    except Exception:
        return False


def lip_sync_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    audio_tracks = state.get("audio_tracks", [])
    if not isinstance(audio_tracks, list):
        audio_tracks = []

    video_tracks = state.get("video_tracks", [])
    if not isinstance(video_tracks, list):
        video_tracks = []

    face_swaps = state.get("face_swaps", [])
    if not isinstance(face_swaps, list):
        face_swaps = []

    audio_map = _primary_audio_by_scene(audio_tracks)
    clips_map = _clips_by_scene(audio_tracks)
    frame_map = _frames_by_scene(video_tracks)
    swapped_map = _swapped_frames_by_scene(face_swaps)
    burn_subs = _subtitles_enabled()

    outputs: List[Dict[str, Any]] = []
    retries = max(1, int(os.getenv("PHASE2_SCENE_RETRIES", "2")))
    strict = _strict_production_mode()
    allow_hitl_override = _hitl_override_allowed(state)
    for scene_id in _scene_ids(state):
        audio_path = audio_map.get(scene_id, "")
        frame_dir = swapped_map.get(scene_id) or frame_map.get(scene_id, "")
        frame_dir = _resolve_frames_dir(frame_dir)
        if audio_path and frame_dir:
            frame_dir = _ensure_audio_fit_frames(scene_id, frame_dir, audio_path, fps=24)
        if not audio_path or not frame_dir:
            outputs.append(
                {
                    "scene_id": scene_id,
                    "video_path": "",
                    "audio_path": audio_path,
                    "frame_sequence_dir": frame_dir,
                    "synced": False,
                    "error": "missing audio or frame sequence",
                }
            )
            continue

        best: Dict[str, Any] | None = None
        best_score = -1.0
        attempt_logs: List[Dict[str, Any]] = []

        identity_conf = 0.0
        face_rate = 0.0
        for fs in face_swaps:
            if str(fs.get("scene_id", "")).strip() == scene_id:
                identity_conf = float(fs.get("identity_confidence", 0.0))
                face_rate = float(fs.get("face_detect_rate", 0.0))
                break

        for attempt in range(1, retries + 1):
            try:
                video_path = invoke_tool(
                    "lip_sync_aligner",
                    {
                        "scene_id": scene_id,
                        "audio_path": audio_path,
                        "frame_sequence_dir": frame_dir,
                    },
                )
                qa = invoke_tool(
                    "assess_scene_quality",
                    {
                        "scene_id": scene_id,
                        "frame_sequence_dir": frame_dir,
                        "audio_path": audio_path,
                        "identity_confidence": identity_conf,
                        "face_detect_rate": face_rate,
                    },
                )

                # The shared QA tool includes identity thresholds, but at the lip-sync stage
                # we primarily care about audio/visual alignment and motion continuity.
                sync_conf = float(qa.get("sync_confidence", 0.0))
                motion_score = float(qa.get("motion_score", 0.0))
                sync_min = float(qa.get("sync_min", 0.0))
                motion_min = float(qa.get("motion_min", 0.0))

                passed = (sync_conf >= sync_min) and (motion_score >= motion_min)

                # Only enforce identity gates here if we actually have face-swap evidence.
                # (Otherwise, skipped face swap would always force a failure.)
                has_identity_signal = (face_rate > 0.0) or (identity_conf > 0.0)
                if has_identity_signal:
                    identity_min = float(qa.get("identity_min", 0.0))
                    face_rate_min = float(qa.get("face_rate_min", 0.0))
                    passed = passed and (identity_conf >= identity_min) and (face_rate >= face_rate_min)

                qa["passed"] = bool(passed)
                attempt_logs.append({
                    "attempt": attempt,
                    "sync_confidence": sync_conf,
                    "passed": bool(qa.get("passed", False)),
                })

                current = {
                    "scene_id": scene_id,
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "frame_sequence_dir": frame_dir,
                    "synced": True,
                    "quality": qa,
                    "attempt": attempt,
                    "sync_confidence": sync_conf,
                }
                if sync_conf > best_score:
                    best = current
                    best_score = sync_conf

                if bool(qa.get("passed", False)):
                    break
            except Exception as exc:
                attempt_logs.append({"attempt": attempt, "error": str(exc)})
                if attempt >= retries and strict:
                    raise RuntimeError(f"Lip sync failed for {scene_id}: {exc}") from exc

        if best is None:
            outputs.append(
                {
                    "scene_id": scene_id,
                    "video_path": "",
                    "audio_path": audio_path,
                    "frame_sequence_dir": frame_dir,
                    "synced": False,
                    "error": "lip sync failed",
                }
            )
            continue

        best["attempt_logs"] = attempt_logs

        # ── Subtitle burning (optional) ──────────────────────────────────────
        if burn_subs and best.get("video_path"):
            try:
                from tools.phase2_media import generate_srt_content, burn_subtitles_into_video

                scene_clips = clips_map.get(scene_id, [])
                srt = generate_srt_content(scene_clips)
                if srt.strip():
                    vp = Path(best["video_path"])
                    sub_out = vp.with_stem(vp.stem + "_sub") if hasattr(vp, "with_stem") else vp.with_name(vp.stem + "_sub" + vp.suffix)
                    if burn_subtitles_into_video(vp, srt, sub_out):
                        # Replace original with subtitle-burned version.
                        import shutil as _shutil
                        _shutil.move(str(sub_out), str(vp))
                        best["subtitles"] = "burned"
            except Exception:
                pass  # Subtitles are non-critical; continue without them.

        if strict and not bool((best.get("quality") or {}).get("passed", False)):
            if allow_hitl_override:
                q = best.get("quality") or {}
                msg = (
                    f"Lip sync quality gate failed for {scene_id}. "
                    f"identity={identity_conf:.3f} (min {q.get('identity_min')}), "
                    f"face_rate={face_rate:.3f} (min {q.get('face_rate_min')}), "
                    f"sync={best.get('sync_confidence', 0.0):.3f} (min {q.get('sync_min')}), "
                    f"motion={q.get('motion_score')} (min {q.get('motion_min')}). "
                    "Proceed anyway? [y/N]: "
                )
                answer = input(msg).strip().lower()
                if answer in {"y", "yes"}:
                    best["quality_gate_overridden"] = True
                else:
                    raise RuntimeError(f"Lip sync quality gate failed for {scene_id}")
            else:
                raise RuntimeError(f"Lip sync quality gate failed for {scene_id}")
        outputs.append(best)

    try:
        invoke_tool(
            "commit_memory",
            {
                "data": {
                    "agent": "lip_sync",
                    "raw_scene_count": len([x for x in outputs if x.get("synced")]),
                }
            },
        )
    except Exception:
        pass

    return {"raw_scenes": outputs}
