import json
from pathlib import Path
from typing import Any, Dict, List

from tools.mcp_registry import invoke_tool


def _characters_for_voice(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    ch = state.get("characters")
    if isinstance(ch, list) and ch:
        return ch
    db = Path(__file__).resolve().parents[1] / "outputs" / "character_db.json"
    if db.exists():
        try:
            payload = json.loads(db.read_text(encoding="utf-8"))
            inner = payload.get("characters")
            if isinstance(inner, list):
                return inner
        except Exception:
            pass
    return []


def _character_voice_gender_map(state: Dict[str, Any]) -> Dict[str, str]:
    """Lowercased character name -> voice_gender for Edge TTS pool selection."""
    chars = _characters_for_voice(state)
    if not isinstance(chars, list):
        return {}
    out: Dict[str, str] = {}
    for c in chars:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "")).strip()
        if not name:
            continue
        vg = str(c.get("voice_gender", "neutral")).strip().lower()
        if vg not in ("male", "female", "neutral"):
            vg = "neutral"
        out[name.lower()] = vg
    return out


def _iter_target_scenes(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    scene_payload = state.get("scene_payload")
    if isinstance(scene_payload, dict):
        return [scene_payload]

    manifest = state.get("scene_manifest_data")
    if isinstance(manifest, dict) and isinstance(manifest.get("scenes"), list):
        return list(manifest.get("scenes") or [])

    return []


def _scene_voice_payload(scene: Dict[str, Any], idx: int, voice_gender_by_character: Dict[str, str]) -> Dict[str, Any]:
    scene_id = str(scene.get("scene_id", f"scene_{idx:02d}"))
    heading = str(scene.get("heading", f"Scene {idx}"))

    dialogues = scene.get("dialogues", [])
    if not isinstance(dialogues, list):
        dialogues = []

    if not dialogues:
        dialogues = [
            {
                "character": "Narrator",
                "line": f"{heading}. Silent beat for visual continuity.",
            }
        ]

    clips: List[Dict[str, Any]] = []
    for clip_index, dialogue in enumerate(dialogues, start=1):
        character = str(dialogue.get("character", "Narrator"))
        line = str(dialogue.get("line", "")) or f"Narration for {heading}."
        voice_gender = voice_gender_by_character.get(character.strip().lower(), "neutral")
        audio_path = invoke_tool(
            "voice_cloning_synthesizer",
            {
                "scene_id": scene_id,
                "character_name": character,
                "text": line,
                "emotion": "neutral",
                "voice_gender": voice_gender,
            },
        )
        clips.append(
            {
                "clip_id": f"{scene_id}_clip_{clip_index:02d}",
                "character": character,
                "line": line,
                "audio_path": audio_path,
            }
        )

    primary_audio = clips[0]["audio_path"] if clips else ""
    return {
        "scene_id": scene_id,
        "primary_audio": primary_audio,
        "clips": clips,
    }


def voice_synth_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    scenes = _iter_target_scenes(state)
    outputs: List[Dict[str, Any]] = []
    voice_gender_by_character = _character_voice_gender_map(state)

    for idx, scene in enumerate(scenes, start=1):
        try:
            outputs.append(_scene_voice_payload(scene, idx, voice_gender_by_character))
        except Exception as exc:
            scene_id = str(scene.get("scene_id", f"scene_{idx:02d}"))
            outputs.append(
                {
                    "scene_id": scene_id,
                    "primary_audio": "",
                    "clips": [],
                    "error": str(exc),
                }
            )

    try:
        invoke_tool(
            "commit_memory",
            {
                "data": {
                    "agent": "voice_synth",
                    "scene_audio_count": len(outputs),
                }
            },
        )
    except Exception:
        pass

    return {"audio_tracks": outputs}
