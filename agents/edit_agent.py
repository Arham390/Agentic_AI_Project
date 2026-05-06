"""Phase 5 — Intelligent Edit & Undo Agent.

Accepts a free-text edit instruction, classifies intent via LLM (with a
rule-based fallback), and executes the targeted phase re-run.  Every edit is
snapshotted so the user can undo any number of times.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.lc_chains import (
    get_intent_chain,
    get_scene_patch_chain,
    get_style_extraction_chain,
)
from tools.llm_factory import describe_llm, get_chat_llm, llm_configured


def _slug_asset(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return (slug[:48] or "asset")


def hydrate_state_for_edit(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing pipeline fields from outputs/ so edits work without a perfect snapshot.

    Load order:
    1. Start from the provided state dict.
    2. Merge in the full pipeline_state.json saved after the last run (fills
       audio_tracks, video_tracks, face_swaps, characters, images, etc.).
    3. Patch individual files (manifest, script, character_db) on top.
    """
    s = dict(state)

    # ── 1. Load full saved pipeline state (survives server restarts) ──────────
    saved_path = _OUTPUTS_DIR / "pipeline_state.json"
    if saved_path.exists():
        try:
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            for key in (
                "audio_tracks", "video_tracks", "face_swaps", "raw_scenes",
                "scene_tasks", "characters", "images", "script",
                "scene_manifest_data",
            ):
                if not s.get(key):
                    val = saved.get(key)
                    if isinstance(val, list) and val:
                        s[key] = val
                    elif isinstance(val, dict) and val:
                        s[key] = val
                    elif isinstance(val, str) and val.strip():
                        s[key] = val
        except Exception:
            pass

    # ── 2. Individual file overrides (most authoritative for these fields) ────
    man_path = _OUTPUTS_DIR / "scene_manifest.json"
    m = s.get("scene_manifest_data")
    if (not isinstance(m, dict) or not m.get("scenes")) and man_path.exists():
        try:
            s["scene_manifest_data"] = json.loads(man_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not str(s.get("script", "")).strip():
        sp = _OUTPUTS_DIR / "script.txt"
        if sp.exists():
            try:
                s["script"] = sp.read_text(encoding="utf-8")
            except Exception:
                pass

    if not s.get("characters"):
        ch_path = _OUTPUTS_DIR / "character_db.json"
        if ch_path.exists():
            try:
                payload = json.loads(ch_path.read_text(encoding="utf-8"))
                inner = payload.get("characters")
                if isinstance(inner, list) and inner:
                    s["characters"] = inner
            except Exception:
                pass

    if not s.get("images"):
        img_dir = _OUTPUTS_DIR / "image_assets"
        images: List[Dict[str, Any]] = []
        if img_dir.exists() and isinstance(s.get("characters"), list):
            for c in s["characters"]:
                if not isinstance(c, dict):
                    continue
                name = str(c.get("name", "")).strip()
                if not name:
                    continue
                slug = _slug_asset(name)
                found = ""
                for ext in (".png", ".jpg", ".jpeg", ".webp"):
                    for p in sorted(img_dir.glob(f"*{ext}")):
                        low = p.name.lower()
                        if slug in low or name.lower().replace(" ", "_") in low:
                            found = str(p.resolve())
                            break
                    if found:
                        break
                if found:
                    images.append({"character": name, "image_path": found})
        if images:
            s["images"] = images

    return s


def _deterministic_patch_scenes(
    scenes_slice: List[Dict[str, Any]], instruction: str
) -> Optional[List[Dict[str, Any]]]:
    """Apply pattern-based dialogue rewrites without an LLM.

    Handles common phrasings so partial edits still work when the model is
    unavailable or returns unparseable JSON:

      • "make CHARACTER say 'NEW LINE'"
      • "change CHARACTER's line to 'NEW LINE'"
      • "replace 'OLD' with 'NEW'"
    """
    if not scenes_slice or not instruction.strip():
        return None

    instr = instruction.strip()
    out: List[Dict[str, Any]] = [dict(s) for s in scenes_slice if isinstance(s, dict)]
    changed = False

    say_re = re.compile(
        r"(?:make|have)\s+(\w[\w\s]+?)\s+say\s+['\"](.+?)['\"]",
        re.IGNORECASE,
    )
    change_line_re = re.compile(
        r"(?:change|update|set)\s+(\w[\w\s]+?)(?:'s)?\s+(?:line|dialogue)\s+to\s+['\"](.+?)['\"]",
        re.IGNORECASE,
    )
    replace_re = re.compile(
        r"replace\s+['\"](.+?)['\"]\s+with\s+['\"](.+?)['\"]",
        re.IGNORECASE,
    )

    for m in list(say_re.finditer(instr)) + list(change_line_re.finditer(instr)):
        char = m.group(1).strip().lower()
        new_line = m.group(2).strip()
        for scene in out:
            for d in scene.get("dialogues") or []:
                if str(d.get("character", "")).strip().lower() == char:
                    d["line"] = new_line
                    changed = True

    for m in replace_re.finditer(instr):
        old, new = m.group(1), m.group(2)
        for scene in out:
            for d in scene.get("dialogues") or []:
                line = str(d.get("line", ""))
                if old in line:
                    d["line"] = line.replace(old, new)
                    changed = True

    return out if changed else None


def _llm_patch_scenes_for_instruction(
    scenes_slice: List[Dict[str, Any]], instruction: str
) -> Optional[List[Dict[str, Any]]]:
    """Use the LangChain scene_patch chain; fall back to deterministic patcher.

    Returns updated scene dicts (same scene_ids) or None if both paths fail.
    """
    if not scenes_slice or not instruction.strip():
        return None

    chain = get_scene_patch_chain(temperature=0.2)
    if chain is not None:
        try:
            payload = json.dumps(scenes_slice, indent=2, ensure_ascii=True)
            patch_list = chain.invoke({"instruction": instruction, "scenes": payload})
            if patch_list is not None:
                by_id = {p.scene_id: p for p in patch_list.scenes}
                out: List[Dict[str, Any]] = []
                for sc in scenes_slice:
                    if not isinstance(sc, dict):
                        continue
                    sid = str(sc.get("scene_id", "")).strip()
                    if sid and sid in by_id:
                        merged = dict(sc)
                        p = by_id[sid]
                        if p.heading is not None:
                            merged["heading"] = p.heading
                        if p.dialogues is not None:
                            merged["dialogues"] = [d.model_dump() for d in p.dialogues]
                        if p.actions is not None:
                            merged["actions"] = list(p.actions)
                        if p.style is not None:
                            merged["style"] = p.style
                        out.append(merged)
                    else:
                        out.append(sc)
                if out:
                    return out
        except Exception:
            pass

    # Fall back to deterministic patcher so dialogue edits still work without an LLM.
    return _deterministic_patch_scenes(scenes_slice, instruction)


def _merge_scene_patch_into_manifest(
    scene_manifest: Dict[str, Any], patched: List[Dict[str, Any]]
) -> None:
    scenes = scene_manifest.get("scenes")
    if not isinstance(scenes, list) or not isinstance(patched, list):
        return
    by_id = {str(p.get("scene_id", "")).strip(): p for p in patched if isinstance(p, dict) and p.get("scene_id")}
    for i, sc in enumerate(scenes):
        if not isinstance(sc, dict):
            continue
        sid = str(sc.get("scene_id", "")).strip()
        if sid in by_id:
            p = by_id[sid]
            merged = dict(sc)
            for k in ("heading", "dialogues", "actions", "style"):
                if k in p and p[k] is not None:
                    merged[k] = p[k]
            scenes[i] = merged


# ─────────────────────────────────────────────────────────────────────────────
# Intent schema
# ─────────────────────────────────────────────────────────────────────────────

VALID_INTENTS = {
    "change_voice_tone": "audio",
    "add_background_music": "audio",
    "change_voice_speed": "audio",
    "re_synthesize_audio": "audio",
    "make_scene_darker": "video_frame",
    "make_scene_brighter": "video_frame",
    "change_character_design": "video_frame",
    "change_visual_style": "video_frame",
    "re_generate_visuals": "video_frame",
    "remove_subtitle": "video",
    "speed_up_scene": "video",
    "slow_down_scene": "video",
    "recompose_video": "video",
    "regenerate_script": "script",
    "add_scene": "script",
    "change_scene_dialogue": "scene",
    "unknown": "unknown",
}

INTENT_KEYWORDS: List[tuple[str, str]] = [
    (r"voice|tone|speak|narrat|tts|audio|sound", "re_synthesize_audio"),
    (r"music|background|bgm|soundtrack|score", "add_background_music"),
    (
        r"dark|bright|light|color|colour|visual|style|look|aesthetic|"
        r"sci.fi|scifi|cyberpunk|noir|horror|fantasy|western|neon|gothic|"
        r"cinematic|dramatic|moody|theme|atmosphere|vibe|gritty|pastel|"
        r"warm|cool|saturated|desaturated|vintage|retro|futuristic|apocalyptic",
        "change_visual_style",
    ),
    (r"character.design|appearance|redesign|character look", "change_character_design"),
    (r"subtitle|caption|text.overlay", "remove_subtitle"),
    (r"speed up|faster|slow|slower|timing", "speed_up_scene"),
    (
        r"dialogue|edit.scene.text|modify.line|change.line|change.scene|edit.scene|"
        r"edit\s+the\s+scene|edit\s+scene|scene\s+dialogue|rewrite\s+line",
        "change_scene_dialogue",
    ),
    (r"script|story|rewrite|regenerate", "regenerate_script"),
    (r"scene|frame|image|picture|visual|render", "re_generate_visuals"),
    (r"video|compose|export|mp4|output", "recompose_video"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Intent classification
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_intent(query: str) -> Dict[str, Any]:
    """Fallback: keyword pattern matching."""
    q = query.lower()
    for pattern, intent in INTENT_KEYWORDS:
        if re.search(pattern, q):
            target = VALID_INTENTS.get(intent, "unknown")
            return {
                "intent": intent,
                "target": target,
                "scope": "all",
                "parameters": {"raw_query": query},
                "confidence": "rule_based",
            }
    return {
        "intent": "unknown",
        "target": "unknown",
        "scope": "all",
        "parameters": {"raw_query": query},
        "confidence": "rule_based",
    }


def _llm_intent(query: str) -> Optional[Dict[str, Any]]:
    """LangChain intent classification chain → EditIntent → dict."""
    chain = get_intent_chain(temperature=0.0)
    if chain is None:
        return None
    try:
        result = chain.invoke({"query": query})
    except Exception:
        return None
    if result is None:
        return None
    payload = result.model_dump()
    payload.setdefault("parameters", {})
    payload["parameters"].setdefault("raw_query", query)
    payload["confidence"] = "llm"
    return payload


def classify_intent(query: str) -> Dict[str, Any]:
    """Classify edit query. Tries LLM first, falls back to rule-based."""
    result = _llm_intent(query)
    if result is None:
        result = _rule_based_intent(query)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_manifest(manifest: Dict[str, Any]) -> None:
    """Persist scene_manifest.json to disk so UI/Outputs tab reflects changes."""
    path = _OUTPUTS_DIR / "scene_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


def _filter_scenes_by_scope(
    scenes: list, scope: str
) -> list:
    """Return only the scenes matching the scope, or all scenes if scope is 'all'."""
    if not scope or scope == "all":
        return scenes
    # scope format: "scene:scene_01" or "scene:2"
    if scope.startswith("scene:"):
        target = scope.split(":", 1)[1].strip().lower()
        filtered = []
        for s in scenes:
            sid = str(s.get("scene_id", "")).strip().lower()
            heading = str(s.get("heading", "")).strip().lower()
            if target in sid or target in heading:
                filtered.append(s)
        return filtered if filtered else scenes
    return scenes


def _merge_scene_tracks(
    existing: List[Dict[str, Any]], updates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Replace entries by scene_id so a partial re-run does not drop other scenes from state."""
    out: List[Dict[str, Any]] = []
    index: Dict[str, int] = {}
    for x in existing or []:
        if not isinstance(x, dict):
            continue
        sid = str(x.get("scene_id", "")).strip()
        if not sid:
            continue
        index[sid] = len(out)
        out.append(dict(x))
    for u in updates or []:
        if not isinstance(u, dict):
            continue
        sid = str(u.get("scene_id", "")).strip()
        if not sid:
            continue
        if sid in index:
            out[index[sid]] = dict(u)
        else:
            index[sid] = len(out)
            out.append(dict(u))
    return out


def _build_voice_synth_state(
    manifest: Dict[str, Any],
    target_scenes: List[Dict[str, Any]],
    characters: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "scene_manifest_data": {**manifest, "scenes": target_scenes},
        "audio_tracks": [],
        "llm_invocations": [],
        "characters": characters,
    }


def _build_video_gen_state(
    manifest: Dict[str, Any],
    target_scenes: List[Dict[str, Any]],
    characters: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
    audio_tracks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Optional *audio_tracks* lets video_gen size frame counts to WAV length (edit pipeline order)."""
    out: Dict[str, Any] = {
        "scene_manifest_data": {**manifest, "scenes": target_scenes},
        "video_tracks": [],
        "images": images,
        "characters": characters,
        "llm_invocations": [],
    }
    if audio_tracks:
        out["audio_tracks"] = list(audio_tracks)
    return out


def _finalize_scene_media(
    state: Dict[str, Any],
    updated_manifest: Dict[str, Any],
    partial_audio: List[Dict[str, Any]],
    partial_video: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge new audio/video tracks with prior state then run face-swap + lip-sync.

    Produces updated raw .mp4 files on disk and returns the combined track lists.
    """
    from agents.face_swap import face_swap_agent
    from agents.lip_sync import lip_sync_agent

    merged_audio = _merge_scene_tracks(state.get("audio_tracks") or [], partial_audio or [])
    merged_video = _merge_scene_tracks(state.get("video_tracks") or [], partial_video or [])

    sub: Dict[str, Any] = {
        "scene_manifest_data": updated_manifest,
        "scene_tasks": state.get("scene_tasks") or [],
        "images": state.get("images") or [],
        "audio_tracks": merged_audio,
        "video_tracks": merged_video,
        "face_swaps": state.get("face_swaps") or [],
        "characters": state.get("characters") or [],
        "llm_invocations": [],
    }
    sub.update(face_swap_agent(sub))
    sub.update(lip_sync_agent(sub))
    return {
        "audio_tracks": merged_audio,
        "video_tracks": merged_video,
        "face_swaps": sub.get("face_swaps") or [],
        "raw_scenes": sub.get("raw_scenes") or [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Edit executors
# ─────────────────────────────────────────────────────────────────────────────

def _execute_audio_edit(intent: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Re-synthesise voice audio for the scoped scenes, then recompose .mp4."""
    from agents.voice_synth import voice_synth_agent

    scene_manifest = state.get("scene_manifest_data") or {}
    if not isinstance(scene_manifest, dict):
        return {"audio_edit_applied": False, "error": "No scene manifest found"}

    scope = intent.get("scope", "all")
    target_scenes = _filter_scenes_by_scope(list(scene_manifest.get("scenes") or []), scope)

    audio_result = voice_synth_agent(
        _build_voice_synth_state(scene_manifest, target_scenes, state.get("characters") or [])
    )

    finalized = _finalize_scene_media(
        state, scene_manifest,
        audio_result.get("audio_tracks") or [],
        [],  # keep existing video frames
    )
    return {
        "audio_edit_applied": True,
        "regenerated_scenes": [s.get("scene_id") for s in target_scenes],
        "scene_manifest_data": scene_manifest,
        **finalized,
    }


def _extract_style_from_query(raw_query: str) -> str:
    """Convert a free-text visual edit query into a concise SD-prompt style string.

    Tries the LLM first; falls back to a simple rule-based extraction.
    """
    q = raw_query.strip()
    if not q:
        return ""

    # LangChain LCEL chain — concise style extraction
    chain = get_style_extraction_chain(temperature=0.0)
    if chain is not None:
        try:
            style = (chain.invoke({"instruction": q}) or "").strip().strip('"').strip("'")
            if style and len(style) < 120:
                return style
        except Exception:
            pass

    # Rule-based fallback — map recognisable keywords to proper SD style phrases
    q_low = q.lower()
    _STYLE_MAP = [
        ("sci.fi|scifi|futuristic",               "sci-fi futuristic, neon lights, high tech"),
        ("cyberpunk",                              "cyberpunk city, neon, rain, dark, 2049"),
        ("noir",                                   "film noir, black and white, hard shadows"),
        ("horror|scary|dark",                      "dark horror, low-key lighting, ominous shadows"),
        ("fantasy|magical",                        "epic fantasy, golden light, painterly"),
        ("western|cowboy",                         "western frontier, warm dusty tones"),
        ("apocalyptic|post.apocalyptic",           "post-apocalyptic wasteland, desaturated, gritty"),
        ("gothic",                                 "gothic atmosphere, dark arches, cold tones"),
        ("neon",                                   "neon-lit, vibrant colours, night city"),
        ("warm|golden|sunset",                     "warm golden hour, orange tones, cinematic"),
        ("cool|cold|blue",                         "cold blue tones, overcast, desaturated"),
        ("bright|vivid|vibrant",                   "bright vivid high-key lighting, saturated"),
        ("pastel|soft",                            "soft pastel tones, dreamy, gentle lighting"),
        ("vintage|retro",                          "vintage film look, grain, faded colours"),
        ("dramatic",                               "dramatic cinematic lighting, high contrast"),
        ("moody",                                  "moody atmospheric, rim-lit, dark palette"),
    ]
    for pattern, style in _STYLE_MAP:
        if re.search(pattern, q_low):
            return style

    # Last resort — use the raw query verbatim (SD handles natural language)
    return q


def _execute_video_frame_edit(intent: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Re-generate video frames for the scoped scenes with a new visual style, then recompose .mp4."""
    import copy
    from agents.video_gen import video_gen_agent

    scene_manifest = state.get("scene_manifest_data") or {}
    if not isinstance(scene_manifest, dict):
        return {"video_frame_edit_applied": False, "error": "No scene manifest found"}

    scope = intent.get("scope", "all")
    params = intent.get("parameters") or {}
    raw_q = (params.get("raw_query") or "").strip()

    # Resolve the visual style from the user's query
    new_style = _extract_style_from_query(raw_q)

    # Deep-copy scenes so manifest mutations don't affect other references
    all_scenes = [copy.deepcopy(s) if isinstance(s, dict) else s for s in (scene_manifest.get("scenes") or [])]
    target_scenes = _filter_scenes_by_scope(all_scenes, scope)

    # Stamp the resolved style onto every targeted scene
    for scene in target_scenes:
        if not isinstance(scene, dict):
            continue
        if new_style:
            scene["style"] = new_style
        elif "style" not in scene:
            scene["style"] = "cinematic"

    # Rebuild and persist the updated manifest
    updated_manifest = {**scene_manifest, "scenes": all_scenes}
    _write_manifest(updated_manifest)

    video_result = video_gen_agent(
        _build_video_gen_state(
            updated_manifest,
            target_scenes,
            state.get("characters") or [],
            state.get("images") or [],
            state.get("audio_tracks") or [],
        )
    )

    finalized = _finalize_scene_media(
        state, updated_manifest,
        [],  # keep existing audio
        video_result.get("video_tracks") or [],
    )
    return {
        "video_frame_edit_applied": True,
        "regenerated_scenes": [s.get("scene_id") for s in target_scenes],
        "scene_manifest_data": updated_manifest,
        **finalized,
    }


def _execute_video_edit(intent: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Recompose .mp4 files from existing audio + frames (no re-render)."""
    from agents.lip_sync import lip_sync_agent

    scene_manifest = state.get("scene_manifest_data") or {}
    result = lip_sync_agent({
        "scene_manifest_data": scene_manifest,
        "audio_tracks": state.get("audio_tracks") or [],
        "video_tracks": state.get("video_tracks") or [],
        "face_swaps": state.get("face_swaps") or [],
        "scene_tasks": state.get("scene_tasks") or [],
        "characters": state.get("characters") or [],
        "raw_scenes": [],
    })
    return {
        "video_edit_applied": True,
        "scene_manifest_data": scene_manifest,
        **result,
    }


def _execute_script_edit(intent: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Regenerate the script, characters, and images, then produce new audio/video/mp4."""
    from agents.scriptwriter import scriptwriter_agent
    from agents.validator import validator_agent
    from agents.character import character_agent
    from agents.image import image_agent
    from agents.scene_parser import scene_parser_agent
    from agents.voice_synth import voice_synth_agent
    from agents.video_gen import video_gen_agent

    params = intent.get("parameters") or {}
    prompt = (params.get("prompt") or params.get("raw_query") or
              "Regenerate the script with improvements.")

    s: Dict[str, Any] = {
        "input_prompt": prompt, "manual_script": "", "mode": "autonomous",
        "validated": False, "validation_report": {}, "characters": [],
        "images": [], "scene_manifest_data": {}, "scene_tasks": [],
        "audio_tracks": [], "video_tracks": [], "face_swaps": [],
        "raw_scenes": [], "llm_invocations": [], "script_repair_count": 0,
        "require_hitl": False, "approved": False, "script": "",
    }
    s.update(scriptwriter_agent(s))
    s.update(validator_agent(s))
    s.update(character_agent(s))
    s.update(image_agent(s))
    s.update(scene_parser_agent(s))

    # Persist script + characters + manifest to disk before media regen
    (_OUTPUTS_DIR / "script.txt").write_text(s.get("script") or "", encoding="utf-8")
    (_OUTPUTS_DIR / "character_db.json").write_text(
        json.dumps({"character_count": len(s.get("characters") or []), "characters": s.get("characters") or []}, indent=2),
        encoding="utf-8",
    )
    new_manifest = s.get("scene_manifest_data") or {}
    if isinstance(new_manifest, dict) and new_manifest.get("scenes"):
        _write_manifest(new_manifest)

    # Cascade: regenerate audio + frames + .mp4 for all new scenes
    scenes = (new_manifest.get("scenes") or []) if isinstance(new_manifest, dict) else []
    if scenes:
        audio_result = voice_synth_agent(
            _build_voice_synth_state(new_manifest, scenes, s.get("characters") or [])
        )
        video_result = video_gen_agent(
            _build_video_gen_state(
                new_manifest,
                scenes,
                s.get("characters") or [],
                s.get("images") or [],
                audio_result.get("audio_tracks") or [],
            )
        )
        finalized = _finalize_scene_media(
            s, new_manifest,
            audio_result.get("audio_tracks") or [],
            video_result.get("video_tracks") or [],
        )
        s.update(finalized)

    return {
        "script_edit_applied": True,
        "regenerated_scenes": [sc.get("scene_id") for sc in scenes if isinstance(sc, dict)],
        "scene_manifest_data": new_manifest,
        **{k: s[k] for k in ("characters", "images", "audio_tracks", "video_tracks", "face_swaps", "raw_scenes", "script") if k in s},
    }


def _execute_scene_edit(intent: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a free-text edit to one or more scenes, then regenerate audio + frames + .mp4."""
    from agents.voice_synth import voice_synth_agent
    from agents.video_gen import video_gen_agent

    scene_manifest = state.get("scene_manifest_data") or {}
    if not isinstance(scene_manifest, dict):
        return {"scene_edit_applied": False, "error": "No scene manifest found"}

    scope = intent.get("scope", "all")
    raw_instr = str((intent.get("parameters") or {}).get("raw_query") or "").strip()

    # ── Step 1: apply the instruction to the manifest ──────────────────────
    import copy
    # Work on a deep copy of the manifest so we can freely mutate
    working_manifest: Dict[str, Any] = copy.deepcopy(scene_manifest)

    # Identify which scenes to update
    target_ids: List[str] = [
        str(s.get("scene_id", "")).strip()
        for s in _filter_scenes_by_scope(working_manifest.get("scenes") or [], scope)
        if isinstance(s, dict) and s.get("scene_id")
    ]
    if not target_ids:
        return {"scene_edit_applied": False, "error": f"No scenes matched scope: {scope}"}

    target_scenes_before = [
        s for s in (working_manifest.get("scenes") or [])
        if isinstance(s, dict) and str(s.get("scene_id", "")).strip() in target_ids
    ]

    patched_via_llm = False
    if raw_instr:
        patched = _llm_patch_scenes_for_instruction(
            [copy.deepcopy(s) for s in target_scenes_before], raw_instr
        )
        if patched:
            _merge_scene_patch_into_manifest(working_manifest, patched)
            patched_via_llm = True

    # ── Step 2: re-extract target_scenes from the UPDATED manifest ─────────
    # Must happen AFTER the LLM patch so audio/video see the new text.
    target_scenes = [
        s for s in (working_manifest.get("scenes") or [])
        if isinstance(s, dict) and str(s.get("scene_id", "")).strip() in target_ids
    ]

    # ── Step 3: persist the updated manifest ───────────────────────────────
    _write_manifest(working_manifest)

    # ── Step 4: regenerate audio with the updated dialogue text ────────────
    audio_result = voice_synth_agent(
        _build_voice_synth_state(working_manifest, target_scenes, state.get("characters") or [])
    )

    # ── Step 5: regenerate frames for the targeted scenes ──────────────────
    video_result = video_gen_agent(
        _build_video_gen_state(
            working_manifest,
            target_scenes,
            state.get("characters") or [],
            state.get("images") or [],
            audio_result.get("audio_tracks") or [],
        )
    )

    # ── Step 6: face-swap (if enabled) + lip-sync → new .mp4 ──────────────
    finalized = _finalize_scene_media(
        state, working_manifest,
        audio_result.get("audio_tracks") or [],
        video_result.get("video_tracks") or [],
    )

    return {
        "scene_edit_applied": True,
        "dialogue_patched_via_llm": patched_via_llm,
        "regenerated_scenes": target_ids,
        "scene_manifest_data": working_manifest,
        **finalized,
    }


_EXECUTORS = {
    "audio": _execute_audio_edit,
    "video_frame": _execute_video_frame_edit,
    "video": _execute_video_edit,
    "script": _execute_script_edit,
    "scene": _execute_scene_edit,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def process_edit(
    query: str,
    current_state: Dict[str, Any],
    state_manager: Any,
    scene_id: str = "",
) -> Dict[str, Any]:
    """Classify the edit query, execute it, and save a versioned snapshot.

    Args:
        query: Free-text edit instruction from the user.
        current_state: Current pipeline state dict.
        state_manager: StateManager instance for versioning.
        scene_id: Optional scene ID to scope the edit to a specific scene.

    Returns a summary dict with keys: intent, edit_result, version.
    """
    intent = classify_intent(query)
    target = intent.get("target", "unknown")

    # Inject scene_id as scope if provided by the UI
    if scene_id:
        intent["scope"] = f"scene:{scene_id}"

    executor = _EXECUTORS.get(target)
    if executor is None:
        return {
            "intent": intent,
            "edit_result": {"status": "skipped", "reason": f"Unknown target: {target}"},
            "version": None,
        }

    try:
        hydrated = hydrate_state_for_edit(dict(current_state))
        edit_result = executor(intent, hydrated)
        merged = {**hydrated, **edit_result}
        description = f"Edit [{intent.get('intent', '?')}]: {query[:80]}"
        if scene_id:
            description += f" (scene: {scene_id})"
        version = state_manager.snapshot(merged, description=description)
        return {"intent": intent, "edit_result": edit_result, "version": version}
    except Exception as exc:
        return {
            "intent": intent,
            "edit_result": {"status": "error", "error": str(exc)},
            "version": None,
        }
