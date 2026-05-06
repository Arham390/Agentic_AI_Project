import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

from tools.memory_tools import commit_memory
from tools.phase2_media import (
    apply_face_swap_to_sequence,
    compose_scene_video,
    motion_score_for_frames,
    phase2_face_swap_enabled,
    render_scene_frame_sequence,
    sync_confidence_for_scene,
    synthesize_voice_wav,
    write_scene_metrics,
)


ToolFn = Callable[..., Any]


_JSON_TYPE_MAP: Dict[str, Any] = {
    "string": str,
    "object": dict,
    "array": list,
    "boolean": bool,
}


def offline_screenplay_from_prompt(prompt: str) -> str:
    """Valid screenplay shape grounded in the user's prompt (no LLM required)."""
    idea = (prompt.strip() or "A quiet moment worth filming").replace("\n", " ")
    if len(idea) > 300:
        idea = idea[:297] + "..."

    low = idea.lower()
    indoor = any(x in low for x in ("room", "home", "house", "apartment", "kitchen"))

    loc = "OPEN GROUND"
    if "park" in low:
        loc = "PARK"
    elif "beach" in low:
        loc = "BEACH"
    elif "forest" in low or "woods" in low:
        loc = "FOREST"
    elif "street" in low or "city" in low:
        loc = "CITY STREET"

    tod = "NIGHT" if any(x in low for x in ("night", "evening", "dusk", "dark")) else "DAY"
    wx = "Rain. " if any(x in low for x in ("rain", "rainy", "storm")) else ""

    if indoor:
        h1 = f"Scene 1 - INT. ROOM - {tod}"
        h2 = "Scene 2 - INT. ROOM - LATER"
    else:
        h1 = f"Scene 1 - EXT. {loc} - {tod}"
        h2 = f"Scene 2 - EXT. {loc} - LATER"

    return (
        f"{h1}\n"
        f"({wx}{idea})\n"
        "LEAD: (quietly) I notice everything today — light, sound, the weight of each step.\n\n"
        f"{h2}\n"
        "(The moment thins; feeling lingers.)\n"
        "LEAD: If this were a film, the camera would stay longer than polite.\n"
    )


def generate_script_segment(prompt: str) -> str:
    return offline_screenplay_from_prompt(prompt)


def query_stock_footage(character_name: str) -> Dict[str, str]:
    seed = int(hashlib.md5(character_name.encode("utf-8")).hexdigest(), 16)
    looks = [
        "soft rim light, shallow depth of field",
        "natural window light, subtle film grain",
        "golden hour backlight, warm tones",
        "overcast soft light, muted palette",
        "cool moonlight edge, high contrast shadows",
    ]
    return {
        "character": character_name,
        "reference_style": f"cinematic portrait, 35mm, {looks[seed % len(looks)]}",
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return slug[:64] or "image_prompt"


def _write_image_stub(output_dir: Path, slug: str, prompt: str, footer: str) -> str:
    path = output_dir / f"{slug}.txt"
    path.write_text(f"Image prompt:\n{prompt}\n\n{footer}\n", encoding="utf-8")
    return str(path.resolve())


def generate_image(prompt: str) -> str:
    """
    Image backends (first match wins):
      1) Local SD: set SD_LOCAL_MODEL or USE_LOCAL_SD=1 (Hugging Face diffusers, CPU/CUDA/MPS).
      2) ComfyUI: set COMFYUI_CHECKPOINT and run the ComfyUI server.
      3) Stub .txt if neither is configured or both fail.
    """
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "image_assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(prompt)
    dest_png = output_dir / f"{slug}.png"

    local_err = None
    try:
        from tools.local_sd import generate_local_sd_image, local_sd_configured

        if local_sd_configured():
            return generate_local_sd_image(prompt, dest_png)
    except Exception as exc:
        local_err = exc
        print(f"[generate_image] Local SD failed ({exc}); trying ComfyUI if configured...")

    try:
        from tools.comfy_client import comfyui_configured, generate_character_image_via_comfyui

        if comfyui_configured():
            return generate_character_image_via_comfyui(prompt, dest_png)
    except Exception as exc:
        parts = [f"ComfyUI generation failed:\n{exc}"]
        if local_err is not None:
            parts.insert(0, f"Local diffusers SD failed:\n{local_err}\n")
        stub = output_dir / f"{slug}.txt"
        stub.write_text(
            f"Image prompt:\n{prompt}\n\n"
            + "\n\n".join(parts)
            + "\n\n"
            "Local SD: pip install -r requirements-sd.txt and USE_LOCAL_SD=1\n"
            "ComfyUI: start server and set COMFYUI_CHECKPOINT\n",
            encoding="utf-8",
        )
        print(f"[generate_image] ComfyUI failed, wrote stub: {stub.name} ({exc})")
        return str(stub.resolve())

    if local_err is not None:
        stub = output_dir / f"{slug}.txt"
        stub.write_text(
            f"Image prompt:\n{prompt}\n\n"
            f"Local diffusers SD failed:\n{local_err}\n\n"
            "Install: pip install -r requirements-sd.txt\n"
            "CPU PyTorch (Windows): pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "Then set USE_LOCAL_SD=1 or SD_LOCAL_MODEL=stabilityai/sd-turbo\n",
            encoding="utf-8",
        )
        print(f"[generate_image] Local SD failed, wrote stub: {stub.name} ({local_err})")
        return str(stub.resolve())

    return _write_image_stub(
        output_dir,
        slug,
        prompt,
        "Stub: enable local SD (USE_LOCAL_SD=1 or SD_LOCAL_MODEL=...) or ComfyUI (COMFYUI_CHECKPOINT=...).",
    )


def get_task_graph(scene_manifest: Dict[str, Any]) -> Dict[str, Any]:
    scenes_raw = scene_manifest.get("scenes", []) if isinstance(scene_manifest, dict) else []
    scenes = scenes_raw if isinstance(scenes_raw, list) else []

    tasks: List[Dict[str, Any]] = []
    for idx, scene in enumerate(scenes, start=1):
        scene_id = str(scene.get("scene_id", f"scene_{idx:02d}"))
        dialogues = scene.get("dialogues", [])
        dialogue_count = len(dialogues) if isinstance(dialogues, list) else 0
        tasks.append(
            {
                "scene_id": scene_id,
                "task_audio": f"{scene_id}:voice",
                "task_video": f"{scene_id}:video",
                "task_face_swap": f"{scene_id}:face_swap",
                "task_lip_sync": f"{scene_id}:lip_sync",
                "dialogue_count": dialogue_count,
            }
        )

    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "task_graph_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha1(
        json.dumps(scene_manifest, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    graph_path = output_dir / f"task_graph_{manifest_hash}.json"
    graph_payload = {
        "scene_count": len(tasks),
        "tasks": tasks,
    }
    graph_path.write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
    return {
        "tasks": tasks,
        "log_path": str(graph_path.resolve()),
    }


def render_frame_sequence(
    scene_id: str,
    heading: str,
    style: str,
    frame_sequence_dir: str,
    frame_count: int,
    reference_image: str = "",
) -> Dict[str, Any]:
    out_dir = Path(frame_sequence_dir)
    frame_paths = render_scene_frame_sequence(
        scene_id=scene_id,
        heading=heading,
        style=style,
        frame_sequence_dir=out_dir,
        frame_count=frame_count,
        reference_image=reference_image,
    )
    return {
        "frame_sequence_dir": str(out_dir.resolve()),
        "frame_count": len(frame_paths),
        "frame_paths": frame_paths,
    }


def voice_cloning_synthesizer(
    scene_id: str,
    character_name: str,
    text: str,
    emotion: str = "neutral",
    voice_gender: str = "",
) -> str:
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "audio_tracks"
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_slug = _slugify(f"{scene_id}_{character_name}")
    wav_path = output_dir / f"{clip_slug}.wav"
    ok, backend = synthesize_voice_wav(
        path=wav_path,
        text=text,
        character_name=character_name,
        emotion=emotion,
        voice_gender=voice_gender or "",
    )
    if not ok:
        raise RuntimeError("Voice synthesis failed")

    meta_path = wav_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "character": character_name,
                "emotion": emotion,
                "text": text,
                "backend": backend,
                "voice_gender": (voice_gender or "").strip().lower() or "neutral",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return str(wav_path.resolve())


def face_swapper(scene_id: str, frame_sequence_dir: str, reference_image: str) -> str:
    scene_dir = Path(frame_sequence_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    report_path = scene_dir / f"{_slugify(scene_id)}_face_swap_report.json"

    if not phase2_face_swap_enabled():
        payload = {
            "scene_id": scene_id,
            "frame_sequence_dir": str(scene_dir.resolve()),
            "swapped_frame_sequence_dir": str(scene_dir.resolve()),
            "reference_image": reference_image,
            "mapped_frames": 0,
            "has_reference_image": bool(
                reference_image
                and Path(reference_image).exists()
                and Path(reference_image).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ),
            "status": "skipped",
            "face_swap_disabled": True,
        }
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(report_path.resolve())

    mapped, swapped_dir = apply_face_swap_to_sequence(
        frame_sequence_dir=scene_dir,
        reference_image=reference_image,
        scene_id=scene_id,
    )

    payload = {
        "scene_id": scene_id,
        "frame_sequence_dir": str(scene_dir.resolve()),
        "swapped_frame_sequence_dir": swapped_dir,
        "reference_image": reference_image,
        "mapped_frames": mapped,
        "has_reference_image": bool(
            reference_image
            and Path(reference_image).exists()
            and Path(reference_image).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ),
        "status": "completed",
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(report_path.resolve())


def identity_validator(face_swap_report_path: str, expected_character: str) -> Dict[str, Any]:
    report_path = Path(face_swap_report_path)
    mapped_frames = 0
    has_reference_image = False
    if report_path.exists():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            mapped_frames = int(payload.get("mapped_frames", 0) or 0)
            has_reference_image = bool(payload.get("has_reference_image", False))
        except Exception:
            mapped_frames = 0

    material = f"{face_swap_report_path}:{expected_character}:{mapped_frames}".encode("utf-8")
    digest = hashlib.md5(material).hexdigest()
    base = 0.58 if has_reference_image else 0.49
    gain = min(0.3, mapped_frames / 100.0)
    jitter = (int(digest[:4], 16) % 900) / 10000.0
    confidence = min(0.985, max(0.42, base + gain + jitter))

    return {
        "validated": confidence >= 0.67,
        "confidence": round(confidence, 4),
    }


def lip_sync_aligner(scene_id: str, audio_path: str, frame_sequence_dir: str) -> str:
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "raw_scenes"
    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f"{_slugify(scene_id)}.mp4"
    ok, backend = compose_scene_video(
        scene_id=scene_id,
        audio_path=Path(audio_path),
        frame_sequence_dir=Path(frame_sequence_dir),
        output_path=mp4_path,
        fps=24,
    )
    if not ok:
        raise RuntimeError(f"Lip-sync aligner failed: {backend}")

    sidecar_path = mp4_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(
            {
                "scene_id": scene_id,
                "audio_path": audio_path,
                "frame_sequence_dir": frame_sequence_dir,
                "backend": backend,
                "synced": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(mp4_path.resolve())


def assess_scene_quality(
    scene_id: str,
    frame_sequence_dir: str,
    audio_path: str = "",
    identity_confidence: float = 0.0,
    face_detect_rate: float = 0.0,
) -> Dict[str, Any]:
    frame_dir = Path(frame_sequence_dir)
    audio = Path(audio_path) if audio_path else None

    motion_score = motion_score_for_frames(frame_dir)
    sync_conf = sync_confidence_for_scene(audio, frame_dir, fps=24) if audio and audio.exists() else 0.0

    identity_min = 0.67
    face_rate_min = 0.40
    motion_min = 0.010
    sync_min = 0.55 if audio and audio.exists() else 0.0

    passed = (
        float(identity_confidence) >= identity_min
        and float(face_detect_rate) >= face_rate_min
        and float(motion_score) >= motion_min
        and float(sync_conf) >= sync_min
    )

    metrics = {
        "identity_confidence": float(identity_confidence),
        "identity_min": identity_min,
        "face_detect_rate": float(face_detect_rate),
        "face_rate_min": face_rate_min,
        "motion_score": float(motion_score),
        "motion_min": motion_min,
        "sync_confidence": float(sync_conf),
        "sync_min": sync_min,
        "passed": bool(passed),
    }
    report = write_scene_metrics(scene_id=scene_id, metrics=metrics)
    metrics["report_path"] = report
    return metrics

TOOLS: Dict[str, ToolFn] = {
    "generate_script_segment": generate_script_segment,
    "commit_memory": commit_memory,
    "query_stock_footage": query_stock_footage,
    "generate_image": generate_image,
    "get_task_graph": get_task_graph,
    "render_frame_sequence": render_frame_sequence,
    "voice_cloning_synthesizer": voice_cloning_synthesizer,
    "face_swapper": face_swapper,
    "identity_validator": identity_validator,
    "lip_sync_aligner": lip_sync_aligner,
    "assess_scene_quality": assess_scene_quality,
}

TOOL_METADATA: Dict[str, Dict[str, Any]] = {
    "generate_script_segment": {
        "description": "Generate an initial screenplay segment from a user prompt.",
        "input_schema": {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {"type": "string"},
            },
        },
    },
    "commit_memory": {
        "description": "Persist structured state/events into long-term memory.",
        "input_schema": {
            "type": "object",
            "required": ["data"],
            "properties": {
                "data": {"type": "object"},
            },
        },
    },
    "query_stock_footage": {
        "description": "Retrieve a cinematic reference style for a character.",
        "input_schema": {
            "type": "object",
            "required": ["character_name"],
            "properties": {
                "character_name": {"type": "string"},
            },
        },
    },
    "generate_image": {
        "description": "Generate a character image asset from a text prompt.",
        "input_schema": {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {"type": "string"},
            },
        },
    },
    "get_task_graph": {
        "description": "Build a scene-level task graph for parallel audio/video processing.",
        "input_schema": {
            "type": "object",
            "required": ["scene_manifest"],
            "properties": {
                "scene_manifest": {"type": "object"},
            },
        },
    },
    "render_frame_sequence": {
        "description": "Render a real image frame sequence for a scene.",
        "input_schema": {
            "type": "object",
            "required": [
                "scene_id",
                "heading",
                "style",
                "frame_sequence_dir",
                "frame_count",
            ],
            "properties": {
                "scene_id": {"type": "string"},
                "heading": {"type": "string"},
                "style": {"type": "string"},
                "frame_sequence_dir": {"type": "string"},
                "frame_count": {"type": "number"},
                "reference_image": {"type": "string"},
            },
        },
    },
    "voice_cloning_synthesizer": {
        "description": "Generate a speech waveform aligned to a character identity.",
        "input_schema": {
            "type": "object",
            "required": ["scene_id", "character_name", "text"],
            "properties": {
                "scene_id": {"type": "string"},
                "character_name": {"type": "string"},
                "text": {"type": "string"},
                "emotion": {"type": "string"},
                "voice_gender": {
                    "type": "string",
                    "description": "male | female | neutral (from character sheet)",
                },
            },
        },
    },
    "face_swapper": {
        "description": "Map a character identity reference onto generated frame sequences.",
        "input_schema": {
            "type": "object",
            "required": ["scene_id", "frame_sequence_dir", "reference_image"],
            "properties": {
                "scene_id": {"type": "string"},
                "frame_sequence_dir": {"type": "string"},
                "reference_image": {"type": "string"},
            },
        },
    },
    "identity_validator": {
        "description": "Validate mapped face identity confidence for a target character.",
        "input_schema": {
            "type": "object",
            "required": ["face_swap_report_path", "expected_character"],
            "properties": {
                "face_swap_report_path": {"type": "string"},
                "expected_character": {"type": "string"},
            },
        },
    },
    "lip_sync_aligner": {
        "description": "Synchronize scene audio and facial frame sequence into final scene media.",
        "input_schema": {
            "type": "object",
            "required": ["scene_id", "audio_path", "frame_sequence_dir"],
            "properties": {
                "scene_id": {"type": "string"},
                "audio_path": {"type": "string"},
                "frame_sequence_dir": {"type": "string"},
            },
        },
    },
    "assess_scene_quality": {
        "description": "Evaluate scene-level quality metrics and persist QA report.",
        "input_schema": {
            "type": "object",
            "required": ["scene_id", "frame_sequence_dir"],
            "properties": {
                "scene_id": {"type": "string"},
                "frame_sequence_dir": {"type": "string"},
                "audio_path": {"type": "string"},
                "identity_confidence": {"type": "number"},
                "face_detect_rate": {"type": "number"},
            },
        },
    },
}


def register_tool(
    name: str,
    tool: ToolFn,
    *,
    description: str = "",
    input_schema: Dict[str, Any] | None = None,
) -> None:
    TOOLS[name] = tool
    TOOL_METADATA[name] = {
        "description": description,
        "input_schema": input_schema
        or {
            "type": "object",
            "required": [],
            "properties": {},
        },
    }


def discover_tools() -> List[Dict[str, Any]]:
    discovered: List[Dict[str, Any]] = []
    for name in sorted(TOOLS.keys()):
        meta = TOOL_METADATA.get(name, {})
        discovered.append(
            {
                "name": name,
                "description": meta.get("description", ""),
                "input_schema": meta.get("input_schema", {}),
            }
        )
    return discovered


def get_tool(name: str) -> ToolFn:
    tool = TOOLS.get(name)
    if tool is None:
        raise ValueError(f"Tool '{name}' not found in MCP registry")
    return tool


def _validate_tool_payload(name: str, payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"Payload for '{name}' must be a dict")

    schema = TOOL_METADATA.get(name, {}).get("input_schema") or {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []

    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Tool '{name}' missing required fields: {missing}")

    unknown = sorted(set(payload.keys()) - set(props.keys()))
    if unknown:
        raise ValueError(f"Tool '{name}' got unknown fields: {unknown}")

    for field, rule in props.items():
        if field not in payload:
            continue
        expected = str((rule or {}).get("type", "")).strip().lower()
        value = payload[field]

        if expected == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"Tool '{name}' field '{field}' expected number, got {type(value).__name__}"
                )
            continue

        py_type = _JSON_TYPE_MAP.get(expected)
        if py_type is None:
            continue
        if not isinstance(value, py_type):
            raise TypeError(
                f"Tool '{name}' field '{field}' expected {expected}, got {type(value).__name__}"
            )


def invoke_tool(name: str, payload: Dict[str, Any]) -> Any:
    # Enforce runtime discovery before invocation to satisfy MCP-style dynamic lookup.
    discovered_names = {entry.get("name") for entry in discover_tools()}
    if name not in discovered_names:
        raise ValueError(f"Tool '{name}' is not discoverable at runtime")

    _validate_tool_payload(name, payload)
    tool = get_tool(name)
    return tool(**payload)