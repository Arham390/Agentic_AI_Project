import json
import re
from typing import Dict, List, Optional, Set, Tuple

from tools.llm_factory import describe_llm, get_chat_llm, llm_configured
from tools.mcp_registry import invoke_tool


_SCENE_RE = re.compile(r"^\s*(Scene\s+\d+|INT\.|EXT\.)", re.IGNORECASE)
_DIALOGUE_RE = re.compile(r"^\s*([A-Z][A-Z0-9_ ]{1,30})(\([^)]+\))?\s*:\s")

def extract_json(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    return "[]"


def _fallback_characters(script: str) -> List[Dict[str, object]]:
    character_scenes: Dict[str, Set[str]] = {}
    current_scene = "Scene 1"
    scene_counter = 0

    for line in script.splitlines():
        if _SCENE_RE.search(line):
            scene_counter += 1
            current_scene = f"Scene {scene_counter}"

        dialogue_match = _DIALOGUE_RE.match(line)
        if not dialogue_match:
            continue

        raw_name = dialogue_match.group(1).strip()
        if raw_name in {"INT", "EXT", "SCENE"}:
            continue

        normalized_name = raw_name.title()
        character_scenes.setdefault(normalized_name, set()).add(current_scene)

    characters: List[Dict[str, object]] = []
    for index, (name, scenes) in enumerate(sorted(character_scenes.items()), start=1):
        characters.append(
            {
                "id": f"char_{index:02d}",
                "name": name,
                "personality": "Driven and expressive",
                "appearance": "Derived from the screenplay context",
                "reference_style": "cinematic portrait",
                "voice_gender": "neutral",
                "scenes": sorted(scenes),
            }
        )

    return characters


def _try_llm_characters(script: str) -> Tuple[Optional[List[Dict[str, object]]], Optional[Dict[str, str]]]:
    if not llm_configured():
        return None, None

    llm = get_chat_llm(temperature=0)
    if llm is None:
        return None, None

    meta = describe_llm(llm)
    prompt = f"""
    You are a Character Designer Agent.

    From the script, extract all unique characters.

    For each character:
    - Assign a unique ID
    - Extract personality traits
    - Extract physical appearance
    - Mention scenes they appear in
    - voice_gender: exactly one of "male", "female", or "neutral" — which spoken voice fits this character in dialogue (infer from role and pronouns; use "neutral" only if truly ambiguous).

    Return ONLY valid JSON list.

    Script:
    {script}
    """

    try:
        response = llm.invoke(prompt)
    except Exception:
        return None, None

    content = getattr(response, "content", "")
    if not isinstance(content, str):
        return None, None

    json_text = extract_json(content)
    try:
        parsed = json.loads(json_text)
    except Exception:
        return None, None

    if isinstance(parsed, list):
        return parsed, meta
    return None, None


def character_agent(state):
    script = state.get("script", "")
    inv = list(state.get("llm_invocations") or [])
    characters, llm_meta = _try_llm_characters(script)
    if llm_meta:
        inv.append({"step": "character", **llm_meta})
    if not characters:
        characters = _fallback_characters(script)

    enriched: List[Dict[str, object]] = []
    for idx, character in enumerate(characters, start=1):
        name = str(character.get("name", f"Character {idx}"))
        updated = dict(character)
        if "id" not in updated:
            updated["id"] = f"char_{idx:02d}"

        try:
            ref = invoke_tool("query_stock_footage", {"character_name": name})
            if isinstance(ref, dict) and ref.get("reference_style"):
                updated["reference_style"] = str(ref["reference_style"])
        except Exception:
            pass

        if "physical_appearance" in updated and "appearance" not in updated:
            updated["appearance"] = updated["physical_appearance"]
        if "personality_traits" in updated and "personality" not in updated:
            updated["personality"] = updated["personality_traits"]

        vg = str(updated.get("voice_gender", "neutral")).strip().lower()
        if vg not in ("male", "female", "neutral"):
            vg = "neutral"
        updated["voice_gender"] = vg

        enriched.append(updated)

    try:
        invoke_tool(
            "commit_memory",
            {
                "data": {
                    "agent": "character",
                    "characters": enriched,
                }
            },
        )
    except Exception:
        pass

    return {"characters": enriched, "llm_invocations": inv}