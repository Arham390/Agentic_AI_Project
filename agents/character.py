"""Phase 1 — Character Designer agent (LangChain LCEL chain)."""
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from tools.lc_chains import get_character_chain
from tools.llm_factory import describe_llm, get_chat_llm, llm_configured
from tools.mcp_registry import invoke_tool


_SCENE_RE = re.compile(r"^\s*(Scene\s+\d+|INT\.|EXT\.)", re.IGNORECASE)
_DIALOGUE_RE = re.compile(r"^\s*([A-Z][A-Z0-9_ ]{1,30})(\([^)]+\))?\s*:\s")


# Well-known animated / fictional animal characters — name → species.
_KNOWN_ANIMAL_CHARACTERS: Dict[str, str] = {
    "tom": "cat", "jerry": "mouse", "garfield": "cat", "tweety": "bird",
    "sylvester": "cat", "bugs bunny": "rabbit", "bugs": "rabbit", "daffy": "duck",
    "donald": "duck", "goofy": "dog", "pluto": "dog", "scooby": "dog",
    "lassie": "dog", "simba": "lion", "dumbo": "elephant", "bambi": "deer",
    "thumper": "rabbit", "pikachu": "electric mouse", "toothless": "dragon",
    "baloo": "bear", "mowgli": "human",  # explicitly human so it's not mis-detected
}

# Animal keywords that may appear in character names or appearance descriptions.
_ANIMAL_KEYWORDS: List[str] = [
    "cat", "kitten", "mouse", "rat", "dog", "puppy", "rabbit", "bunny",
    "bird", "duck", "goose", "bear", "lion", "tiger", "fox", "wolf",
    "horse", "cow", "pig", "sheep", "elephant", "monkey", "ape",
    "dragon", "dinosaur", "fish", "shark", "frog",
]


def _detect_species(name: str, appearance: str) -> str:
    """Return the animal species string, or '' if the character is human."""
    name_low = name.lower().strip()

    if name_low in _KNOWN_ANIMAL_CHARACTERS:
        return _KNOWN_ANIMAL_CHARACTERS[name_low]

    for known, species in _KNOWN_ANIMAL_CHARACTERS.items():
        if known in name_low:
            return species

    combined = f"{name_low} {appearance.lower()}"
    for keyword in _ANIMAL_KEYWORDS:
        if keyword in combined:
            return keyword

    return ""


def _fallback_characters(script: str) -> List[Dict[str, Any]]:
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

    characters: List[Dict[str, Any]] = []
    for index, (name, scenes) in enumerate(sorted(character_scenes.items()), start=1):
        species = _detect_species(name, "")
        characters.append(
            {
                "id": f"char_{index:02d}",
                "name": name,
                "species": species or "human",
                "personality": "Driven and expressive",
                "appearance": "Derived from the screenplay context",
                "reference_style": "cinematic portrait",
                "voice_gender": "neutral",
                "scenes": sorted(scenes),
            }
        )

    return characters


def _try_lc_characters(script: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, str]]]:
    """LangChain LCEL chain → CharacterRoster Pydantic model → list of dicts."""
    chain = get_character_chain(temperature=0.0)
    if chain is None:
        return None, None
    meta = describe_llm(get_chat_llm(temperature=0.0)) if llm_configured() else None
    try:
        roster = chain.invoke({"script": script})
    except Exception:
        return None, None
    if roster is None:
        return None, None
    try:
        return [c.model_dump() for c in roster.characters], meta
    except Exception:
        return None, None


def character_agent(state):
    script = state.get("script", "")
    inv = list(state.get("llm_invocations") or [])
    characters, llm_meta = _try_lc_characters(script)
    if llm_meta:
        inv.append({"step": "character", **llm_meta})
    if not characters:
        characters = _fallback_characters(script)

    enriched: List[Dict[str, Any]] = []
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

        # Trust the LLM's species but fall back to detection so animal characters
        # are never silently labelled "human".
        llm_species = str(updated.get("species", "")).strip().lower()
        if not llm_species or llm_species == "human":
            appearance_text = str(updated.get("appearance", "")).lower()
            detected = _detect_species(name, appearance_text)
            if detected:
                updated["species"] = detected
            elif not llm_species:
                updated["species"] = "human"
        else:
            updated["species"] = llm_species

        enriched.append(updated)

    try:
        invoke_tool(
            "commit_memory",
            {"data": {"agent": "character", "characters": enriched}},
        )
    except Exception:
        pass

    return {"characters": enriched, "llm_invocations": inv}
