import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tools.mcp_registry import invoke_tool


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return slug[:64] or "character"


def _as_text(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(x).strip() for x in val if str(x).strip())
    return str(val).strip()


def _extract_visual_fields(char: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    """Map LLM / fallback shapes into strings for prompting."""
    name = _as_text(char.get("name")) or "Lead"
    appearance = _as_text(
        char.get("appearance")
        or char.get("physical_appearance")
        or char.get("looks")
    )
    personality = _as_text(
        char.get("personality")
        or char.get("personality_traits")
        or char.get("traits")
    )
    reference_style = _as_text(char.get("reference_style"))
    species = _as_text(char.get("species") or "human").lower() or "human"
    scenes_raw = char.get("scenes", [])
    if isinstance(scenes_raw, list):
        scene_hint = "; ".join(str(s) for s in scenes_raw[:4] if s)
    else:
        scene_hint = _as_text(scenes_raw)
    return name, appearance, personality, reference_style, scene_hint, species


def _build_cartoon_animal_prompt(
    name: str,
    species: str,
    appearance: str,
    personality: str,
    index: int,
) -> str:
    """Prompt for animated/cartoon animal characters (Tom, Jerry, etc.).

    Pollinations + Flux respond well to leading-noun anchoring: putting the
    species first, repeated, with strong "no human" negatives.
    """
    seed = int(hashlib.md5(f"{name}:{index}".encode("utf-8")).hexdigest(), 16)
    styles = [
        "classic 2D cartoon animation style",
        "hand-drawn animated cartoon character",
        "vibrant Saturday morning cartoon style",
        "expressive Warner Bros cartoon style",
    ]
    parts: List[str] = [
        # Anchor the species hard — this is the single biggest determinant.
        f"a {species}, cartoon {species}, anthropomorphic {species} character",
        f"the {species} is named {name}",
        styles[seed % len(styles)],
        "full body four-legged animal character design",
        f"distinct {species} body shape, snout, ears, fur, and tail",
        "expressive cartoon face with large eyes",
        "colorful vibrant cartoon illustration, clean line art, flat color shading",
        # Strong negatives in-prompt — Flux follows these well.
        "NOT a human, NO human face, NO person, NO human body, NO clothes on a human",
        f"the subject is an animal — a {species} — never a man, woman, or child",
    ]
    if appearance:
        parts.append(f"appearance details: {appearance}")
    if personality:
        parts.append(f"personality conveyed through pose: {personality}")
    parts.extend([
        "white background, character reference sheet",
        "no text, no watermark, no logo, no signature",
    ])
    return ", ".join(parts)


def _build_production_prompt(
    name: str,
    appearance: str,
    personality: str,
    reference_style: str,
    scene_hint: str,
    index: int,
    species: str = "human",
) -> str:
    """Build an image generation prompt appropriate for the character's species."""
    seed = int(hashlib.md5(f"{name}:{index}".encode("utf-8")).hexdigest(), 16)

    # Non-human characters get a cartoon/animation prompt — no cinematic photography terms.
    if species and species.lower() not in ("human", ""):
        return _build_cartoon_animal_prompt(name, species, appearance, personality, index)

    # Human characters: cinematic portrait (original behaviour).
    framings = [
        "three-quarter portrait, shoulders up",
        "eye-level close portrait",
        "slight low angle heroic framing",
        "contemplative profile three-quarter view",
    ]
    palettes = [
        "muted natural palette",
        "cool shadows with warm skin tones",
        "soft overcast daylight color grade",
        "golden-hour rim with soft fill",
        "desaturated cinematic teal and orange",
    ]
    lenses = [
        "85mm portrait lens compression",
        "50mm natural perspective",
        "slight anamorphic character vibe",
    ]
    parts: List[str] = [
        "single subject, highly detailed cinematic film still",
        "character reference sheet quality, consistent facial features for casting continuity",
        framings[seed % len(framings)],
        lenses[seed % len(lenses)],
        f"named character: {name}",
    ]
    if appearance:
        parts.append(f"physical appearance: {appearance}")
    if personality:
        parts.append(f"expression and body language suggesting: {personality}")
    if scene_hint:
        parts.append(f"narrative context from screenplay: {scene_hint}")
    parts.append(reference_style or "35mm film grain, soft key light, natural skin texture")
    parts.append(f"overall look: {palettes[seed % len(palettes)]}")
    parts.append(
        "professional photography, sharp eyes, no duplicate faces, no text, no watermark, no logo"
    )
    return ", ".join(parts)


def _fallback_image_file(name: str, prompt: str) -> str:
    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "image_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{_slugify(name)}.txt"
    file_path.write_text(f"Image prompt:\n{prompt}\n", encoding="utf-8")
    return str(file_path)


def image_agent(state):
    """
    One generated asset per entry in state['characters'] (same as screenplay cast list).
    If only one speaking role was extracted (e.g. Jessica alone), you get one image.
    """
    characters = state.get("characters", [])

    images = []

    for index, char in enumerate(characters, start=1):
        name, appearance, personality, reference_style, scene_hint, species = _extract_visual_fields(char)
        body = _build_production_prompt(
            name, appearance, personality, reference_style, scene_hint, index,
            species=species,
        )
        # Unique prefix so output filenames differ when prompts are long and similar
        prompt = f"[ref:{name.replace(' ', '_')}] {body}"

        try:
            image_path = invoke_tool("generate_image", {"prompt": prompt})
        except Exception:
            image_path = _fallback_image_file(name, prompt)

        images.append(
            {
                "character": name,
                "image_path": image_path,
            }
        )

    try:
        invoke_tool(
            "commit_memory",
            {
                "data": {
                    "agent": "image",
                    "images": images,
                }
            },
        )
    except Exception:
        pass

    return {"images": images}
