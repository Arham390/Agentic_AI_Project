"""LangChain LCEL chains — the structured-output entry points used by every
agent that talks to an LLM.

Each public function returns a callable chain. Callers do `chain.invoke({...})`
and receive a Pydantic model (or string).  The chain is None when no LLM is
configured — agents must check and fall back to deterministic logic.

Why LCEL: the spec calls for "structured output techniques (JSON schema
enforcement, prompt chaining)".  LCEL pipes (`prompt | llm | parser`) are the
canonical way to express that with retry/fallback semantics.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from tools.llm_factory import get_chat_llm, llm_configured
from tools.schemas import (
    CharacterRoster,
    EditIntent,
    ScenePatchList,
    ScriptValidation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json_blob(text: str, prefer: str = "object") -> str:
    """Pull the first balanced {...} or [...] from a model response.

    Falls back to ``"{}"``/``"[]"`` so downstream parsers always see valid JSON.
    """
    if not isinstance(text, str):
        return "{}" if prefer == "object" else "[]"

    # Strip code fences first.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)

    if prefer == "array":
        m = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if m:
            return m.group(0)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        return m.group(0)
    return "{}" if prefer == "object" else "[]"


def _safe_pydantic_parse(parser: PydanticOutputParser, raw: str) -> Optional[Any]:
    """Parser with one fallback attempt — extract JSON blob and retry."""
    try:
        return parser.parse(raw)
    except Exception:
        pass
    blob = _extract_json_blob(raw, prefer="object")
    try:
        return parser.parse(blob)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Scriptwriter (string output)
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPTWRITER_SYSTEM = """You are a screenwriter agent. Convert a user idea into a
multi-scene screenplay.

Output rules (plain text only — no markdown, no **bold**, no code fences):
- Scene headings like: Scene 1 - EXT. PARK - DAY (or start lines with INT. / EXT.)
- Action in parentheses: (Tom chases Jerry across the aisle.)
- Dialogue: CHARACTER IN ALL CAPS: what they say
  You may use JESSICA (V.O.): for voice-over; keep (V.O.) before the colon.
- Write AT LEAST 4 scenes, each with at least 3 dialogue exchanges.
- Use at least 2 named characters who interact across scenes.
- Each dialogue line should be 1-3 sentences (15-20 sec of speech per scene).

CRITICAL — Animal and non-human characters:
- If the story has animal characters (cats, mice, dogs, birds, etc.), write them
  AS THEIR ACTUAL ANIMAL SPECIES — never as humans.
- Tom is a CAT. Jerry is a MOUSE. They may speak (cartoon style) but actions must
  reflect their nature (Tom pounces, Jerry scurries).
- In every action line that introduces an animal character, mention the species:
  (Tom the orange tabby cat leaps over a suitcase.)
"""


def get_scriptwriter_chain(temperature: float = 0.7) -> Optional[Runnable]:
    """Returns an LCEL chain: dict[idea: str] -> screenplay text. None if no LLM."""
    if not llm_configured():
        return None
    llm = get_chat_llm(temperature=temperature)
    if llm is None:
        return None

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SCRIPTWRITER_SYSTEM),
            ("human", "User idea:\n{idea}"),
        ]
    )
    return prompt | llm | StrOutputParser()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Character roster (Pydantic output)
# ─────────────────────────────────────────────────────────────────────────────

def get_character_chain(temperature: float = 0.0) -> Optional[Runnable]:
    """Returns an LCEL chain: dict[script: str] -> CharacterRoster. None if no LLM."""
    if not llm_configured():
        return None
    llm = get_chat_llm(temperature=temperature)
    if llm is None:
        return None

    parser = PydanticOutputParser(pydantic_object=CharacterRoster)

    instructions = """You are a Character Designer Agent.

Read the screenplay and emit a structured roster covering EVERY speaking role.

Field rules:
- "id": unique string like "char_01", "char_02".
- "name": as written in the script.
- "species": MOST IMPORTANT — exactly one of human, cat, mouse, dog, rabbit, bird,
  duck, bear, lion, dragon, or another concrete animal/creature name.
  A cartoon cat named Tom = "cat". A mouse named Jerry = "mouse". A pilot named
  Owais = "human". Do NOT default everyone to "human" — match the screenplay.
- "appearance": physical description matching the species (fur color, size, breed,
  for animals; ethnicity, build, hair, clothes for humans).
- "personality": personality traits.
- "voice_gender": exactly one of male / female / neutral.
- "scenes": list of scene IDs they appear in (e.g. ["scene_01", "scene_02"]).

{format_instructions}

Screenplay:
{script}
"""
    prompt = ChatPromptTemplate.from_template(instructions).partial(
        format_instructions=parser.get_format_instructions(),
    )

    def _parse(raw: str) -> Optional[CharacterRoster]:
        return _safe_pydantic_parse(parser, raw)

    return prompt | llm | StrOutputParser() | RunnableLambda(_parse)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Script validator (advisory)
# ─────────────────────────────────────────────────────────────────────────────

def get_validator_chain(temperature: float = 0.0) -> Optional[Runnable]:
    if not llm_configured():
        return None
    llm = get_chat_llm(temperature=temperature)
    if llm is None:
        return None

    parser = PydanticOutputParser(pydantic_object=ScriptValidation)
    instructions = """You are a Script Validator Agent (advisory only — another
layer checks structure mechanically). Be lenient: minor stylistic choices are fine.

Set "valid" to true unless there are clear structural problems
(no scenes, no dialogue, etc.).

{format_instructions}

Script:
{script}
"""
    prompt = ChatPromptTemplate.from_template(instructions).partial(
        format_instructions=parser.get_format_instructions(),
    )
    return prompt | llm | StrOutputParser() | RunnableLambda(
        lambda raw: _safe_pydantic_parse(parser, raw)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Script repair (string output)
# ─────────────────────────────────────────────────────────────────────────────

def get_script_repair_chain(temperature: float = 0.25) -> Optional[Runnable]:
    if not llm_configured():
        return None
    llm = get_chat_llm(temperature=temperature)
    if llm is None:
        return None

    template = """Rewrite the screenplay below so it passes a strict structural checker.

Required format (plain text only — no markdown, no **bold**, no ``` fences):
1) Scene headings: e.g. "Scene 1 - EXT. PARK - DAY" or lines starting with INT./EXT.
2) Action lines in parentheses, e.g. (Rain falls on empty paths.)
3) Dialogue lines: CHARACTER IN ALL CAPS: spoken text
   Optional parenthetical before colon: JESSICA (V.O.): whispered line
4) At least two scenes and multiple dialogue lines when it fits the story.

Original creative brief (keep the same story and tone):
{prompt}

Current script:
---
{script}
---

Reported problems:
{issues}

Suggested fixes:
{suggestions}

Return ONLY the full revised screenplay. No title line, no commentary."""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Edit intent classification
# ─────────────────────────────────────────────────────────────────────────────

VALID_INTENTS = (
    "change_voice_tone",
    "add_background_music",
    "change_voice_speed",
    "re_synthesize_audio",
    "make_scene_darker",
    "make_scene_brighter",
    "change_character_design",
    "change_visual_style",
    "re_generate_visuals",
    "remove_subtitle",
    "speed_up_scene",
    "slow_down_scene",
    "recompose_video",
    "regenerate_script",
    "add_scene",
    "change_scene_dialogue",
    "unknown",
)


def get_intent_chain(temperature: float = 0.0) -> Optional[Runnable]:
    if not llm_configured():
        return None
    llm = get_chat_llm(temperature=temperature)
    if llm is None:
        return None

    parser = PydanticOutputParser(pydantic_object=EditIntent)
    valid_intents_str = ", ".join(VALID_INTENTS)
    template = """You are an intent-classification agent for a video editing pipeline.

Classify the user's edit query into a structured JSON object.

Valid intents: {intents}
Valid targets: audio | video_frame | video | script | scene | unknown
Valid scope:   "all" | "character:NAME" | "scene:scene_01"

{format_instructions}

User query: {query}
"""
    prompt = ChatPromptTemplate.from_template(template).partial(
        intents=valid_intents_str,
        format_instructions=parser.get_format_instructions(),
    )
    return prompt | llm | StrOutputParser() | RunnableLambda(
        lambda raw: _safe_pydantic_parse(parser, raw)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Scene patcher (used by edit agent for dialogue rewrites)
# ─────────────────────────────────────────────────────────────────────────────

def get_scene_patch_chain(temperature: float = 0.2) -> Optional[Runnable]:
    if not llm_configured():
        return None
    llm = get_chat_llm(temperature=temperature)
    if llm is None:
        return None

    parser = PydanticOutputParser(pydantic_object=ScenePatchList)
    template = """You are editing a film pipeline scene manifest. Apply the user's
instruction to ONLY the scenes provided. Each output object MUST keep the same
scene_id as its input. You may change heading, dialogues, actions, and style.

{format_instructions}

Instruction:
{instruction}

Scenes JSON:
{scenes}
"""
    prompt = ChatPromptTemplate.from_template(template).partial(
        format_instructions=parser.get_format_instructions(),
    )
    return prompt | llm | StrOutputParser() | RunnableLambda(
        lambda raw: _safe_pydantic_parse(parser, raw)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Style extractor (string)
# ─────────────────────────────────────────────────────────────────────────────

def get_style_extraction_chain(temperature: float = 0.0) -> Optional[Runnable]:
    if not llm_configured():
        return None
    llm = get_chat_llm(temperature=temperature)
    if llm is None:
        return None

    template = """Extract a concise visual style description (max 12 words) for a
Stable Diffusion / Pollinations prompt from the following video edit instruction.
Output ONLY the style phrase, no explanations, no quotes.

Instruction: {instruction}
"""
    return ChatPromptTemplate.from_template(template) | llm | StrOutputParser()
