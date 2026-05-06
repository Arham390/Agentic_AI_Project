"""Pydantic schemas — single source of truth for inter-phase JSON contracts.

These are used by the LangChain chains in tools/lc_chains.py for structured
output parsing, and by the agents to validate data crossing phase boundaries.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Phase 1 — Story / Script / Characters ───────────────────────────────────

class Dialogue(BaseModel):
    character: str = Field(description="Speaking character name (Title Case).")
    line: str = Field(description="The spoken dialogue text.")
    emotion: str = Field(default="neutral", description="Emotion tag for TTS prosody.")


class Scene(BaseModel):
    scene_id: str = Field(description="Stable scene identifier, e.g. 'scene_01'.")
    heading: str = Field(description="Scene heading line, e.g. 'Scene 1 - EXT. PARK - DAY'.")
    dialogues: List[Dialogue] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list, description="Parenthesised action lines.")
    raw_lines: List[str] = Field(default_factory=list)
    style: str = Field(default="", description="Scene-level visual style override (set by edit agent).")


class Character(BaseModel):
    id: str = Field(description="Stable character id, e.g. 'char_01'.")
    name: str
    species: str = Field(
        default="human",
        description="One of: human, cat, mouse, dog, rabbit, bird, duck, bear, lion, dragon, etc.",
    )
    appearance: str = Field(default="")
    personality: str = Field(default="")
    voice_gender: Literal["male", "female", "neutral"] = "neutral"
    scenes: List[str] = Field(default_factory=list)
    reference_style: str = Field(default="")


class CharacterRoster(BaseModel):
    """LangChain-friendly wrapper so PydanticOutputParser can handle a list root."""
    characters: List[Character]


class ScriptValidation(BaseModel):
    valid: bool
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


# ── Phase 5 — Edit intent classification ────────────────────────────────────

EditTarget = Literal["audio", "video_frame", "video", "script", "scene", "unknown"]


class EditIntent(BaseModel):
    intent: str = Field(description="Specific edit intent name, e.g. 'change_voice_tone'.")
    target: EditTarget = "unknown"
    scope: str = Field(
        default="all",
        description="'all', 'character:NAME', or 'scene:SCENE_ID'.",
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ScenePatch(BaseModel):
    """A single-scene update emitted by the edit agent."""
    scene_id: str
    heading: Optional[str] = None
    dialogues: Optional[List[Dialogue]] = None
    actions: Optional[List[str]] = None
    style: Optional[str] = None


class ScenePatchList(BaseModel):
    scenes: List[ScenePatch]
