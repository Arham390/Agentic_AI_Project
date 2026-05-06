"""Phase 1 + 2 + 5 — LangChain chains, Pollinations backend switch, MCP server.

These tests cover the three architectural requirements added to the project:
  * LangChain LCEL chains (structured output via Pydantic).
  * Pollinations.ai backend toggle (IMAGE_BACKEND env var).
  * FastMCP server tool registration.

No network calls are made; the Pollinations test only validates that the URL
shape and parameters are correct.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Schemas + chain builders
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemas:
    def test_character_roster_schema(self):
        from tools.schemas import Character, CharacterRoster

        roster = CharacterRoster(
            characters=[
                Character(id="char_01", name="Tom", species="cat", voice_gender="male"),
                Character(id="char_02", name="Jerry", species="mouse", voice_gender="neutral"),
            ]
        )
        assert len(roster.characters) == 2
        assert roster.characters[0].species == "cat"
        # voice_gender literal validation
        with pytest.raises(Exception):
            Character(id="x", name="X", voice_gender="alien")

    def test_edit_intent_schema(self):
        from tools.schemas import EditIntent

        ei = EditIntent(intent="change_voice_tone", target="audio", scope="character:Tom")
        assert ei.target == "audio"
        # Default scope + parameters
        ei2 = EditIntent(intent="unknown")
        assert ei2.scope == "all"
        assert ei2.parameters == {}


class TestChains:
    """Chain builders should return None when no LLM is configured."""

    def test_chains_return_none_without_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from tools.lc_chains import (
            get_character_chain, get_intent_chain, get_scene_patch_chain,
            get_script_repair_chain, get_scriptwriter_chain,
            get_style_extraction_chain, get_validator_chain,
        )
        for builder in (
            get_scriptwriter_chain, get_character_chain, get_validator_chain,
            get_script_repair_chain, get_intent_chain, get_scene_patch_chain,
            get_style_extraction_chain,
        ):
            assert builder() is None, f"{builder.__name__} should return None without keys"

    def test_intent_chain_pipes_to_pydantic(self, monkeypatch):
        """When a key IS set, the chain should be a Runnable composition."""
        monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

        from tools.lc_chains import get_intent_chain
        chain = get_intent_chain()
        # Must be a runnable with .invoke (don't actually call — would hit Groq).
        assert chain is not None
        assert hasattr(chain, "invoke")


# ─────────────────────────────────────────────────────────────────────────────
# Pollinations
# ─────────────────────────────────────────────────────────────────────────────

class TestPollinationsBackend:
    def test_url_construction(self, monkeypatch):
        from tools.pollinations_client import _sanitize_for_url, _seed_from_prompt

        # Newlines should be flattened, special chars url-encoded.
        encoded = _sanitize_for_url("a cat\nin a hat & friends")
        assert "%20" in encoded or "+" in encoded or "%26" in encoded
        assert "\n" not in encoded
        # Seed is deterministic.
        assert _seed_from_prompt("hello") == _seed_from_prompt("hello")

    def test_backend_switch_default_is_pollinations(self, monkeypatch):
        monkeypatch.delenv("IMAGE_BACKEND", raising=False)
        from tools.mcp_registry import _image_backend
        assert _image_backend() == "pollinations"

    def test_backend_switch_local(self, monkeypatch):
        monkeypatch.setenv("IMAGE_BACKEND", "local")
        from tools.mcp_registry import _image_backend
        assert _image_backend() == "local"

    def test_pollinations_configured_when_default(self, monkeypatch):
        monkeypatch.delenv("IMAGE_BACKEND", raising=False)
        from tools.pollinations_client import pollinations_configured
        assert pollinations_configured() is True

    def test_pollinations_disabled_when_local(self, monkeypatch):
        monkeypatch.setenv("IMAGE_BACKEND", "local")
        from tools.pollinations_client import pollinations_configured
        assert pollinations_configured() is False


# ─────────────────────────────────────────────────────────────────────────────
# Edit agent — deterministic patcher (no LLM required)
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterministicPatcher:
    def test_make_character_say(self):
        from agents.edit_agent import _deterministic_patch_scenes

        scenes = [
            {
                "scene_id": "scene_01",
                "dialogues": [
                    {"character": "Tom", "line": "I'll catch you, mouse."},
                    {"character": "Jerry", "line": "Never!"},
                ],
            }
        ]
        result = _deterministic_patch_scenes(
            scenes, 'make Tom say "Get back here, you furry thief"'
        )
        assert result is not None
        assert result[0]["dialogues"][0]["line"] == "Get back here, you furry thief"
        assert result[0]["dialogues"][1]["line"] == "Never!"  # Untouched

    def test_replace_substring(self):
        from agents.edit_agent import _deterministic_patch_scenes

        scenes = [
            {
                "scene_id": "scene_01",
                "dialogues": [{"character": "Tom", "line": "I love cheese."}],
            }
        ]
        result = _deterministic_patch_scenes(
            scenes, "replace 'cheese' with 'chasing mice'"
        )
        assert result is not None
        assert result[0]["dialogues"][0]["line"] == "I love chasing mice."

    def test_no_match_returns_none(self):
        from agents.edit_agent import _deterministic_patch_scenes

        scenes = [{"scene_id": "scene_01", "dialogues": [{"character": "X", "line": "y"}]}]
        result = _deterministic_patch_scenes(scenes, "make the scene darker")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# MCP server — tool registration smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestMcpServer:
    def test_server_registers_pipeline_tools(self):
        import mcp_server  # imports trigger @mcp.tool() decorators

        manager = getattr(mcp_server.mcp, "_tool_manager", None)
        assert manager is not None, "FastMCP should have _tool_manager"
        names = {t.name for t in manager.list_tools()}
        for required in (
            "generate_script_segment",
            "generate_image",
            "voice_cloning_synthesizer",
            "render_frame_sequence",
            "lip_sync_aligner",
            "list_pipeline_tools",
        ):
            assert required in names, f"MCP server missing tool: {required}"
