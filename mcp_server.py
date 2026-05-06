"""Model Context Protocol server — exposes the project's pipeline tools to any
MCP client (Claude Desktop, the official `mcp` CLI, custom clients, etc.).

Run via stdio transport (default for desktop MCP clients):
    python mcp_server.py

The server is a thin wrapper around tools/mcp_registry.py — every registered
tool there becomes an MCP tool here, with the same name and schema. The
in-process pipeline keeps using mcp_registry.invoke_tool() directly, so this
server is purely an external integration point.

Wire this into Claude Desktop with a config block like:

    {
      "mcpServers": {
        "agentic-film-pipeline": {
          "command": "python",
          "args": ["D:/uni-work/Agentic_AI_Project/mcp_server.py"]
        }
      }
    }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make sibling packages importable when run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from tools.mcp_registry import discover_tools, invoke_tool  # noqa: E402


mcp = FastMCP(
    "agentic-film-pipeline",
    instructions=(
        "Tools for the AI-Powered Animated Video Generation System. "
        "Use generate_script_segment to seed a screenplay, generate_image for "
        "character/scene art (Pollinations.ai by default), voice_cloning_synthesizer "
        "for TTS, render_frame_sequence for scene frames, lip_sync_aligner to "
        "compose final scene MP4s, and commit_memory to persist state."
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# One thin MCP tool per registry tool — keeps the wire format consistent.
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def generate_script_segment(prompt: str) -> str:
    """Generate an initial screenplay segment from a user prompt."""
    return str(invoke_tool("generate_script_segment", {"prompt": prompt}))


@mcp.tool()
def query_stock_footage(character_name: str) -> Dict[str, str]:
    """Retrieve a cinematic reference style for a character."""
    return invoke_tool("query_stock_footage", {"character_name": character_name})


@mcp.tool()
def generate_image(prompt: str) -> str:
    """Generate one image asset (character or scene) and return its file path.

    Backend honours the IMAGE_BACKEND env var:
      - pollinations (default): Pollinations.ai cloud API, no key needed
      - local: local Stable Diffusion / ComfyUI
    """
    return str(invoke_tool("generate_image", {"prompt": prompt}))


@mcp.tool()
def voice_cloning_synthesizer(
    scene_id: str,
    character_name: str,
    text: str,
    emotion: str = "neutral",
    voice_gender: str = "",
) -> str:
    """Generate a speech WAV aligned to a character identity. Returns the file path."""
    return str(
        invoke_tool(
            "voice_cloning_synthesizer",
            {
                "scene_id": scene_id,
                "character_name": character_name,
                "text": text,
                "emotion": emotion,
                "voice_gender": voice_gender,
            },
        )
    )


@mcp.tool()
def render_frame_sequence(
    scene_id: str,
    heading: str,
    style: str,
    frame_sequence_dir: str,
    frame_count: int,
    reference_image: str = "",
) -> Dict[str, Any]:
    """Render an animation-ready frame sequence for a scene."""
    return invoke_tool(
        "render_frame_sequence",
        {
            "scene_id": scene_id,
            "heading": heading,
            "style": style,
            "frame_sequence_dir": frame_sequence_dir,
            "frame_count": frame_count,
            "reference_image": reference_image,
        },
    )


@mcp.tool()
def lip_sync_aligner(scene_id: str, audio_path: str, frame_sequence_dir: str) -> str:
    """Synchronise audio + frames into a final scene MP4."""
    return str(
        invoke_tool(
            "lip_sync_aligner",
            {
                "scene_id": scene_id,
                "audio_path": audio_path,
                "frame_sequence_dir": frame_sequence_dir,
            },
        )
    )


@mcp.tool()
def get_task_graph(scene_manifest_json: str) -> Dict[str, Any]:
    """Build a per-scene task graph for parallel audio/video processing.

    Accepts the scene_manifest as a JSON string (MCP tool args must be primitive).
    """
    try:
        manifest = json.loads(scene_manifest_json)
    except Exception as exc:
        raise ValueError(f"scene_manifest_json must be valid JSON: {exc}") from exc
    return invoke_tool("get_task_graph", {"scene_manifest": manifest})


@mcp.tool()
def commit_memory(data_json: str) -> str:
    """Persist a structured event into long-term memory.

    Accepts the data payload as a JSON string for compatibility.
    """
    try:
        data = json.loads(data_json)
    except Exception as exc:
        raise ValueError(f"data_json must be valid JSON: {exc}") from exc
    return str(invoke_tool("commit_memory", {"data": data}))


@mcp.tool()
def list_pipeline_tools() -> List[Dict[str, Any]]:
    """Discover all available pipeline tools, their descriptions, and JSON schemas."""
    return discover_tools()


# ─────────────────────────────────────────────────────────────────────────────
# Resources — expose generated artifacts so MCP clients can read them.
# ─────────────────────────────────────────────────────────────────────────────

_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"


@mcp.resource("pipeline://script")
def get_script() -> str:
    """Latest generated screenplay (script.txt)."""
    p = _OUTPUTS_DIR / "script.txt"
    return p.read_text(encoding="utf-8") if p.exists() else "(no script yet)"


@mcp.resource("pipeline://characters")
def get_characters() -> str:
    """Latest character roster (character_db.json)."""
    p = _OUTPUTS_DIR / "character_db.json"
    return p.read_text(encoding="utf-8") if p.exists() else "{}"


@mcp.resource("pipeline://scenes")
def get_scenes() -> str:
    """Latest scene manifest (scene_manifest.json)."""
    p = _OUTPUTS_DIR / "scene_manifest.json"
    return p.read_text(encoding="utf-8") if p.exists() else "{}"


if __name__ == "__main__":
    # Default to stdio transport — works for Claude Desktop and the `mcp` CLI.
    # Override with MCP_TRANSPORT=sse for the HTTP/SSE server (uvicorn-backed).
    import os
    transport = (os.getenv("MCP_TRANSPORT") or "stdio").strip().lower()
    if transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
