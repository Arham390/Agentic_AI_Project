# AgenticAI_Project — AI-Powered Animated Video Generation System

> **Agentic AI Semester Project 2026** — NUCES Islamabad  
> *From Prompt to Polished Short Film, End-to-End with LLM Agents*

---

## Project Overview

This system accepts a single natural-language prompt and autonomously produces a complete animated short video — including story, dialogue, character voices, visual scenes, and final composited output — with zero manual creative intervention.

The pipeline is divided into **five modular phases**, each a distinct AI-powered agent with defined inputs and outputs, unified by a full-stack web interface.

---

## Architecture

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Story & Script Agent (LangChain/LangGraph)    │
│  → Structured screenplay + character roster (JSON)      │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────────┐
│  Phase 2: Audio     │       │  Phase 3: Video         │
│  Edge-TTS / Coqui   │       │  SD / FFmpeg / OpenCV   │
│  → .wav + manifest  │       │  → frame sequences      │
└─────────┬───────────┘       └─────────┬───────────────┘
          └───────────────┬─────────────┘
                          ▼
                   Face Swap + Lip Sync
                          │
                          ▼
                   final_scene.mp4
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 4: Web Interface (FastAPI + HTML/JS)             │
│  → Real-time progress, phase re-runs, video preview     │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 5: Intelligent Edit Agent + Undo (LangGraph)     │
│  → Free-text edit → intent classification → targeted    │
│    phase re-run → state snapshot → undo/revert          │
└─────────────────────────────────────────────────────────┘
```

---

## Phases

| Phase | Description | Key Tools |
|-------|-------------|-----------|
| **Phase 1** | Story & Script Generation | LangChain LCEL chains, Groq/OpenAI, Pydantic schemas |
| **Phase 2** | Audio Generation & TTS | Edge-TTS, Coqui/pyttsx3, FFmpeg |
| **Phase 3** | Video Generation & Composition | Pollinations.ai (default) or Stable Diffusion (local), OpenCV, FFmpeg |
| **Phase 4** | Web Interface | FastAPI, Uvicorn, vanilla JS, SSE |
| **Phase 5** | Intelligent Edit Agent & Undo | LangGraph, LangChain intent chain, state snapshots |
| **MCP**     | External Tool Server          | FastMCP — exposes generate_image / TTS / scene rendering over stdio or SSE |

### LangChain integration

Every LLM call goes through an LCEL chain in [`tools/lc_chains.py`](tools/lc_chains.py)
with structured `PydanticOutputParser` outputs (schemas in [`tools/schemas.py`](tools/schemas.py)):

| Chain | Output Schema |
|-------|---------------|
| `get_scriptwriter_chain()`     | screenplay text |
| `get_character_chain()`        | `CharacterRoster` |
| `get_validator_chain()`        | `ScriptValidation` |
| `get_script_repair_chain()`    | repaired screenplay text |
| `get_intent_chain()`           | `EditIntent` (Phase 5) |
| `get_scene_patch_chain()`      | `ScenePatchList` (Phase 5) |
| `get_style_extraction_chain()` | style phrase |

---

## Quick Start

### 1. Clone & Install

```powershell
git clone https://github.com/Arham390/AgenticAI_Project
cd AgenticAI_Project

# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# CPU-only PyTorch (if no NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Configure Environment

Copy `.env_example` to `.env` and fill in your keys:

```dotenv
# ── LLM (LangChain) ──────────────────────────────────────────────
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# ── Image generation backend (NEW) ───────────────────────────────
# pollinations  → Pollinations.ai cloud API (default, no key needed)
# local         → Local Stable Diffusion / ComfyUI
IMAGE_BACKEND=pollinations
POLLINATIONS_MODEL=flux            # flux | turbo | sdxl

# ── Phase 2/3 toggles ────────────────────────────────────────────
PHASE2_STRICT_PRODUCTION=1   # require a real image backend (Pollinations counts)
USE_LOCAL_SD=0               # 1 = download Stable Diffusion (~4GB) for IMAGE_BACKEND=local
```

### 3a. Run via CLI

```powershell
# Full pipeline with a prompt
python main.py "A young astronaut discovers a hidden ocean on Mars"

# Manual script
python main.py --script-text "Scene 1 - EXT. MARS - DAY\nASTRONAUT: What is that?"

# With human-in-the-loop approval
python main.py --prompt "A detective story" --hitl
```

### 3b. Run the Web Interface (Phase 4)

```powershell
uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### 3c. Run the MCP Server (optional)

The pipeline tools are exposed over the [Model Context Protocol](https://modelcontextprotocol.io)
for use with Claude Desktop, the `mcp` CLI, or any other MCP client:

```powershell
# stdio transport (default — pipe in from desktop clients)
python mcp_server.py

# HTTP/SSE transport
$env:MCP_TRANSPORT = "sse"; python mcp_server.py
```

Claude Desktop config (`%APPDATA%\Claude\claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "agentic-film-pipeline": {
      "command": "python",
      "args": ["D:/uni-work/Agentic_AI_Project/mcp_server.py"]
    }
  }
}
```

---

## Web Interface Features

- **Generate tab** — Enter a prompt and watch real-time pipeline progress via Server-Sent Events
- **Outputs tab** — Browse generated script, characters, audio tracks, and scene videos
- **Edit tab (Phase 5)** — Send free-text edit commands; the agent classifies intent and re-runs the affected phase
- **Version History tab** — View all state snapshots and revert to any previous version

---

## Edit Agent (Phase 5)

The edit agent accepts natural-language instructions and maps them to targeted pipeline re-runs:

| Example Query | Detected Target | Action |
|---------------|-----------------|--------|
| "Change voice tone to dramatic" | `audio` | Re-synthesise TTS for all scenes |
| "Make the scene darker" | `video_frame` | Re-generate frames with dark style prompt |
| "Add background music" | `audio` | Re-run audio phase with music parameters |
| "Remove subtitle overlay" | `video` | Recompose video without subtitles |
| "Regenerate the script" | `script` | Re-invoke Phase 1 and cascade downstream |
| "Change character design to sci-fi" | `video_frame` | Re-generate character visuals |

**State Versioning & Undo:** Every run and edit creates a JSON snapshot. Use the Version History tab or the API to revert to any previous state.

---

## CLI Reference

```
main.py [prompt]
        [--prompt TEXT]          Prompt text (overrides positional)
        [--script-text TEXT]     Provide a full manual script
        [--script-file PATH]     Load script from UTF-8 text file
        [--hitl]                 Enable human-in-the-loop approval checkpoint
```

---

## API Reference (Phase 4)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/run` | Run full pipeline `{"prompt": "..."}` |
| `GET` | `/api/status/{job_id}` | SSE stream of job progress |
| `GET` | `/api/job/{job_id}` | Poll job status (JSON) |
| `POST` | `/api/phase/{phase}` | Re-run single phase (`voice_synth`, `video_gen`, `scriptwriter`) |
| `POST` | `/api/edit` | Apply edit `{"query": "..."}` |
| `GET` | `/api/history` | List all state snapshots |
| `POST` | `/api/revert/{version}` | Revert to a specific version |
| `GET` | `/api/outputs` | List generated output files |
| `GET` | `/api/script` | Get current script text |
| `GET` | `/api/characters` | Get character database |
| `GET` | `/api/scenes` | Get scene manifest |
| `GET` | `/api/file/{path}` | Serve an output file |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |

---

## Output Structure

```
outputs/
├── script.txt              # Generated screenplay
├── character_db.json       # Character roster with descriptions
├── scene_manifest.json     # Scene-level task manifest
├── audio_tracks/           # Per-character .wav files
├── intermediate_frames/    # Per-scene frame sequences
│   └── scene_01/
│       ├── _keyframes/     # SD/ComfyUI keyframes
│       ├── _scene_backend.txt
│       └── frame_*.png
├── raw_scenes/             # Final composited scene .mp4 files
├── image_assets/           # Character reference images
├── qa_metrics/             # Quality gate reports per scene
├── task_graph_logs/        # Parallel task graph logs
├── memory_store.json       # Agent memory commits
└── state_versions/         # Phase 5 state snapshots (for undo)
    ├── index.json
    └── v0001/
        ├── state.json
        ├── script.txt
        └── scene_manifest.json
```

---

## Running Tests

```powershell
pytest tests/ -v
```

Tests cover all five phases:
- `tests/test_phase1.py` — Scriptwriter, Validator, Character agent
- `tests/test_phase2.py` — Voice synthesis, Audio generation
- `tests/test_phase3.py` — Frame generation, Scene parser, Video gen
- `tests/test_phase4.py` — FastAPI endpoints
- `tests/test_phase5.py` — Edit agent intent classification, State manager

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| LLM / Agents | Groq API (LLaMA 3.3 70B), LangChain, LangGraph |
| TTS | Edge-TTS (cloud, free), pyttsx3 (local fallback) |
| Image Generation | Stable Diffusion (local via diffusers), built-in fallback renderer |
| Video Composition | FFmpeg, imageio, OpenCV |
| Face Swap | OpenCV Haar cascade + seamless clone |
| Web Backend | FastAPI, Uvicorn, Server-Sent Events |
| Web Frontend | Vanilla HTML/CSS/JS (no build step) |
| State Store | File-based JSON snapshots (append-only) |
| Memory | ChromaDB (optional), JSON file fallback |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key for LLaMA |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model ID |
| `OPENAI_API_KEY` | — | OpenAI key (alternative to Groq) |
| `PHASE2_STRICT_PRODUCTION` | `0` | `1` = fail-fast without GPU backends |
| `USE_LOCAL_SD` | `0` | `1` = use local Stable Diffusion |
| `SD_LOCAL_MODEL` | `stabilityai/sd-turbo` | HuggingFace model ID |
| `PHASE2_DEBUG_PLACEHOLDERS` | `0` | `1` = show debug overlays |
| `PHASE2_SCENE_RETRIES` | `2` | Retry attempts per scene |
| `PHASE2_FACE_SWAP_ENABLED` | `1` | `0` = skip face swap; use frames from `video_gen` directly |

---

## Group Members & Contributions

| Member | Phase | Responsibilities |
|--------|-------|-----------------|
| Member 1 | Phase 1 — LLM Story & Script | Prompt engineering, LangChain integration, Validator, Script repair |
| Member 2 | Phase 2 — Audio Generation | TTS integration, voice consistency, audio manifest |
| Member 3 | Phase 3 — Video Generation | Frame generation, face swap, lip sync, MP4 export |
| Member 4 | Phase 4 & 5 — Web Interface & Edit Agent | FastAPI backend, frontend UI, edit intent agent, state versioning, undo |
