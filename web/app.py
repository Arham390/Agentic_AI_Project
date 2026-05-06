"""Phase 4 — FastAPI web interface for the Agentic AI Film Pipeline."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent dir so we can import project modules.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from main import persist_outputs, run_pipeline, _load_dotenv_if_present
from tools.llm_factory import normalize_llm_env_vars
from agents.state_manager import StateManager
from agents.edit_agent import process_edit

_load_dotenv_if_present()
normalize_llm_env_vars()

app = FastAPI(title="Agentic AI Film Pipeline", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

_jobs: Dict[str, Dict[str, Any]] = {}
_state_manager = StateManager()

# ─────────────────────────────────────────────
# Background job runner
# ─────────────────────────────────────────────

def _run_job(job_id: str, prompt: str, manual_script: str = "") -> None:
    job = _jobs[job_id]
    try:
        job["progress"].append("Initialising pipeline…")
        result = run_pipeline(prompt=prompt, manual_script=manual_script, require_hitl=False)
        job["progress"].append("Pipeline finished — persisting outputs…")
        paths = persist_outputs(result)

        job["progress"].append("Saving state snapshot…")
        version = _state_manager.snapshot(result, description=f"Initial run: {prompt[:80]}")

        # Persist full pipeline state so edits can reconstruct it after restart.
        try:
            (OUTPUTS_DIR / "pipeline_state.json").write_text(
                json.dumps(result, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

        summary = {
            "script_path": str(paths["script"]),
            "character_db_path": str(paths["character_db"]),
            "scene_manifest_path": str(paths["scene_manifest"]),
            "mode": result.get("mode"),
            "validated": result.get("validated"),
            "character_count": len(result.get("characters", [])),
            "raw_scene_count": len(result.get("raw_scenes", [])),
            "audio_track_count": len(result.get("audio_tracks", [])),
            "video_track_count": len(result.get("video_tracks", [])),
            "version": version,
        }
        job.update({"status": "completed", "result": summary})
        job["progress"].append(f"Done. State saved as version {version}.")
    except Exception as exc:
        job.update({"status": "failed", "error": str(exc)})
        job["progress"].append(f"Error: {exc}")


def _run_phase_job(job_id: str, phase: str, params: Dict[str, Any]) -> None:
    job = _jobs[job_id]
    try:
        job["progress"].append(f"Re-running phase: {phase}…")

        manifest_path = OUTPUTS_DIR / "scene_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Run the full pipeline first.")

        scene_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenes = scene_manifest.get("scenes", [])
        state: Dict[str, Any] = {
            "scene_manifest_data": scene_manifest,
            "scene_tasks": [],
            "audio_tracks": [],
            "video_tracks": [],
            "face_swaps": [],
            "raw_scenes": [],
            "llm_invocations": [],
            "images": [],
            "characters": [],
            "script": "",
            "mode": "manual",
            "validated": True,
            "require_hitl": False,
            "approved": True,
            "script_repair_count": 0,
            "validation_report": {},
        }

        if phase == "voice_synth":
            from agents.voice_synth import voice_synth_agent
            result = voice_synth_agent(state)
            job["progress"].append(f"Voice synthesis done: {len(result.get('audio_tracks', []))} tracks")

        elif phase == "video_gen":
            from agents.video_gen import video_gen_agent
            result = video_gen_agent(state)
            job["progress"].append(f"Video generation done: {len(result.get('video_tracks', []))} tracks")

        elif phase == "scriptwriter":
            prompt = params.get("prompt", "Re-generate the script.")
            from agents.scriptwriter import scriptwriter_agent
            from agents.validator import validator_agent
            s2 = {**state, "input_prompt": prompt, "manual_script": "", "mode": "autonomous"}
            r1 = scriptwriter_agent(s2)
            s2.update(r1)
            result = validator_agent(s2)
            job["progress"].append("Script re-generated and validated.")
        else:
            raise ValueError(f"Unknown phase: {phase}")

        version = _state_manager.snapshot(result, description=f"Phase re-run: {phase}")
        job.update({"status": "completed", "result": result, "version": version})
        job["progress"].append(f"Phase '{phase}' complete. Version {version} saved.")
    except Exception as exc:
        job.update({"status": "failed", "error": str(exc)})
        job["progress"].append(f"Phase failed: {exc}")


def _run_edit_job(job_id: str, query: str, scene_id: str = "") -> None:
    job = _jobs[job_id]
    try:
        job["progress"].append(f"Processing edit: {query}")

        # Load full current state — try in-memory snapshot first, then disk.
        current_state: Dict[str, Any] = {}
        history = _state_manager.history()
        if history:
            latest_version = history[-1]["version"]
            restored = _state_manager.load_state(latest_version)
            if isinstance(restored, dict):
                current_state = restored
                job["progress"].append(f"Loaded state from snapshot v{latest_version}")

        # Fallback: load full saved pipeline state from disk (survives restarts).
        if not current_state:
            saved_state_path = OUTPUTS_DIR / "pipeline_state.json"
            if saved_state_path.exists():
                try:
                    current_state = json.loads(saved_state_path.read_text(encoding="utf-8"))
                    job["progress"].append("Loaded state from saved pipeline_state.json")
                except Exception as exc:
                    job["progress"].append(f"Warning: could not load pipeline_state.json: {exc}")

        # Ensure manifest is always present.
        manifest_path = OUTPUTS_DIR / "scene_manifest.json"
        if "scene_manifest_data" not in current_state and manifest_path.exists():
            current_state["scene_manifest_data"] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

        # If a specific scene_id was provided, inject it into the query scope
        if scene_id:
            job["progress"].append(f"Targeting scene: {scene_id}")

        job["progress"].append("Classifying edit intent…")
        result = process_edit(query, current_state, _state_manager, scene_id=scene_id)
        intent_name = (result.get("intent") or {}).get("intent", "unknown")
        job["progress"].append(f"Intent classified as: {intent_name}")

        edit_result = result.get("edit_result", {})
        if isinstance(edit_result, dict) and edit_result.get("status") not in ("error", "skipped"):
            merged = {**current_state, **edit_result}
            try:
                persist_outputs(merged)
                job["progress"].append("Synced outputs (script / characters / manifest) to disk")
            except Exception as exc:
                job["progress"].append(f"Warning: persist_outputs: {exc}")
            updated_manifest = edit_result.get("scene_manifest_data")
            if isinstance(updated_manifest, dict):
                manifest_path.write_text(
                    json.dumps(updated_manifest, indent=2, ensure_ascii=True),
                    encoding="utf-8",
                )
                job["progress"].append("Updated scene_manifest.json on disk")
            # Keep pipeline_state.json current so future edits have full context.
            try:
                (OUTPUTS_DIR / "pipeline_state.json").write_text(
                    json.dumps(merged, indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception:
                pass

        intent = result.get("intent", {})
        regen = edit_result.get("regenerated_scenes", [])
        regen_msg = f" Regenerated: {regen}" if regen else ""

        job.update({"status": "completed", "result": result})
        job["progress"].append(
            f"Edit applied. Intent: {intent.get('intent', '?')}{regen_msg}"
        )
    except Exception as exc:
        job.update({"status": "failed", "error": str(exc)})
        job["progress"].append(f"Edit failed: {exc}")


# ─────────────────────────────────────────────
# SSE helper
# ─────────────────────────────────────────────

async def _sse_stream(job_id: str, request: Request):
    seen = 0
    while True:
        if await request.is_disconnected():
            break
        job = _jobs.get(job_id)
        if job is None:
            yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
            break
        progress = job.get("progress", [])
        for line in progress[seen:]:
            yield f"data: {json.dumps({'type': 'progress', 'message': line})}\n\n"
        seen = len(progress)
        status_payload = {
            "type": "status",
            "status": job["status"],
            "result": job.get("result"),
            "error": job.get("error"),
        }
        yield f"data: {json.dumps(status_payload)}\n\n"
        if job["status"] in ("completed", "failed"):
            break
        await asyncio.sleep(0.8)


# ─────────────────────────────────────────────
# Endpoints — Pipeline
# ─────────────────────────────────────────────

@app.post("/api/run")
async def run_pipeline_endpoint(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    prompt = (data.get("prompt") or "").strip()
    manual_script = (data.get("manual_script") or "").strip()
    if not prompt and not manual_script:
        raise HTTPException(400, "Provide 'prompt' or 'manual_script'.")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "progress": ["Job queued…"], "result": None}
    background_tasks.add_task(_run_job, job_id, prompt, manual_script)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def job_status_sse(job_id: str, request: Request):
    return StreamingResponse(
        _sse_stream(job_id, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/job/{job_id}")
async def job_poll(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


# ─────────────────────────────────────────────
# Endpoints — Phase re-run
# ─────────────────────────────────────────────

@app.post("/api/phase/{phase}")
async def rerun_phase(phase: str, request: Request, background_tasks: BackgroundTasks):
    data = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    if not (OUTPUTS_DIR / "scene_manifest.json").exists():
        raise HTTPException(400, "Run the full pipeline first.")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "progress": [f"Queued phase re-run: {phase}"], "result": None}
    background_tasks.add_task(_run_phase_job, job_id, phase, data)
    return {"job_id": job_id}


# ─────────────────────────────────────────────
# Endpoints — Outputs
# ─────────────────────────────────────────────

@app.get("/api/outputs")
async def list_outputs():
    result: Dict[str, Any] = {}
    for subdir in ("raw_scenes", "audio_tracks", "intermediate_frames", "image_assets", "qa_metrics"):
        p = OUTPUTS_DIR / subdir
        result[subdir] = [f.name for f in sorted(p.iterdir()) if f.is_file()] if p.exists() else []
    for fname in ("script.txt", "character_db.json", "scene_manifest.json", "memory_store.json"):
        result[fname] = (OUTPUTS_DIR / fname).exists()
    return result


@app.get("/api/file/{path:path}")
async def serve_output_file(path: str):
    file_path = (OUTPUTS_DIR / path).resolve()
    try:
        file_path.relative_to(OUTPUTS_DIR.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")
    if not file_path.exists():
        raise HTTPException(404, f"Not found: {path}")
    return FileResponse(
        str(file_path),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/script")
async def get_script():
    p = OUTPUTS_DIR / "script.txt"
    if not p.exists():
        raise HTTPException(404, "Script not generated yet.")
    return {"script": p.read_text(encoding="utf-8")}


@app.get("/api/characters")
async def get_characters():
    p = OUTPUTS_DIR / "character_db.json"
    if not p.exists():
        raise HTTPException(404, "Characters not generated yet.")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/scenes")
async def get_scenes():
    p = OUTPUTS_DIR / "scene_manifest.json"
    if not p.exists():
        raise HTTPException(404, "Scene manifest not generated yet.")
    return json.loads(p.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
# Endpoints — Edit Agent (Phase 5)
# ─────────────────────────────────────────────

@app.post("/api/edit")
async def edit_endpoint(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    query = (data.get("query") or "").strip()
    scene_id = (data.get("scene_id") or "").strip()
    if not query:
        raise HTTPException(400, "Provide 'query' with the edit instruction.")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "progress": ["Edit queued…"], "result": None}
    background_tasks.add_task(_run_edit_job, job_id, query, scene_id)
    return {"job_id": job_id}


@app.get("/api/history")
async def version_history():
    return {"versions": _state_manager.history()}


@app.post("/api/revert/{version}")
async def revert_version(version: int):
    state = _state_manager.revert(version)
    if state is None:
        raise HTTPException(404, f"Version {version} not found.")
    return {"status": "reverted", "version": version, "state_keys": list(state.keys())}


@app.get("/api/version/current")
async def current_version():
    history = _state_manager.history()
    if not history:
        return {"version": 0, "description": "No snapshots yet."}
    return history[-1]


# ─────────────────────────────────────────────
# Static files & root
# ─────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Agentic AI Film Pipeline API", "docs": "/docs"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
