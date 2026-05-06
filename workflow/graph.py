import sys
from typing import Any, Dict

try:
    from langgraph.graph import END, StateGraph
    try:
        from langgraph.constants import Send
    except Exception:
        Send = None
except Exception:
    END = None
    StateGraph = None
    Send = None

from state import AgentState

from agents.scriptwriter import scriptwriter_agent
from agents.validator import validator_agent
from agents.script_repair import script_repair_agent
from agents.character import character_agent
from agents.image import image_agent
from agents.scene_parser import scene_parser_agent
from agents.voice_synth import voice_synth_agent
from agents.video_gen import video_gen_agent
from agents.face_swap import face_swap_agent
from agents.lip_sync import lip_sync_agent
from tools.llm_factory import llm_configured
from tools.mcp_registry import invoke_tool

MAX_SCRIPT_REPAIRS = 2


def mode_selector_agent(state):
    manual_script = str(state.get("manual_script", "") or "").strip()
    if manual_script:
        return {
            "mode": "manual",
            "script": manual_script,
        }
    return {"mode": "autonomous"}


def validation_reject_agent(state):
    print("\nValidation failed. Pipeline stopped (character and image steps skipped).")
    print("Report:")
    print(state.get("validation_report"))
    print("Fix the script structure, or re-run with --hitl to review and optionally override.\n")
    return {"approved": False, "stop_reason": "validation"}


def hitl_agent(state):
    validated = bool(state.get("validated", False))
    require_hitl = bool(state.get("require_hitl", False))
    report = state.get("validation_report")

    if not validated:
        print("\nScript failed validation:")
        print(report)
        if not require_hitl:
            return {"approved": False, "stop_reason": "validation"}
        if not sys.stdin.isatty():
            print(
                "No interactive terminal: cannot override a failed validation. "
                "Use a TTY or fix the script.\n"
            )
            return {"approved": False, "stop_reason": "validation"}
        answer = input("Validation failed. Override and continue to character/images? [y/N]: ").strip().lower()
        approved = answer in {"y", "yes"}
        return {"approved": approved} if approved else {"approved": False, "stop_reason": "hitl"}

    print("\nCheckpoint: script passed validation.")
    print(report)

    if not require_hitl:
        return {"approved": True}

    if not sys.stdin.isatty():
        print(
            "No interactive terminal: --hitl requires a TTY for approval. "
            "Re-run in a terminal or omit --hitl to auto-continue after validation.\n"
        )
        return {"approved": False, "stop_reason": "hitl"}

    answer = input("Approve continuation to character and image generation? [y/N]: ").strip().lower()
    approved = answer in {"y", "yes"}
    return {"approved": approved} if approved else {"approved": False, "stop_reason": "hitl"}


def memory_commit_agent(state):
    validated = bool(state.get("validated", False))
    require_hitl = bool(state.get("require_hitl", False))
    user_approved = bool(state.get("approved", False))
    skipped_hitl_after_success = validated and not require_hitl
    effective_approved = user_approved or skipped_hitl_after_success

    payload = {
        "agent": "memory_commit",
        "mode": state.get("mode"),
        "validated": validated,
        "approved": effective_approved,
        "stop_reason": state.get("stop_reason") or None,
        "character_count": len(state.get("characters", [])),
        "image_count": len(state.get("images", [])),
        "scene_count": len((state.get("scene_manifest_data") or {}).get("scenes", [])),
        "audio_count": len(state.get("audio_tracks", [])),
        "video_count": len(state.get("video_tracks", [])),
        "face_swap_count": len(state.get("face_swaps", [])),
        "raw_scene_count": len(state.get("raw_scenes", [])),
        "validation_report": state.get("validation_report", {}),
        "llm_invocations": state.get("llm_invocations") or [],
    }

    try:
        status = str(invoke_tool("commit_memory", {"data": payload}))
    except Exception as exc:
        status = f"error:{exc}"

    return {"memory_commit_status": status}


def route_after_mode(state):
    return "validator" if state.get("mode") == "manual" else "scriptwriter"


def _can_script_repair(state):
    if state.get("mode") != "autonomous":
        return False
    if not llm_configured():
        return False
    if int(state.get("script_repair_count") or 0) >= MAX_SCRIPT_REPAIRS:
        return False
    return True


def route_after_validation(state):
    validated = bool(state.get("validated", False))
    require_hitl = bool(state.get("require_hitl", False))
    if validated and not require_hitl:
        return "character"
    if validated and require_hitl:
        return "hitl"
    if _can_script_repair(state):
        return "script_repair"
    if require_hitl:
        return "hitl"
    return "validation_reject"


def route_after_hitl(state):
    if state.get("approved"):
        return "character"
    return "memory_commit"


def route_phase2_dispatch(state):
    scenes = (state.get("scene_manifest_data") or {}).get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        return ["voice_synth", "video_gen"]

    if Send is None:
        return ["voice_synth", "video_gen"]

    sends = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        shared = {
            "scene_payload": scene,
            "characters": state.get("characters") or [],
            "images": state.get("images") or [],
            "scene_manifest_data": state.get("scene_manifest_data") or {},
        }
        sends.append(Send("voice_synth", shared))
        sends.append(Send("video_gen", shared))
    return sends or ["voice_synth", "video_gen"]


class _FallbackCompiledGraph:
    def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        state = dict(initial_state)
        state.update(mode_selector_agent(state))
        if route_after_mode(state) == "scriptwriter":
            state.update(scriptwriter_agent(state))
        while True:
            state.update(validator_agent(state))
            nxt = route_after_validation(state)
            if nxt == "script_repair":
                state.update(script_repair_agent(state))
                continue
            break

        if nxt == "validation_reject":
            state.update(validation_reject_agent(state))
            state.update(memory_commit_agent(state))
            return state
        if nxt == "hitl":
            state.update(hitl_agent(state))
            if not state.get("approved"):
                state.update(memory_commit_agent(state))
                return state

        state.update(character_agent(state))
        state.update(image_agent(state))
        state.update(scene_parser_agent(state))
        state.update(voice_synth_agent(state))
        state.update(video_gen_agent(state))
        state.update(face_swap_agent(state))
        state.update(lip_sync_agent(state))
        state.update(memory_commit_agent(state))
        return state


def build_graph():
    if StateGraph is None:
        return _FallbackCompiledGraph()

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("mode_selector", mode_selector_agent)
    builder.add_node("scriptwriter", scriptwriter_agent)
    builder.add_node("validator", validator_agent)
    builder.add_node("script_repair", script_repair_agent)
    builder.add_node("character", character_agent)
    builder.add_node("image", image_agent)
    builder.add_node("scene_parser", scene_parser_agent)
    builder.add_node("voice_synth", voice_synth_agent)
    builder.add_node("video_gen", video_gen_agent)
    builder.add_node("face_swap", face_swap_agent)
    builder.add_node("lip_sync", lip_sync_agent)
    builder.add_node("hitl", hitl_agent)
    builder.add_node("validation_reject", validation_reject_agent)
    builder.add_node("memory_commit", memory_commit_agent)

    # Entry
    builder.set_entry_point("mode_selector")

    builder.add_conditional_edges(
        "mode_selector",
        route_after_mode,
        {
            "validator": "validator",
            "scriptwriter": "scriptwriter",
        },
    )

    # Flow
    builder.add_edge("scriptwriter", "validator")

    builder.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "character": "character",
            "hitl": "hitl",
            "validation_reject": "validation_reject",
            "script_repair": "script_repair",
        },
    )

    builder.add_edge("script_repair", "validator")

    builder.add_edge("validation_reject", "memory_commit")

    builder.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {
            "character": "character",
            "memory_commit": "memory_commit",
        },
    )

    builder.add_edge("character", "image")
    builder.add_edge("image", "scene_parser")

    builder.add_conditional_edges(
        "scene_parser",
        route_phase2_dispatch,
        ["voice_synth", "video_gen"],
    )

    builder.add_edge("voice_synth", "face_swap")
    builder.add_edge("video_gen", "face_swap")
    builder.add_edge("face_swap", "lip_sync")
    builder.add_edge("lip_sync", "memory_commit")
    builder.add_edge("memory_commit", END)

    return builder.compile()