from typing import Dict, Optional, Tuple

from tools.llm_factory import describe_llm, get_chat_llm, llm_configured
from tools.mcp_registry import invoke_tool, offline_screenplay_from_prompt


def _try_llm_script(prompt: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    if not llm_configured():
        return None, None

    llm = get_chat_llm(temperature=0.7)
    if llm is None:
        return None, None

    meta = describe_llm(llm)
    instructions = """Convert the user's idea into a multi-scene screenplay.

Output rules (plain text only — no markdown, no **bold**, no code fences):
- Scene headings like: Scene 1 - EXT. PARK - DAY  (or start lines with INT. / EXT.)
- Action in parentheses: (Rain on the path.)
- Dialogue: CHARACTER IN ALL CAPS: what they say
  You may use JESSICA (V.O.): for voice-over; keep (V.O.) before the colon.
- Write AT LEAST 4 scenes.
- Each scene MUST have at least 3 dialogue exchanges — characters speaking back and forth.
  Do NOT write scenes with only one line of dialogue; that is too short.
- Use at least 2 named characters who interact with each other across scenes.
- Each dialogue line should be 1-3 sentences so scenes last 15-20 seconds of speech.

User idea:
"""
    try:
        response = llm.invoke(instructions + prompt)
    except Exception:
        return None, None

    content = getattr(response, "content", "")
    if isinstance(content, str) and content.strip():
        return content.strip(), meta
    return None, None


def _fallback_script(prompt: str) -> str:
    return offline_screenplay_from_prompt(prompt)


def scriptwriter_agent(state):
    prompt = state.get("input_prompt", "")
    inv = list(state.get("llm_invocations") or [])
    try:
        seed_script = str(
            invoke_tool("generate_script_segment", {"prompt": prompt})
        )
    except Exception:
        seed_script = ""

    script, llm_meta = _try_llm_script(prompt)
    if llm_meta:
        inv.append({"step": "scriptwriter", **llm_meta})
    if not script:
        script = seed_script or _fallback_script(prompt)

    try:
        invoke_tool(
            "commit_memory",
            {
                "data": {
                    "agent": "scriptwriter",
                    "prompt": prompt,
                    "script": script,
                }
            },
        )
    except Exception:
        pass

    return {"script": script, "llm_invocations": inv}