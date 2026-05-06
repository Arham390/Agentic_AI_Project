"""Phase 1 — Scriptwriter agent (LangChain LCEL chain)."""
from typing import Dict, Optional, Tuple

from tools.lc_chains import get_scriptwriter_chain
from tools.llm_factory import describe_llm, get_chat_llm, llm_configured
from tools.mcp_registry import invoke_tool, offline_screenplay_from_prompt


def _try_lc_script(prompt: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """LangChain LCEL chain: prompt | llm | StrOutputParser."""
    chain = get_scriptwriter_chain(temperature=0.7)
    if chain is None:
        return None, None
    # Best-effort metadata for the llm_invocations log.
    meta = describe_llm(get_chat_llm(temperature=0.7)) if llm_configured() else None
    try:
        text = chain.invoke({"idea": prompt})
    except Exception:
        return None, None
    if isinstance(text, str) and text.strip():
        return text.strip(), meta
    return None, None


def _fallback_script(prompt: str) -> str:
    return offline_screenplay_from_prompt(prompt)


def scriptwriter_agent(state):
    prompt = state.get("input_prompt", "")
    inv = list(state.get("llm_invocations") or [])
    try:
        seed_script = str(invoke_tool("generate_script_segment", {"prompt": prompt}))
    except Exception:
        seed_script = ""

    script, llm_meta = _try_lc_script(prompt)
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
