"""Phase 1 — Script Repair (LangChain LCEL chain rewrites the screenplay)."""
from typing import List

from tools.lc_chains import get_script_repair_chain
from tools.llm_factory import describe_llm, get_chat_llm, llm_configured
from tools.screenplay_format import strip_markdown_fences


def script_repair_agent(state):
    inv = list(state.get("llm_invocations") or [])
    count = int(state.get("script_repair_count") or 0) + 1
    script = str(state.get("script", "") or "")
    report = state.get("validation_report") or {}

    issues: List[str] = list(report.get("issues") or [])
    suggestions: List[str] = list(report.get("suggestions") or [])
    llm_notes = report.get("llm_notes")
    if isinstance(llm_notes, dict):
        li = llm_notes.get("issues") or []
        ls = llm_notes.get("suggestions") or []
        if isinstance(li, list):
            issues.extend(str(x) for x in li)
        if isinstance(ls, list):
            suggestions.extend(str(x) for x in ls)

    print(
        f"\nScript did not pass structural validation — auto-repair attempt {count}/2 "
        "(rewriting from validator feedback)...\n"
    )

    chain = get_script_repair_chain(temperature=0.25)
    if chain is None:
        return {"script_repair_count": count, "llm_invocations": inv}

    meta = describe_llm(get_chat_llm(temperature=0.25)) if llm_configured() else None
    issue_block = "\n".join(f"- {i}" for i in issues) if issues else "- (none listed)"
    hint_block = "\n".join(f"- {s}" for s in suggestions) if suggestions else "- (none listed)"

    try:
        text = chain.invoke(
            {
                "prompt": state.get("input_prompt", ""),
                "script": script,
                "issues": issue_block,
                "suggestions": hint_block,
            }
        )
    except Exception:
        return {"script_repair_count": count, "llm_invocations": inv}

    if isinstance(text, str):
        fixed = strip_markdown_fences(text)
        if fixed:
            if meta:
                inv.append({"step": "script_repair", **meta})
            return {
                "script": fixed,
                "script_repair_count": count,
                "llm_invocations": inv,
            }

    return {"script_repair_count": count, "llm_invocations": inv}
