"""Phase 1 — Script Validator (rule-based + LangChain advisory)."""
from typing import Dict, Optional, Tuple

from tools.lc_chains import get_validator_chain
from tools.llm_factory import describe_llm, get_chat_llm, llm_configured
from tools.mcp_registry import invoke_tool
from tools.screenplay_format import rule_validate_structure


def _try_lc_validation(script: str) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, str]]]:
    chain = get_validator_chain(temperature=0.0)
    if chain is None:
        return None, None
    meta = describe_llm(get_chat_llm(temperature=0.0)) if llm_configured() else None
    try:
        result = chain.invoke({"script": script})
    except Exception:
        return None, None
    if result is None:
        return None, None
    return (
        {
            "valid": bool(result.valid),
            "issues": list(result.issues),
            "suggestions": list(result.suggestions),
        },
        meta,
    )


def validator_agent(state):
    script = state.get("script", "")
    inv = list(state.get("llm_invocations") or [])
    rule_result = rule_validate_structure(script)
    llm_result, llm_meta = _try_lc_validation(script)
    if llm_meta:
        inv.append({"step": "validator", **llm_meta})

    if llm_result is None:
        report = {
            "source": "rule-only",
            "valid": bool(rule_result.get("valid", False)),
            "issues": list(rule_result.get("issues", [])),
            "suggestions": list(rule_result.get("suggestions", [])),
        }
    else:
        report = {
            "source": "rule+llm_advisory",
            "valid": bool(rule_result.get("valid", False)),
            "issues": list(rule_result.get("issues", [])),
            "suggestions": list(rule_result.get("suggestions", [])),
            "llm_notes": {
                "agrees": bool(llm_result.get("valid", False)),
                "issues": llm_result.get("issues", []),
                "suggestions": llm_result.get("suggestions", []),
            },
        }

    result = {
        "validated": bool(report["valid"]),
        "validation_report": report,
    }

    try:
        invoke_tool(
            "commit_memory",
            {"data": {"agent": "validator", "validated": result["validated"], "report": report}},
        )
    except Exception:
        pass

    result["llm_invocations"] = inv
    return result
