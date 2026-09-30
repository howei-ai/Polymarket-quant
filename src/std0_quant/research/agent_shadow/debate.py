"""One bounded bull/bear debate round over validated analyst reports."""

from __future__ import annotations

from typing import Any

from .analysts import K3_PROMPT_SCHEMA_VERSION, reports_payload
from .context import ShadowContext
from .role_instructions import role_instruction
from .schemas import ANALYST_ROLES, DEBATE_ROLES, AnalystReport


def debate_request(role: str, context: ShadowContext,
                   analysts: tuple[AnalystReport, ...]) -> dict[str, Any]:
    if role not in DEBATE_ROLES or tuple(x.role for x in analysts) != ANALYST_ROLES:
        raise ValueError("debate requires the four validated analysts in fixed order")
    return {
        "schema_version": K3_PROMPT_SCHEMA_VERSION,
        "role": role,
        "round": 1,
        "instruction": role_instruction(role),
        "context": context.to_dict(),
        "analyst_reports": reports_payload(analysts),
    }
