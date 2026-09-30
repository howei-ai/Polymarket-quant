"""Fixed one-pass manager synthesis; no agent self-loop."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .analysts import K3_PROMPT_SCHEMA_VERSION, reports_payload
from .context import ShadowContext
from .role_instructions import role_instruction
from .schemas import ANALYST_ROLES, DEBATE_ROLES, AnalystReport, ManagerReport


def manager_request(context: ShadowContext, analysts: tuple[AnalystReport, ...],
                    bull: AnalystReport, bear: AnalystReport) -> dict[str, Any]:
    if tuple(x.role for x in analysts) != ANALYST_ROLES or (bull.role, bear.role) != DEBATE_ROLES:
        raise ValueError("manager requires all validated fixed-role reports")
    return {
        "schema_version": K3_PROMPT_SCHEMA_VERSION,
        "role": "RESEARCH_MANAGER",
        "round": 1,
        "instruction": role_instruction("RESEARCH_MANAGER"),
        "context": context.to_dict(),
        "analyst_reports": reports_payload(analysts),
        "bull_report": asdict(bull),
        "bear_report": asdict(bear),
    }


def validate_manager_synthesis(manager: ManagerReport, reports: tuple[AnalystReport, ...]) -> None:
    """Bind claimed support and opposition to the actual typed role reports."""
    by_role = {report.role: report for report in reports}
    for role in manager.supporting_roles + manager.contradicting_roles:
        if role not in by_role or by_role[role].abstain:
            raise ValueError("manager cited missing or abstaining role")
    if manager.direction in ("UP", "DOWN"):
        if any(by_role[role].direction != manager.direction for role in manager.supporting_roles):
            raise ValueError("manager support contradicts actual report")
        if any(by_role[role].direction == manager.direction for role in manager.contradicting_roles):
            raise ValueError("manager contradiction agrees with direction")
