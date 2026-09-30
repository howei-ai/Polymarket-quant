"""The four fixed public-state analyst roles."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .context import ShadowContext
from .role_instructions import role_instruction
from .schemas import ANALYST_ROLES, AnalystReport


K3_PROMPT_SCHEMA_VERSION = "k3_shadow_research_v1"


def analyst_request(role: str, context: ShadowContext) -> dict[str, Any]:
    if role not in ANALYST_ROLES:
        raise ValueError("unknown analyst role")
    return {
        "schema_version": K3_PROMPT_SCHEMA_VERSION,
        "role": role,
        "instruction": role_instruction(role),
        "context": context.to_dict(),
    }


def reports_payload(reports: tuple[AnalystReport, ...]) -> list[dict[str, Any]]:
    return [asdict(report) for report in reports]
