"""Frozen, distinct role instructions over caller-supplied PIT public features."""

from __future__ import annotations

from types import MappingProxyType


ROLE_INSTRUCTIONS = MappingProxyType({
    "MICROSTRUCTURE": (
        "Inspect only pre-cutoff opp_mid, spread, OBI, bid/ask depth, "
        "short-horizon PM changes, and book update activity. Report uncertainty."
    ),
    "MOMENTUM": (
        "Inspect only pre-cutoff BTC returns at 1/3/5/10/30 seconds, "
        "BTC distance in basis points, signed flow, and short-horizon "
        "directional consistency. Report uncertainty."
    ),
    "LIQUIDITY": (
        "Inspect only pre-cutoff spread, bid/ask depth, update count, "
        "book coverage, and execution-condition fragility. Do not recommend a trade."
    ),
    "REGIME": (
        "Compare pre-cutoff momentum versus mean reversion and cross-feature "
        "agreement or disagreement. State uncertainty; never use future outcomes."
    ),
    "BULL_RESEARCHER": (
        "Construct the strongest evidence-supported UP case from validated "
        "pre-cutoff reports. Do not fabricate evidence; abstain with UNCLEAR "
        "if no defensible UP thesis exists."
    ),
    "BEAR_RESEARCHER": (
        "Construct the strongest evidence-supported DOWN case from validated "
        "pre-cutoff reports. Do not fabricate evidence; abstain with UNCLEAR "
        "if no defensible DOWN thesis exists."
    ),
    "RESEARCH_MANAGER": (
        "Synthesize both validated sides without inventing new evidence. "
        "May output ABSTAIN when unresolved; never request an action."
    ),
})


def role_instruction(role: str) -> str:
    try:
        return ROLE_INSTRUCTIONS[role]
    except KeyError as exc:
        raise ValueError("unknown K3 role") from exc
