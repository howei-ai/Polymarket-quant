"""Offline-first, research-only K3/Jev SHADOW classification.

This package has no trading, publication, or cohort mutation capability.
"""

from .runner import ShadowRunner
from .schemas import GateEvidence, ShadowResult

__all__ = ["GateEvidence", "ShadowResult", "ShadowRunner"]
