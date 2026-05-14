"""
Trust package — Layer 1 governance for the ABCA platform.

Usage:
    from trust import configure_trust, get_emitter

    # At startup (once TRUST_EVENTS_TABLE_NAME env var is set):
    configure_trust()

    # Throughout the codebase:
    emitter = get_emitter()
    if emitter:
        emitter.task_complete(...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trust.emitter import TrustEventEmitter
    from trust.graduation import AutonomyGraduationEngine

_emitter: TrustEventEmitter | None = None
_graduation_engine: AutonomyGraduationEngine | None = None


def configure_trust(table_name: str | None = None) -> TrustEventEmitter:
    """
    Initialise the global trust emitter and graduation engine using DynamoDB.
    Call once at agent startup. table_name defaults to TRUST_EVENTS_TABLE_NAME env var.
    """
    global _emitter, _graduation_engine
    from trust.dynamo_store import DynamoTrustEventStore
    from trust.emitter import TrustEventEmitter
    from trust.graduation import AutonomyGraduationEngine

    store = DynamoTrustEventStore(table_name=table_name)
    _emitter = TrustEventEmitter(store)
    _graduation_engine = AutonomyGraduationEngine(store)
    return _emitter


def get_emitter() -> TrustEventEmitter | None:
    """Return the configured emitter, or None if not yet configured."""
    return _emitter


def get_graduation_engine() -> AutonomyGraduationEngine | None:
    """Return the configured graduation engine, or None if not yet configured."""
    return _graduation_engine
