"""
DynamoTrustEventStore — DynamoDB-backed persistence for trust events (ABCA).

Table schema:
  PK: agent_id (String)
  SK: event_id (String, ULID-prefixed)
  GSI AgentTaskTypeIndex: PK=agent_id, SK=task_type

Fail-open: all operations wrapped in try/except — trust failures must
never block task execution.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

from trust.models import TrustEvent, TrustSignalStrength
from trust.store import TrustEventStore

logger = logging.getLogger(__name__)

_TABLE_ENV_VAR = "TRUST_EVENTS_TABLE_NAME"
_GSI_NAME = "AgentTaskTypeIndex"


class DynamoTrustEventStore(TrustEventStore):
    """
    DynamoDB-backed TrustEventStore for ABCA.

    All public methods are fail-open: they log warnings on error but
    never raise, so trust recording never blocks task execution.
    """

    def __init__(self, table_name: str | None = None) -> None:
        self._table_name = table_name or os.environ.get(_TABLE_ENV_VAR, "")
        self._table: Any = None  # lazy-initialised

    def _get_table(self) -> Any:
        """Lazily initialise the DynamoDB Table resource."""
        if self._table is None:
            if not self._table_name:
                raise RuntimeError(
                    f"DynamoTrustEventStore: table name not configured "
                    f"(set {_TABLE_ENV_VAR} env var)"
                )
            dynamodb = boto3.resource("dynamodb")
            self._table = dynamodb.Table(self._table_name)
        return self._table

    # ------------------------------------------------------------------
    # TrustEventStore interface
    # ------------------------------------------------------------------

    def write(self, event: TrustEvent) -> None:
        """PutItem the trust event into DynamoDB. Fail-open."""
        try:
            table = self._get_table()
            item: dict[str, Any] = {
                "agent_id": event.agent_id,
                "event_id": event.event_id,
                "task_id": event.task_id,
                "task_type": event.task_type,
                "event_type": event.event_type.value,
                "signal": event.signal.value,
                "timestamp": event.timestamp.isoformat(),
                "autonomy_level": event.autonomy_level,
            }
            # Store non-empty metadata as a map attribute
            if event.metadata:
                item["metadata"] = event.metadata
            table.put_item(Item=item)
            logger.debug(
                "DynamoTrustEventStore: wrote %s for agent=%s task_id=%s",
                event.event_type.value,
                event.agent_id,
                event.task_id,
            )
        except Exception as exc:
            logger.warning("DynamoTrustEventStore.write failed (fail-open): %s", exc)

    def count_by_signal(
        self,
        agent_id: str,
        task_type: str | None = None,
    ) -> dict[str, int]:
        """
        Query by agent_id (+ optional task_type filter) and count by signal.

        Uses AgentTaskTypeIndex GSI when task_type is supplied; falls back
        to a base-table query on agent_id alone otherwise.

        Returns {"positive": N, "neutral": N, "negative": N, "critical": N}
        """
        counts: dict[str, int] = {s.value: 0 for s in TrustSignalStrength}
        try:
            table = self._get_table()
            if task_type:
                # Use GSI: agent_id PK + task_type SK
                response = table.query(
                    IndexName=_GSI_NAME,
                    KeyConditionExpression=(
                        Key("agent_id").eq(agent_id) & Key("task_type").eq(task_type)
                    ),
                    ProjectionExpression="signal",
                )
            else:
                # Base table query on agent_id PK only
                response = table.query(
                    KeyConditionExpression=Key("agent_id").eq(agent_id),
                    ProjectionExpression="signal",
                )
            for item in response.get("Items", []):
                sig = item.get("signal", "")
                if sig in counts:
                    counts[sig] += 1
            # Handle pagination (unlikely for trust events, but correct)
            while "LastEvaluatedKey" in response:
                kwargs: dict[str, Any] = {
                    "ExclusiveStartKey": response["LastEvaluatedKey"],
                    "ProjectionExpression": "signal",
                }
                if task_type:
                    kwargs["IndexName"] = _GSI_NAME
                    kwargs["KeyConditionExpression"] = Key("agent_id").eq(agent_id) & Key(
                        "task_type"
                    ).eq(task_type)
                else:
                    kwargs["KeyConditionExpression"] = Key("agent_id").eq(agent_id)
                response = table.query(**kwargs)
                for item in response.get("Items", []):
                    sig = item.get("signal", "")
                    if sig in counts:
                        counts[sig] += 1
        except Exception as exc:
            logger.warning("DynamoTrustEventStore.count_by_signal failed (fail-open): %s", exc)
        return counts
