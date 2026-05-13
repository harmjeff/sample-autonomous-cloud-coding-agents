import httpx

from interfaces.ltm import (
    CapabilityMismatchError,
    ConsolidationResult,
    ContradictionResult,
    LTMCapability,
    LTMInterface,
    MemoryCategory,
    MemoryMetadata,
    MemoryResult,
    NegotiationResult,
    RecallConfig,
    apply_composite_scoring,
    decay_rate_for,
)

_MEM0_CAPABILITIES = {
    LTMCapability.SEMANTIC_SEARCH,
    LTMCapability.ENTITY_LOOKUP,
    LTMCapability.RELATIONSHIP_GRAPH,
    LTMCapability.WRITE,
    LTMCapability.IMPORTANCE_SCORING,
    LTMCapability.DECAY,
    LTMCapability.CONSOLIDATION,
    LTMCapability.ATOMIC_EXTRACTION,
    LTMCapability.CONTRADICTION_DETECTION,
}


class Mem0LTM(LTMInterface):
    """
    LTM backend backed by the self-hosted Mem0 service.
    Communicates via the Mem0 HTTP service at infra/mem0/server.py.
    """

    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def capabilities(self) -> set[LTMCapability]:
        return _MEM0_CAPABILITIES

    def read(
        self,
        query: str,
        user_id: str,
        mode: LTMCapability = LTMCapability.SEMANTIC_SEARCH,
        limit: int = 10,
        recall_config: RecallConfig | None = None,
    ) -> list[MemoryResult]:
        if mode == LTMCapability.ENTITY_LOOKUP:
            response = self._client.get(f"/memories/{user_id}")
            response.raise_for_status()
            results = response.json().get("results", {})
            memories = results.get("results", results) if isinstance(results, dict) else results
        else:
            response = self._client.post(
                "/search",
                json={"query": query, "user_id": user_id, "limit": limit},
            )
            response.raise_for_status()
            memories = response.json().get("results", {}).get("results", [])

        out = [_parse_memory_result(m) for m in (memories if isinstance(memories, list) else [])]

        # Reinforce importance for every retrieved memory (spaced repetition)
        for r in out:
            if r.id:
                try:
                    self.reinforce(r.id, user_id)
                except Exception:
                    pass

        if recall_config is not None:
            out = apply_composite_scoring(out, recall_config)

        return out

    def write(self, content: str, user_id: str, metadata: MemoryMetadata) -> str:
        # Derive decay_rate from category if not explicitly overridden
        try:
            cat = MemoryCategory(metadata.category)
        except ValueError:
            cat = MemoryCategory.ACTIVE
        effective_decay_rate = (
            metadata.decay_rate if metadata.decay_rate != 0.1 else decay_rate_for(cat)
        )

        messages = [{"role": "user", "content": content}]
        meta = {
            "source": metadata.source,
            "agent_id": metadata.agent_id,
            "task_id": metadata.task_id,
            "importance_hint": metadata.importance_hint,
            "category": metadata.category,
            "decay_rate": effective_decay_rate,
            "access_count": metadata.access_count,
            "created_at": metadata.created_at,
        }
        if metadata.last_accessed:
            meta["last_accessed"] = metadata.last_accessed
        if metadata.entity_type:
            meta["entity_type"] = metadata.entity_type
        if metadata.entity_id:
            meta["entity_id"] = metadata.entity_id
        if metadata.tags:
            meta["tags"] = ",".join(metadata.tags)

        response = self._client.post(
            "/memories",
            json={"messages": messages, "user_id": user_id, "metadata": meta},
        )
        response.raise_for_status()
        result = response.json().get("result", {})
        results = result.get("results", [])
        if results and isinstance(results, list):
            return results[0].get("id", "")
        return ""

    def forget(self, memory_id: str) -> bool:
        response = self._client.delete(f"/memories/{memory_id}")
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    def decay(self, user_id: str, threshold: float = 0.05) -> int:
        """Prune memories whose effective importance has dropped below threshold."""
        response = self._client.post(
            "/memories/decay",
            json={"user_id": user_id, "threshold": threshold},
        )
        response.raise_for_status()
        return response.json().get("pruned", 0)

    def reinforce(self, memory_id: str, user_id: str) -> None:
        """Bump importance on successful retrieval (spaced repetition)."""
        response = self._client.post(
            f"/memories/{memory_id}/reinforce",
            json={"user_id": user_id},
        )
        # Best-effort — don't raise on 404 (memory may have been pruned)
        if response.status_code not in (200, 404):
            response.raise_for_status()

    def consolidate(
        self,
        user_id: str,
        similarity_threshold: float = 0.85,
    ) -> ConsolidationResult:
        """Merge near-duplicate memories and resolve contradictions by recency."""
        response = self._client.post(
            "/memories/consolidate",
            json={"user_id": user_id, "similarity_threshold": similarity_threshold},
            # Consolidation involves LLM calls — give it more time
        )
        response.raise_for_status()
        data = response.json()
        return ConsolidationResult(
            user_id=user_id,
            clusters_found=data.get("clusters_found", 0),
            merged_count=data.get("merged_count", 0),
            pruned_count=data.get("pruned_count", 0),
        )

    def extract_and_write(
        self,
        content: str,
        user_id: str,
        metadata: MemoryMetadata,
    ) -> list[str]:
        """Decompose blob into atomic facts via LLM, write each separately."""
        response = self._client.post(
            "/memories/extract",
            json={
                "content": content,
                "user_id": user_id,
                "metadata": {
                    "source": metadata.source,
                    "agent_id": metadata.agent_id,
                    "task_id": metadata.task_id,
                    "importance_hint": metadata.importance_hint,
                    "category": metadata.category,
                    "decay_rate": metadata.decay_rate,
                    "entity_type": metadata.entity_type,
                    "entity_id": metadata.entity_id,
                    "tags": ",".join(metadata.tags) if metadata.tags else "",
                },
            },
        )
        response.raise_for_status()
        return response.json().get("memory_ids", [])

    def detect_contradictions(
        self,
        content: str,
        user_id: str,
        source_trust: float = 0.5,
        similarity_threshold: float = 0.85,
    ) -> "ContradictionResult":
        """Search for contradictions before writing. Returns resolution plan."""
        response = self._client.post(
            "/memories/detect_contradictions",
            json={
                "content": content,
                "user_id": user_id,
                "source_trust": source_trust,
                "similarity_threshold": similarity_threshold,
            },
        )
        response.raise_for_status()
        data = response.json()
        return ContradictionResult(
            contradiction_found=data.get("contradiction_found", False),
            resolution=data.get("resolution", "KEEP_BOTH"),
            conflicting_memory_id=data.get("conflicting_memory_id"),
            conflicting_content=data.get("conflicting_content"),
            explanation=data.get("explanation", ""),
        )

    def negotiate(
        self,
        required: list[LTMCapability],
        optional: list[LTMCapability] | None = None,
    ) -> NegotiationResult:
        supported = self.capabilities()
        missing_required = set(required) - supported
        missing_optional = set(optional or []) - supported

        result = NegotiationResult(
            supported=supported,
            missing_required=missing_required,
            missing_optional=missing_optional,
        )

        if missing_required:
            raise CapabilityMismatchError(missing_required)

        return result


def _parse_memory_result(m: dict) -> MemoryResult:
    meta = m.get("metadata", {})
    return MemoryResult(
        id=m.get("id", ""),
        content=m.get("memory", ""),
        metadata=meta,
        score=m.get("score"),
        importance=meta.get("importance_hint"),
        access_count=meta.get("access_count"),
        last_accessed=meta.get("last_accessed"),
        created_at=meta.get("created_at"),
    )
