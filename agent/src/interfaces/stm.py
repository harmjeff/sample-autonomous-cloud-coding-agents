from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from interfaces.ltm import LTMInterface, MemoryMetadata


class STMInterface(ABC):
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Write a value. Optional TTL in seconds."""
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str) -> Any:
        """Read a value. Returns None if key does not exist."""
        raise NotImplementedError

    @abstractmethod
    def append(self, key: str, value: Any) -> None:
        """Append a value to a list stored at key. Creates the list if absent."""
        raise NotImplementedError

    @abstractmethod
    def promote(
        self,
        key: str,
        ltm: "LTMInterface",
        user_id: str,
        metadata: "MemoryMetadata",
    ) -> str:
        """
        Move a STM entry into LTM. Returns the LTM memory ID.
        The STM entry is NOT removed — call clear() separately if desired.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self, key: str) -> None:
        """Delete a specific key."""
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        """Delete all keys in this session's namespace."""
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot of all current STM state."""
        raise NotImplementedError

    @abstractmethod
    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore STM state from a snapshot. Overwrites current state."""
        raise NotImplementedError
