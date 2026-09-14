"""Thread-safe byte-budgeted LRU storage, independent of tensor frameworks."""

from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any


@dataclass
class _Entry:
    payload: Any
    nbytes: int
    pins: int = 0


def _validate_bytes(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, excluding bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_key(key: str) -> None:
    if not isinstance(key, str):
        raise TypeError("cache keys must be strings")
    if not key:
        raise ValueError("cache keys must not be empty")


# REDKNOT: RK-CACHE — byte budget, LRU and in-use eviction protection.
class CacheManager:
    """Store payload references within a caller-accounted byte budget.

    The trusted caller provides accurate ``nbytes`` after cloning its tensors.
    This class does not copy payloads, inspect tensors, or charge Python metadata.
    Payloads must remain read-only, including while leased. References retained
    outside an active lease are caller-owned and outside the cache's budget;
    eviction releases only the cache's ownership, not those external references.

    Successful puts and leases refresh LRU order. Failed operations leave that
    order unchanged. Zero-byte entries are permitted, including at zero budget.

    Args:
        max_bytes: Nonnegative integer limit for cache-owned payload bytes.

    Raises:
        TypeError: The budget is not an integer, or is a bool.
        ValueError: The budget is negative.
    """

    def __init__(self, max_bytes: int):
        _validate_bytes(max_bytes, "max_bytes")
        self._max_bytes = max_bytes
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._rejected_puts = 0
        self._lock = RLock()

    def put(self, key: str, payload: Any, nbytes: int) -> bool:
        """Insert or replace a payload, atomically planning capacity evictions.

        Args:
            key: Nonempty string identifying the payload.
            payload: Caller-owned object to store by reference as read-only.
            nbytes: Trusted nonnegative integer size of this payload in bytes.

        Returns:
            True on success. False if the key is pinned, the payload exceeds the
            budget, or unpinned victims cannot free enough space. Refusal does
            not remove or replace any entry.

        Raises:
            TypeError: Key or size has an invalid type; bool sizes are invalid.
            ValueError: Key is empty or size is negative.
        """
        _validate_key(key)
        _validate_bytes(nbytes, "nbytes")
        with self._lock:
            previous = self._entries.get(key)
            if nbytes > self._max_bytes or (previous is not None and previous.pins):
                self._rejected_puts += 1
                return False

            projected_bytes = self._bytes + nbytes
            if previous is not None:
                projected_bytes -= previous.nbytes
            victims = []
            if projected_bytes > self._max_bytes:
                for victim_key, entry in self._entries.items():
                    if victim_key != key and entry.pins == 0:
                        victims.append(victim_key)
                        projected_bytes -= entry.nbytes
                        if projected_bytes <= self._max_bytes:
                            break
            if projected_bytes > self._max_bytes:
                self._rejected_puts += 1
                return False

            new_entry = _Entry(payload, nbytes)
            # Keep retired payloads alive until the lock has been released.
            retired = [self._entries.pop(victim_key) for victim_key in victims]
            self._entries[key] = new_entry
            self._entries.move_to_end(key)
            self._bytes = projected_bytes
            self._evictions += len(retired)
            return True

    @contextmanager
    def lease(self, keys: Sequence[str]) -> Iterator[Mapping[str, Any] | None]:
        """Pin all requested entries until context exit, or yield None on a miss.

        Duplicate keys pin once per lease; nested leases hold independent pins.
        A successful empty request yields an empty mapping. Hits and misses count
        lease requests, not keys. No lock is held while caller code uses a lease.

        Args:
            keys: Sequence of nonempty string keys, excluding a bare string.

        Yields:
            Read-only mapping of keys to payloads on an all-or-nothing hit, or
            None on any miss. Payload objects themselves are not frozen; callers
            must not mutate them. Pins are released even if caller code raises.

        Raises:
            TypeError: Keys is a bare string/bytes, or a key is not a string.
            ValueError: A key is empty.
        """
        if isinstance(keys, (str, bytes)):
            raise TypeError("keys must be a sequence of cache keys")
        requested = tuple(keys)
        for key in requested:
            _validate_key(key)
        unique_keys = tuple(dict.fromkeys(requested))
        leased = []
        values = None
        with self._lock:
            if any(key not in self._entries for key in unique_keys):
                self._misses += 1
            else:
                leased = [self._entries[key] for key in unique_keys]
                values = MappingProxyType(
                    {key: entry.payload for key, entry in zip(unique_keys, leased)}
                )
                for key, entry in zip(unique_keys, leased):
                    entry.pins += 1
                    self._entries.move_to_end(key)
                self._hits += 1
        try:
            yield values
        finally:
            with self._lock:
                for entry in leased:
                    entry.pins -= 1

    def stats(self) -> dict[str, int]:
        """Return a consistent detached snapshot of occupancy and counters.

        Hits/misses count lease requests. Evictions count capacity removals,
        excluding replacements and clear. Rejected puts count valid requests
        refused for budget/pin constraints, excluding argument validation errors.
        """
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "max_bytes": self._max_bytes,
                "pinned_entries": sum(
                    entry.pins > 0 for entry in self._entries.values()
                ),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "rejected_puts": self._rejected_puts,
            }

    def clear(self) -> int:
        """Remove every currently unpinned entry and return the removal count."""
        with self._lock:
            removable = [key for key, entry in self._entries.items() if not entry.pins]
            retired = [self._entries.pop(key) for key in removable]
            self._bytes -= sum(entry.nbytes for entry in retired)
            return len(retired)
