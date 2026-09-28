"""Head-addressable KV lifecycle for RedKnot's SegPagedAttention.

This package can also be imported as ``head_kv`` with mem_cache on sys.path,
allowing lifecycle/network tests without importing the full SGLang server.
"""
from .pool import CapacityError, HeadPagePool, PageRef, StaleReference
from .manager import (HeadKVManager, ReadLease, RequestVersion, ReuseProof,
                      Segment, SegmentPatch, SegmentWrite, WriteTransaction)
from .segpaged import ManagedSegPagedKVCache

__all__ = ["CapacityError", "HeadPagePool", "PageRef", "StaleReference", "HeadKVManager",
           "ReadLease", "RequestVersion", "ReuseProof", "Segment", "SegmentPatch",
           "SegmentWrite", "WriteTransaction", "ManagedSegPagedKVCache"]
