"""Calculate the common GPU bank budget across eight ranks."""

from kv_cache_engram.core import require

MAX_BANK_BYTES = 4 * 1024**3


def uniform_bank_budget(free_bytes_by_rank):
    require(type(free_bytes_by_rank) in (list, tuple) and len(free_bytes_by_rank) == 8
            and all(type(n) is int and n >= 0 for n in free_bytes_by_rank), 'all eight actual free-memory readings required')
    return min(MAX_BANK_BYTES, min(free_bytes_by_rank)//5)
