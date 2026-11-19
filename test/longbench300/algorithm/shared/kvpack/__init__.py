"""Cross-layer offline KV data movement; native projection/attention stay external."""
from .pack import Document, PackPlan, CachedHaloGeometry, KVPackSession, gather_plan_from_cache, oracle_gather, oracle_scatter, oracle_halo_scatter

__all__ = ['Document', 'PackPlan', 'CachedHaloGeometry', 'KVPackSession', 'gather_plan_from_cache', 'oracle_gather', 'oracle_scatter', 'oracle_halo_scatter']
