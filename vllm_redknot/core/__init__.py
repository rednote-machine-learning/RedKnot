"""Portable RedKnot implementation extracted from the SGLang-hosted source.

REDKNOT-CORE: these modules are code assets, not automatically enabled vLLM
runtime hooks. Import explicit submodules; importing this package does not load
Torch, Triton, SGLang, vLLM, a model, or a CUDA context. See
``docs/CORE_EXTRACTION.md`` and ``docs/core_provenance.json`` for source lineage
and the separate runtime integration boundary.
"""
