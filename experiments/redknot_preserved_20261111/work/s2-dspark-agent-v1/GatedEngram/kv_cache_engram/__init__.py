"""KV Cache Engram: isolated capture, CPU admission and pre-submit fallback.

Importing this package does not import Torch/DeepInfer or install runtime hooks.
Native execution is opt-in through explicitly installed adapter modules.
"""

from .cache import CacheArtifact, decode_cache, load_cache, save_cache
from .calibration import CalibrationRecord
from .core import Binding, CacheIdentity, OnlineRows, ValidationError, token_digest
from .gate import BlockPlan, GateDecision, GatePolicy, evaluate, make_plan
from .runtime import BlockTransaction, PreparedBlock, ProbeFailure, TransactionStateError

__version__ = "0.3.0"
__all__ = [
    "Binding",
    "CacheArtifact",
    "CacheIdentity",
    "OnlineRows",
    "ValidationError",
    "decode_cache",
    "load_cache",
    "save_cache",
    "token_digest",
    "CalibrationRecord",
    "BlockPlan",
    "GateDecision",
    "GatePolicy",
    "evaluate",
    "make_plan",
    "BlockTransaction",
    "PreparedBlock",
    "ProbeFailure",
    "TransactionStateError",
]
