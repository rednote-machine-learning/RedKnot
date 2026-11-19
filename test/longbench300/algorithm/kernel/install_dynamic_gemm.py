"""Switch the frozen per-32 SM90 GEMM in this process, preserving all original callers."""
import ast
import hashlib
import importlib
from pathlib import Path


SOURCE_SHA256 = "8e78883a887784f31cc12462daeb80000c3db52406a5d7fe5e0cbefd8c8490c5"
ORIGINAL = "deepinfer.model_executor.layers.sm90.gemm"


def _kernel_node(path, name):
    return next(n for n in ast.parse(Path(path).read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def verify_sources(native_path, candidate_path):
    if hashlib.sha256(Path(native_path).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Different native SM90 GEMM")
    native = _kernel_node(native_path, "_gemm")
    candidate = _kernel_node(candidate_path, "_gemm_dynamic_m")
    native.name = candidate.name
    for arg in native.args.args:
        if arg.arg in ("M", "ASM", "ASK"):
            if ast.unparse(arg.annotation) != "tl.constexpr":
                raise ValueError(f"Unexpected original annotation for {arg.arg}")
            arg.annotation = None
    native.decorator_list = candidate.decorator_list = []
    if ast.dump(native) != ast.dump(candidate):
        raise ValueError("Candidate changed the original GEMM body or a static argument")


def _cache_size(kernel, device):
    cache = getattr(kernel, "device_caches", None)
    if cache is not None:
        state = cache.get(device)
        if state is None:
            return 0
        if isinstance(state, (tuple, list)):
            state = state[0]
        if isinstance(state, dict):
            return len(state)
        raise TypeError("Unknown Triton device cache structure")
    cache = getattr(kernel, "cache", None)
    if not isinstance(cache, dict):
        raise TypeError("Unrecognized Triton kernel cache API")
    return len(cache.get(device, {}))


class KernelSwitch:
    def __init__(self, module, candidate):
        self.module = module
        self.native = module._gemm
        self.candidate = candidate
        self.active = self.native
        self.closed = False

    def set_dynamic(self, enabled):
        if type(enabled) is not bool or self.closed or self.module._gemm is not self.active:
            raise RuntimeError("Unexpected process-local SM90 kernel ownership")
        self.active = self.candidate if enabled else self.native
        self.module._gemm = self.active

    def stats(self, device=None):
        import torch
        import triton
        device = torch.cuda.current_device() if device is None else int(device)
        return dict(device=device, triton_version=triton.__version__,
                    enabled=self.active is self.candidate,
                    original_variants=_cache_size(self.native, device),
                    dynamic_variants=_cache_size(self.candidate, device),
                    native_source_sha256=SOURCE_SHA256)

    def restore(self):
        self.set_dynamic(False)
        del self.module._redknot_dynamic_gemm_owner
        self.closed = True


def install_dynamic_gemm(enabled=True):
    module = importlib.import_module(ORIGINAL)
    candidate = importlib.import_module("dynamic_kernel")
    if Path(candidate.__file__).resolve().parent != Path(__file__).resolve().parent:
        raise ImportError("Unexpected dynamic_kernel location")
    verify_sources(module.__file__, candidate.__file__)
    if (hasattr(module, "_redknot_dynamic_gemm_owner") or
            module.gemm_nt.__globals__["_gemm"] is not module._gemm or
            module._gemm.fn.__module__ != ORIGINAL):
        raise RuntimeError("SM90 GEMM already replaced by a different installer")
    handle = KernelSwitch(module, candidate._gemm_dynamic_m)
    handle.set_dynamic(enabled)
    module._redknot_dynamic_gemm_owner = handle
    return handle
