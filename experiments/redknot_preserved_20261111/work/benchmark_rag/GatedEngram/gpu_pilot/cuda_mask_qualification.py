"""Additional mask-only CUDA qualification; never a quality or timing sample.

Import is CPU-only. Execution is called inside the original supervised GPU
worker, after native initialization and before the unchanged native/all-selected
qualification. The same helper used by the controller is exercised with CUDA
sync-debug set to error; no allow-sync context surrounds the helper itself.
"""
from kv_cache_engram.core import digest, require


def mask_cases():
    return [
        ("zero_rows", 0, [], False),
        ("none_selected", 8192, [], False),
        ("prefix127", 8192, list(range(127)), False),
        ("noncontiguous1229", 8192, [5*i for i in range(1229)], False),
        ("all_selected", 8192, list(range(8192)), True),
        ("all_via_fill", 8192, list(range(8192)), False),
        ("single", 1, [0], True),
    ]


def validate_report(value):
    require(value.get("schema") == "ffn-only-cuda-mask-qualification-v1"
            and value.get("passed") is True
            and value.get("device_type") == "cuda"
            and type(value.get("device_index")) is int and value["device_index"] >= 0
            and value.get("quality_eligible") is False and value.get("timing_valid") is False,
            "invalid CUDA mask qualification identity")
    tests = value.get("checks")
    require(type(tests) is list and len(tests) == len(mask_cases()), "missing CUDA mask checks")
    for actual, (name, rows, ids, all_selected) in zip(tests, mask_cases()):
        require(actual == dict(name=name, rows=rows, selected_rows=len(ids),
            selected_ids_sha256=digest(ids), all_selected=all_selected,
            sync_debug_mode=2, exact_mask=True, device_preserved=True, dtype_bool=True),
            "CUDA mask semantics or strict sync mode failed")
    return value


def run_cuda_mask_qualification(torch, allow_sync):
    from kv_cache_engram.ffn_only import build_ffn_mask
    require(torch.cuda.is_initialized(), "native model must initialize CUDA first")
    device = torch.device("cuda", torch.cuda.current_device())
    checks = []
    for name, rows, ids, all_selected in mask_cases():
        # These diagnostic inputs/transfers and result reads are deliberately
        # outside the strict region and outside all reported online TTFT.
        with allow_sync():
            selected_ids = torch.tensor(ids, dtype=torch.long, device=device)
            chosen = set(ids)
            expected = torch.tensor([i in chosen for i in range(rows)], dtype=torch.bool, device=device)
            torch.cuda.synchronize(device)
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            strict_mode = torch.cuda.get_sync_debug_mode()
            with torch.inference_mode():
                actual = build_ffn_mask(torch, rows, selected_ids, all_selected=all_selected)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        with allow_sync():
            torch.cuda.synchronize(device)
            exact = bool(torch.equal(actual, expected))
        check = dict(name=name, rows=rows, selected_rows=len(ids), selected_ids_sha256=digest(ids),
            all_selected=all_selected, sync_debug_mode=strict_mode, exact_mask=exact,
            device_preserved=actual.device == selected_ids.device, dtype_bool=actual.dtype == torch.bool)
        checks.append(check)
    report = dict(schema="ffn-only-cuda-mask-qualification-v1", passed=True,
        device_type=device.type, device_index=device.index, checks=checks,
        quality_eligible=False, timing_valid=False)
    return validate_report(report)
