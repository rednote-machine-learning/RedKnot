"""Run only this module's CPU tests and record exact source/test fingerprints.

Torch is optional. If present, accelerator discovery/initialization entry points
are blocked before tests. No model factory or scheduler is invoked. The optional
JSON output is created exclusively and never replaces an existing record.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest
from unittest.mock import patch
from contextlib import ExitStack


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="New JSON file, never overwrite")
    args = parser.parse_args(argv)
    output_path = Path(args.output).resolve() if args.output else None
    root = Path(__file__).resolve().parent
    # Clean-import subprocess tests must resolve the same isolated package even
    # when invoked via an absolute path from the parent engineering workspace.
    os.chdir(root)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    paths = sorted(list((root / "kv_cache_engram").glob("*.py"))
                   + list((root / "tests").glob("*.py"))
                   + list((root / "gpu_pilot").glob("*.py")) + [Path(__file__).resolve()])
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    record = dict(schema="kv-cache-engram-cpu-test-run-v1", python=platform.python_version(),
                  source_sha256=hashes, cuda_visible_devices="", torch=None,
                  cuda_init_guard_installed=False, cuda_initialized_after=None)
    start = time.perf_counter()
    stream = io.StringIO()
    with ExitStack() as stack:
        if importlib.util.find_spec("torch") is not None:
            import torch
            if torch.cuda.is_initialized():
                raise RuntimeError("CPU validation requires uninitialized CUDA")
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            for name in ("init", "_lazy_init", "is_available", "device_count", "set_device",
                         "current_device", "synchronize"):
                stack.enter_context(patch.object(torch.cuda, name,
                                    side_effect=AssertionError("CPU-only validation: CUDA forbidden")))
            record.update(torch=torch.__version__, cpu_threads=1, cuda_init_guard_installed=True)
        suite = unittest.defaultTestLoader.discover(str(root / "tests"))
        result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
        if record["torch"] is not None:
            record["cuda_initialized_after"] = torch.cuda.is_initialized()
    record.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), skipped_tests=[str(t) for t, _ in result.skipped],
                  success=result.wasSuccessful(), elapsed_s=time.perf_counter()-start,
                  output=stream.getvalue())
    if output_path is not None:
        with output_path.open("x", encoding="utf-8") as output:
            json.dump(record, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
    print(json.dumps(record, sort_keys=True, allow_nan=False))
    return 0 if result.wasSuccessful() and record["cuda_initialized_after"] is not True else 1


if __name__ == "__main__":
    sys.exit(main())
