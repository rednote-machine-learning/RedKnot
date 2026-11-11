"""CPU-only full-suite proof for the independent headtail20 three-arm pilot."""
import argparse
from contextlib import ExitStack
import io
import os
from pathlib import Path
import platform
import time
import unittest
from unittest.mock import patch


def test_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from test_ids(item)
        else:
            yield item.id()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from gpu_pilot import launch_pilot as lifecycle
    import torch

    lifecycle.require(not torch.cuda.is_initialized(), "CPU validation found initialized CUDA")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sources = lifecycle.source_files(root)
    stream = io.StringIO()
    start = time.perf_counter()
    with ExitStack() as stack:
        for name in ("init", "_lazy_init", "is_available", "device_count", "set_device", "current_device", "synchronize"):
            stack.enter_context(patch.object(torch.cuda, name, side_effect=AssertionError("CPU-only: CUDA forbidden")))
        suite = unittest.defaultTestLoader.discover(str(root / "tests"))
        names = list(test_ids(suite))
        result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    lifecycle.require(sources == lifecycle.source_files(root), "source changed during CPU tests")
    record = lifecycle.seal(dict(
        schema="kv-cache-engram-natural-boundary-cpu-proof-v1", source_sha256=sources,
        python=platform.python_version(), torch=torch.__version__, cpu_threads=1,
        cuda_visible_devices="", cuda_init_guard_installed=True, cuda_initialized=torch.cuda.is_initialized(),
        tests=result.testsRun, test_ids=names, successful=result.wasSuccessful(),
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        output=stream.getvalue(), elapsed_s=time.perf_counter()-start,
    ))
    lifecycle.write_exclusive(destination, record)
    print(stream.getvalue(), end="")
    print({k: record[k] for k in ("tests", "successful", "failures", "errors", "skipped", "cuda_initialized", "sha256")})
    return 0 if result.wasSuccessful() and not result.skipped and not record["cuda_initialized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
