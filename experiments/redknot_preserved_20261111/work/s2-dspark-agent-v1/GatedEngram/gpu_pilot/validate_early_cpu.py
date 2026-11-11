"""Run the independent CPU suite and exclusively save source-bound evidence."""
from pathlib import Path
import sys
import time
import unittest

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE))
from gpu_pilot import launch_pilot as life


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item.id()


def main():
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU validation must not initialize CUDA")
    sources = life.source_files(STAGE)
    suite = unittest.TestLoader().discover(str(STAGE/"tests"))
    names = list(flatten(suite))
    started = time.time()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if sources != life.source_files(STAGE):
        raise RuntimeError("source changed during CPU tests")
    proof = life.seal(dict(schema="early-mhc-cpu-validation-v1",started_unix=started,
        finished_unix=time.time(),successful=result.wasSuccessful(),tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped),
        test_ids=names,source_sha256=sources,cuda_initialized=torch.cuda.is_initialized(),
        torch_version=torch.__version__))
    life.write_exclusive(STAGE/"EARLY_MHC_CPU_TESTS.json",proof)
    return 0 if result.wasSuccessful() and not result.skipped and not torch.cuda.is_initialized() else 1


if __name__ == "__main__":
    raise SystemExit(main())
