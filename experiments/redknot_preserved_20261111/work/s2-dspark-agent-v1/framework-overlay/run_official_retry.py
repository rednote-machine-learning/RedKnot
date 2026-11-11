"""Load the frozen official adapter with the one independent policy overlay.

All original CLI arguments are passed through. The official threaded worker
shares this module binding; container subprocesses retain the original pinned
routing guard, not a second copy of the host workspace admission policy.
"""
from pathlib import Path
import hashlib
import importlib.util
import sys

FROZEN_FRAMEWORK_ROOT = Path('/tmp/redknot-s2-agent.41ng63_n/assets/framework')
OVERLAY = Path(__file__).resolve().parent
ORIGINAL_ENTRY_SHA256 = 'c8332522973aa62560c27dad70f3a6f01c1d4eb66ad4acb5066db1159f174fde'
ORIGINAL_WORKER_SHA256 = 'bd4610a7489c14aca39079351e212d0f76c36482cb5c077fd72acfc05360b296'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_overlay(framework_root=None):
    root = Path(framework_root or FROZEN_FRAMEWORK_ROOT).resolve()
    if file_sha(root / 'run_official.py') != ORIGINAL_ENTRY_SHA256:
        raise ValueError('Frozen official adapter changed')
    if file_sha(root / 'benchmarks/benchmarks/utils/evaluation.py') != ORIGINAL_WORKER_SHA256:
        raise ValueError('Frozen official threaded worker changed')
    existing = sys.modules.get('runtime_policy')
    if existing is not None:
        if Path(existing.__file__).resolve() != OVERLAY / 'runtime_policy.py' or existing.B != root:
            raise RuntimeError('Another runtime policy was imported before the overlay')
        return existing
    if 'run_official' in sys.modules:
        raise RuntimeError('Frozen adapter was imported before installing the policy')
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('runtime_policy', OVERLAY / 'runtime_policy.py')
    policy = importlib.util.module_from_spec(spec)
    policy.FROZEN_FRAMEWORK_ROOT = root
    sys.modules['runtime_policy'] = policy
    try:
        spec.loader.exec_module(policy)
    except BaseException:
        sys.modules.pop('runtime_policy', None)
        raise
    if policy.B != root or policy.GUARD != root / 'runtime_guard':
        raise RuntimeError('Overlay relocated frozen configuration or routing guard')
    return policy


def load_official(framework_root=None):
    policy = install_overlay(framework_root)
    root = policy.B
    official = sys.modules.get('run_official')
    if official is None:
        spec = importlib.util.spec_from_file_location('run_official', root / 'run_official.py')
        official = importlib.util.module_from_spec(spec)
        sys.modules['run_official'] = official
        try:
            spec.loader.exec_module(official)
        except BaseException:
            sys.modules.pop('run_official', None)
            raise
    if (Path(official.__file__).resolve() != root / 'run_official.py' or official.B != root
            or official.workspace_kwargs is not policy.workspace_kwargs
            or official.admitted_environment is not policy.admitted_environment
            or official.validate_config is not policy.validate_config):
        raise RuntimeError('Original adapter is not bound consistently to the overlay')
    return official


def main(argv=None):
    return load_official().main(argv)


if __name__ == '__main__':
    main()
