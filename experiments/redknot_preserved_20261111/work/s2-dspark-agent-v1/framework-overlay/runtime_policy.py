"""External runtime admission settings; never applied to image construction."""
from __future__ import annotations
import contextlib, hashlib, json, os, pathlib, secrets, sys
if 'FROZEN_FRAMEWORK_ROOT' not in globals():
    raise RuntimeError('Load this independent policy through run_official_retry.install_overlay')
B = pathlib.Path(FROZEN_FRAMEWORK_ROOT).resolve()
GUARD = B / 'runtime_guard'
sys.path.insert(0, str(GUARD))
from routing_guard import validate_llm

RUNTIME_ENV = {'APPTAINER_DROP_CAPS': 'all', 'APPTAINER_NO_PRIVS': 'true',
               'CONVERSATION_TIMEOUT': '1800', 'PYTHONPATH': '/s2_runtime_guard',
               'LITELLM_LOCAL_MODEL_COST_MAP': 'True'}
DANGEROUS_APPTAINER = ('BIND', 'BINDPATH', 'MOUNT', 'OVERLAY', 'OVERLAYIMAGE',
    'WORKDIR', 'HOME', 'CONTAINLIBS', 'ADD_CAPS', 'KEEP_PRIVS', 'ALLOW_SETUID',
    'NO_PID', 'NV', 'NVCCLI', 'ROCM', 'FUSEMOUNT', 'SECURITY')
OFFICIAL_DISABLED_TRACING_ENV = frozenset({'LMNR_PROJECT_API_KEY', 'LMNR_SPAN_CONTEXT'})

def validate_config(path):
    from types import SimpleNamespace
    data = json.loads(pathlib.Path(path).read_text())
    template = json.loads((B/'config/llm.Full.json').read_text())
    if data != template:
        raise ValueError('LLM configuration differs from admitted common Full/S2 policy')
    validate_llm(SimpleNamespace(**data))
    return data

def validate_environment(env):
    denied = ['LLM_BASE_URL', 'LLM_API_MASTER_KEY', 'OPENHANDS_APPTAINER_EXTRA_BINDS', 'LMNR_PROJECT_API_KEY']
    denied += [prefix + key for prefix in ('APPTAINER_', 'SINGULARITY_') for key in DANGEROUS_APPTAINER]
    denied += [k for k in env if k.startswith(('APPTAINERENV_', 'SINGULARITYENV_', 'OH_SESSION_API_KEYS_'))]
    found = sorted(k for k in denied if env.get(k))
    if found:
        raise ValueError('Unadmitted runtime environment keys: ' + ', '.join(found))

@contextlib.contextmanager
def admitted_environment():
    validate_environment(os.environ)
    values = {**RUNTIME_ENV, 'SESSION_API_KEY': secrets.token_urlsafe(32)}
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield {'session_api_key_sha256': hashlib.sha256(values['SESSION_API_KEY'].encode()).hexdigest(),
               'runtime_environment': RUNTIME_ENV, 'session_api_key_generated': True,
               'policy_overlay_source_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
               'frozen_framework_root': str(B), 'external_tracing_enabled': False,
               'official_disabled_tracing_env_names_filtered': sorted(OFFICIAL_DISABLED_TRACING_ENV)}
    finally:
        for k,v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

def workspace_kwargs(kwargs, real_uid):
    if real_uid != 0:
        raise RuntimeError('Root-mode adapter requires real uid0')
    result = dict(kwargs)
    if result.get('use_fakeroot') not in (None, False):
        raise ValueError('Conflicting fakeroot argument')
    if result.get('enable_gpu'):
        raise ValueError('Agent workspace must not expose GPU devices')
    if not os.environ.get('SESSION_API_KEY'):
        raise ValueError('Per-run SESSION_API_KEY must be generated before workspace startup')
    for k,v in RUNTIME_ENV.items():
        if os.environ.get(k) != v:
            raise ValueError(f'Runtime environment not admitted: {k}')
    existing_binds = result.get('extra_bind_mounts') or []
    if existing_binds:
        raise ValueError('Unexpected workspace extra binds (tokenizer is gateway-owned)')
    forward = result.get('forward_env') or []
    # The official outer worker always passes these two tracing names, even
    # when no Laminar project key exists. Accept that exact call signature,
    # while keeping external tracing disabled and forwarding neither field.
    if os.environ.get('LMNR_PROJECT_API_KEY'):
        raise ValueError('External Laminar tracing credentials are not admitted')
    if set(forward) - {'DEBUG'} - OFFICIAL_DISABLED_TRACING_ENV:
        raise ValueError('Unexpected agent-server environment forwarding')
    forward = [key for key in forward if key not in OFFICIAL_DISABLED_TRACING_ENV]
    result.update(use_fakeroot=False, enable_gpu=False, enable_docker_compat=True,
        disable_mount_locations=['hostfs','bind-paths'],
        extra_bind_mounts=[f'{GUARD}:/s2_runtime_guard:ro'],
        forward_env=[*forward, 'SESSION_API_KEY', 'PYTHONPATH', 'LITELLM_LOCAL_MODEL_COST_MAP'])
    return result
