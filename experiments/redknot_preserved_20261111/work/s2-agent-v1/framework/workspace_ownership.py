"""Exact per-process ownership for official Apptainer sessions created by this runner."""
from __future__ import annotations
import hashlib, json, os, pathlib, tempfile, time
ACTIVE = []
RECEIPT_DIR = None

def configure(directory):
    global RECEIPT_DIR
    RECEIPT_DIR = pathlib.Path(directory) / 'workspace-processes'
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)

def proc_identity(pid):
    base = pathlib.Path('/proc')/str(pid)
    stat = (base/'stat').read_text().rsplit(')', 1)[1].split()
    raw = (base/'cmdline').read_bytes()
    argv = [v.decode(errors='replace') for v in raw.split(b'\0') if v]
    redacted = ['SESSION_API_KEY=<redacted>' if v.startswith('SESSION_API_KEY=') else v for v in argv]
    return {'pid': pid, 'start_ticks': int(stat[19]), 'ppid': int(stat[1]),
            'pgid': int(stat[2]), 'sid': int(stat[3]), 'argv_redacted': redacted,
            'cmdline_bytes_sha256': hashlib.sha256(raw).hexdigest(),
            'exe': os.readlink(base/'exe'),
            'boot_id': pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}

def record(workspace, event, process, identity=None):
    if RECEIPT_DIR is None:
        raise RuntimeError('Workspace ownership receipt directory was not configured')
    identity = identity or proc_identity(process.pid)
    report = {'schema':'s2-workspace-process-ownership-v1','event':event,
              'observed_unix':time.time(),'owner_pid':os.getpid(),
              'process':identity,'returncode':process.poll(),
              'sif_path':getattr(workspace,'_sif_path',None),
              'workspace_mount':workspace.mount_dir,
              'guard_mount_readonly':True,'session_api_key_sha256':hashlib.sha256(os.environ['SESSION_API_KEY'].encode()).hexdigest()}
    target = RECEIPT_DIR/f'{identity["pid"]}-{identity["start_ticks"]}-{event}.json'
    descriptor,temporary=tempfile.mkstemp(prefix='.ownership-',suffix='.tmp',dir=RECEIPT_DIR)
    try:
        with os.fdopen(descriptor,'w') as f:
            json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
        os.link(temporary,target)  # Exclusive complete-file publication.
    finally:
        os.unlink(temporary)
    return identity

def make_workspace_class(original, kwargs_adapter):
    class S2OwnedApptainerWorkspace(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs_adapter(kwargs))
            process=getattr(self,'_process',None)
            identity=getattr(self,'_s2_owned_identity',None)
            if process is not None and identity:
                current=proc_identity(process.pid)
                stable=('start_ticks','pgid','sid')
                if any(current[k]!=identity[k] for k in stable):
                    raise RuntimeError('Apptainer identity changed across healthcheck')
                # exec may change argv/exe while preserving this exact process.
                record(self,'ready',process,current)
                object.__setattr__(self,'_s2_owned_identity',current)
        def _start_container(self):
            ACTIVE.append(self)
            try:
                super()._start_container()
            finally:
                if self._process is not None:
                    identity = record(self,'started',self._process)
                    object.__setattr__(self,'_s2_owned_identity',identity)
        def cleanup(self):
            process = getattr(self,'_process',None)
            identity = getattr(self,'_s2_owned_identity',None)
            if process is not None and identity:
                if process.poll() is not None:
                    # Avoid upstream killpg on an already-reaped/reusable PID.
                    self._process = None
                else:
                    actual = proc_identity(process.pid)
                    if (actual['start_ticks'],actual['pgid'],actual['sid']) != (identity['start_ticks'],identity['pgid'],identity['sid']):
                        raise RuntimeError('Apptainer process identity changed; refusing cleanup')
            super().cleanup()
            if process is not None and identity:
                event = 'exited' if process.poll() is not None else 'cleanup_incomplete'
                record(self,event,process,identity)
                object.__setattr__(self,'_s2_owned_identity',None)
    return S2OwnedApptainerWorkspace

def cleanup_all():
    errors = []
    for workspace in reversed(ACTIVE):
        try: workspace.cleanup()
        except Exception as exc: errors.append(type(exc).__name__+': '+str(exc))
    if errors: raise RuntimeError('Owned workspace cleanup failed: '+str(errors))
