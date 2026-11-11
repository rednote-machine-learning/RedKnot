"""Read-only process identity snapshots and exact owned-process termination."""
from __future__ import annotations
import hashlib,json,os,pathlib,signal,subprocess,time

def identity(pid):
    p=pathlib.Path('/proc')/str(pid)
    first=(p/'stat').read_text().rsplit(')',1)[1].split()
    if first[0] in ('Z','X','x'):raise ProcessLookupError(pid)
    raw=(p/'cmdline').read_bytes()
    second=(p/'stat').read_text().rsplit(')',1)[1].split()
    if second[0] in ('Z','X','x'):raise ProcessLookupError(pid)
    if first[1:4]+[first[19]]!=second[1:4]+[second[19]]:raise RuntimeError('Unstable process identity')
    argv=[v.decode(errors='replace') for v in raw.split(b'\0') if v]
    return {'pid':pid,'ppid':int(second[1]),'pgid':int(second[2]),'sid':int(second[3]),'start_ticks':int(second[19]),'cmdline_bytes_sha256':hashlib.sha256(raw).hexdigest(),'argv_redacted':['SESSION_API_KEY=<redacted>' if v.startswith('SESSION_API_KEY=') else v for v in argv],'exe':os.readlink(p/'exe'),'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}

def same_process(expected,actual,*,argv=True):
    keys=['pid','start_ticks','pgid','sid','boot_id']
    if argv:keys+=['cmdline_bytes_sha256']
    return all(expected.get(k)==actual.get(k) for k in keys)

def is_live(expected,*,argv=False):
    try:actual=identity(expected['pid'])
    except (FileNotFoundError,ProcessLookupError):return False
    if not same_process(expected,actual,argv=argv):raise RuntimeError('Owned PID identity changed; refusing operation')
    return True

def signal_exact(expected,signum,*,argv=True):
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):raise RuntimeError('Linux pidfd required')
    try:fd=os.pidfd_open(expected['pid'])
    except ProcessLookupError:return False
    try:
        if not is_live(expected,argv=argv):return False
        signal.pidfd_send_signal(fd,signum)
        return True
    finally:os.close(fd)

def has_marker(pid,marker):
    try:values=(pathlib.Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
    except (FileNotFoundError,ProcessLookupError):return False
    return ('REDKNOT_GLM_RUN_ID='+marker).encode() in values

def live_marked_identity(expected,marker):
    """Bracket /proc/environ with stable live identity reads.

    A compiler may exit after enumeration, and Linux then removes/empties its
    environ. Absence only proves lost ownership if the SAME process is still
    alive after the read. Never authorize a stable live process without marker.
    """
    try:before=identity(expected['pid'])
    except (FileNotFoundError,ProcessLookupError):return None
    if not same_process(expected,before,argv=False):
        raise RuntimeError('Owned PID identity changed before marker read; refusing operation')
    present=has_marker(expected['pid'],marker)
    try:after=identity(expected['pid'])
    except (FileNotFoundError,ProcessLookupError):return None
    if not same_process(expected,after,argv=False):
        raise RuntimeError('Owned PID identity changed after marker read; refusing operation')
    if not present:
        diagnostic={key:after[key] for key in ('pid','ppid','pgid','sid','start_ticks','cmdline_bytes_sha256','exe')}
        raise RuntimeError('Original lifecycle marker absent from stable live process: '+json.dumps(diagnostic,sort_keys=True))
    return after

def authenticate_cohort_member(coordinator,candidate,marker_present):
    if candidate['pid']==coordinator['pid']:raise RuntimeError('Cannot clean coordinator itself')
    if (candidate['pgid'],candidate['sid'])!=(coordinator['pgid'],coordinator['sid']):raise RuntimeError('Process outside original lifecycle session')
    if candidate['start_ticks']<coordinator['start_ticks'] or not marker_present:raise RuntimeError('Unowned process in original lifecycle session')

def cohort_members(coordinator,marker):
    """At a quiescent arm boundary, find reparented members of our original cohort."""
    if not is_live(coordinator,argv=False):raise RuntimeError('Coordinator identity changed')
    result=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==coordinator['pid']:continue
        try:value=identity(int(p.name))
        except (FileNotFoundError,ProcessLookupError):continue
        if (value['pgid'],value['sid'])!=(coordinator['pgid'],coordinator['sid']):continue
        value=live_marked_identity(value,marker)
        if value is None:continue
        authenticate_cohort_member(coordinator,value,True)
        result.append(value)
    return result

def descendants(parent,marker):
    """Authorize descendants while ancestry and inherited lifecycle marker are live."""
    if not is_live(parent,argv=False):return []
    items={}
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:info=identity(int(p.name))
        except (FileNotFoundError,ProcessLookupError):continue
        if info['pgid']==parent['pgid'] and info['sid']==parent['sid']:items[info['pid']]=info
    owned={parent['pid']};found=[]
    while True:
        new=[v for pid,v in items.items() if pid not in owned and v['ppid'] in owned]
        if not new:break
        for value in new:
            current=live_marked_identity(value,marker)
            # Remove the stale snapshot even when it exited, otherwise the next
            # ancestry pass would repeatedly rediscover that dead PID.
            owned.add(value['pid'])
            if current is not None:found.append(current)
    return found

def workspace_members(leader):
    """A recorded fresh Apptainer session authorizes only its stable session members."""
    try:current=identity(leader['pid'])
    except (FileNotFoundError,ProcessLookupError):current=None
    if current is not None and not same_process(leader,current,argv=False):
        raise RuntimeError('Recorded workspace leader PID was reused')
    members=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:v=identity(int(p.name))
        except (FileNotFoundError,ProcessLookupError):continue
        if v['pgid']==leader['pid'] and v['sid']==leader['pid']:
            if v['start_ticks']<leader['start_ticks']:raise RuntimeError('Workspace session predates recorded owner')
            members.append(v)
    return members

def stop_exact_set(identities,grace=8):
    pending={x['pid']:x for x in identities};actions=[]
    for value in pending.values():
        if signal_exact(value,signal.SIGTERM,argv=False):actions.append({'pid':value['pid'],'signal':'SIGTERM'})
    limit=time.monotonic()+grace
    while time.monotonic()<limit and any(is_live(x) for x in pending.values()):time.sleep(.1)
    for value in pending.values():
        if signal_exact(value,signal.SIGKILL,argv=False):actions.append({'pid':value['pid'],'signal':'SIGKILL'})
    limit=time.monotonic()+3
    while time.monotonic()<limit and any(is_live(x) for x in pending.values()):time.sleep(.1)
    survivors=[x for x in pending.values() if is_live(x)]
    if survivors:raise RuntimeError('Owned processes survived exact cleanup')
    return actions

def gpu_snapshot(expected_uuids,*,deadline_monotonic=None):
    def query(fields):
        command=['nvidia-smi',fields,'--format=csv,noheader,nounits']
        timeout=10 if deadline_monotonic is None else min(10,deadline_monotonic-time.monotonic())
        if timeout<=0:raise subprocess.TimeoutExpired(command,0)
        return subprocess.check_output(command,text=True,timeout=timeout)
    gpus=[]
    for line in query('--query-gpu=index,uuid,memory.used').splitlines():
        index,uuid,memory=[v.strip() for v in line.split(',')];gpus.append({'index':int(index),'uuid':uuid,'memory_mib':int(memory)})
    if {str(g['index']):g['uuid'] for g in gpus}!=expected_uuids:raise RuntimeError('Pinned GPU inventory changed')
    apps=[]
    for line in query('--query-compute-apps=pid,gpu_uuid,used_gpu_memory').splitlines():
        if not line.strip():continue
        pid,uuid,memory=[v.strip() for v in line.split(',')];apps.append({'pid':int(pid),'gpu_uuid':uuid,'memory_mib':memory})
    return {'observed_unix':time.time(),'gpus':gpus,'compute_apps':apps,'idle':not apps and all(g['memory_mib']==0 for g in gpus)}
