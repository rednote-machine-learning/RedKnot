"""CPU-only official agent-server workspace admission; uses the actual local SIF.

No model request is made. Must run as real root with the admitted host SDK venv.
Pass real host grader/weight paths to --hidden-path; their contents are never read.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, shlex, sys, tempfile
B=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(B))
from runtime_policy import admitted_environment, workspace_kwargs
import workspace_ownership

PAYLOAD='''import glob,hashlib,json,os,pathlib
p=pathlib.Path('/proc/self/status')
s=dict(line.split(':',1) for line in p.read_text().splitlines() if ':' in line)
paths=HIDDEN_PATHS
r={'uid':os.getuid(),'caps':{k:s[k].strip() for k in ['CapEff','CapPrm','CapBnd','CapAmb']},'no_new_privs':s.get('NoNewPrivs','').strip(),'namespaces':{k:os.readlink('/proc/self/ns/'+k) for k in ['mnt','pid','ipc']},'hidden_paths':{x:os.path.exists(x) for x in paths},'gpu_device_paths':glob.glob('/dev/nvidia*'),'runtime_guard_active':os.getenv('S2_ROUTE_GUARD_ACTIVE'),'host_env_canary_present':'S2_HOST_SECRET_CANARY' in os.environ,'session_api_key_present':bool(os.getenv('SESSION_API_KEY')),'mountinfo':pathlib.Path('/proc/self/mountinfo').read_text()}
p=pathlib.Path('/workspace/cpu-write-probe');p.write_text('workspace write permitted');r['workspace_writable']=p.read_text()=='workspace write permitted';p.unlink()
print(json.dumps(r))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--sif',required=True);p.add_argument('--report-dir',required=True);p.add_argument('--hidden-path',action='append',required=True);p.add_argument('--capture-initial-ordinal',type=int);p.add_argument('--base-environment-proof');args=p.parse_args()
    if args.base_environment_proof and args.capture_initial_ordinal is None:p.error('--base-environment-proof requires --capture-initial-ordinal')
    if os.getuid()!=0:raise RuntimeError('Actual real-root probe required')
    sif=pathlib.Path(args.sif).resolve();assert sif.is_file()
    hidden=[str(pathlib.Path(x).resolve()) for x in args.hidden_path]
    if not all(pathlib.Path(x).exists() for x in hidden):raise ValueError('All hidden-path targets must really exist on host')
    dest=pathlib.Path(args.report_dir).resolve();dest.mkdir(parents=True,exist_ok=False)
    parent_ns={k:os.readlink('/proc/self/ns/'+k) for k in ['mnt','pid','ipc']}
    extension_pin=json.loads((B/'framework-manifest.json').read_text())['sources']['extensions']['commit']
    if os.getenv('EXTENSIONS_REF') not in (None,extension_pin):raise ValueError('Conflicting extensions pin')
    os.environ['EXTENSIONS_REF']=extension_pin
    os.environ['S2_HOST_SECRET_CANARY']='not-forwarded'
    try:
      with tempfile.TemporaryDirectory(prefix='s2-cpu-workspace-') as tmp, admitted_environment() as env_receipt:
        from openhands.workspace import ApptainerWorkspace
        import httpx
        workspace_ownership.configure(dest)
        W=workspace_ownership.make_workspace_class(ApptainerWorkspace,lambda kw:workspace_kwargs(kw,os.getuid()))
        try:
          with W(sif_file=str(sif),mount_dir=tmp,working_dir='/workspace') as w:
            # A protected route must reject both no key and a wrong key.
            no_key=httpx.get(w.host+'/api/conversations/count',timeout=10).status_code
            wrong_key=httpx.get(w.host+'/api/conversations/count',headers={'X-Session-API-Key':'intentionally-invalid-cpu-probe'},timeout=10).status_code
            good_key=httpx.get(w.host+'/api/conversations/count',headers={'X-Session-API-Key':os.environ['SESSION_API_KEY']},timeout=10).status_code
            code=PAYLOAD.replace('HIDDEN_PATHS',repr(hidden))
            result=w.execute_command('/agent-server/.venv/bin/python -c '+shlex.quote(code),timeout=30)
            if result.exit_code!=0:raise RuntimeError('Container CPU probe failed: '+result.stderr)
            inside=json.loads(result.stdout.strip().splitlines()[-1])
            checks={'no_key_rejected':no_key==401,'wrong_key_rejected':wrong_key==401,'correct_key_accepted':good_key==200,'caps_zero':all(int(v,16)==0 for v in inside['caps'].values()),'private_namespaces':all(inside['namespaces'][k]!=parent_ns[k] for k in parent_ns),'no_hidden_paths':not any(inside['hidden_paths'].values()),'no_gpu_devices':not inside['gpu_device_paths'],'clean_environment':not inside['host_env_canary_present'],'guard_loaded_in_actual_server':inside['runtime_guard_active']=='1','workspace_writable':inside['workspace_writable']}
            report={'schema':'s2-actual-apptainer-workspace-cpu-probe-v1','status':'passed' if all(checks.values()) else 'failed','checks':checks,'host_namespaces':parent_ns,'inside':inside,'auth_status_codes':{'missing':no_key,'wrong':wrong_key,'correct':good_key},'sif_sha256':hashlib.file_digest(sif.open('rb'),'sha256').hexdigest(),'runtime':env_receipt,'gpu_or_llm_calls':False,'limitations':['Network namespace is shared; this proves specified filesystem/process/device boundaries, not a hostile-kernel security boundary.','Capabilities and auth are checked in the actual official workspace with the external S2 adapter.']}
            (dest/'workspace-proof.json').write_text(json.dumps(report,indent=2)+'\n')
            if not all(checks.values()):raise RuntimeError('Actual workspace CPU admission failed: '+str(checks))
            if args.capture_initial_ordinal is not None:
                from cpu_capture_initial_v7 import capture
                capture(w,args.capture_initial_ordinal,dest,base_environment_proof=args.base_environment_proof)
        finally:workspace_ownership.cleanup_all()
    finally:os.environ.pop('S2_HOST_SECRET_CANARY',None)
    print(str(dest/'workspace-proof.json'))
if __name__=='__main__':main()
