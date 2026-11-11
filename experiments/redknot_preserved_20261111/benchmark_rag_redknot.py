#!/usr/bin/env python3
"""Full / S1.2 RAG reproduction on the preserved eight-GPU server.

--check-only validates the selected 64K/128K inputs on CPU. An explicit --run
starts a detached server-side benchmark, with a two-hour GPU limit and original
holder restoration. No action occurs without either flag. Full reports are
written on the server; this launcher does not need to remain open during --run.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SERVER_PACKAGE='/mnt/tidal-alsh01/dataset/redone/RedKnot_DeepInfer_20261028/benchmark_rag_redknot_v1'
SERVER_PYTHON='/tmp/redknot-dsv41-sm90.6ZbGzo/di-venv/bin/python'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length',choices=('64k','128k','both'),default='128k')
    parser.add_argument('--cases',type=int,choices=range(1,11),default=1,help='questions per selected length')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only',action='store_true')
    mode.add_argument('--run',action='store_true')
    args=parser.parse_args(argv)
    if not (args.check_only or args.run):parser.print_help();return 0
    ssh=HERE/'work/ssh-h200.sh'
    if not ssh.is_file():parser.error('The preserved workspace SSH helper is missing: '+str(ssh))
    command=[SERVER_PYTHON,'-B',SERVER_PACKAGE+'/benchmark_rag_redknot.py',
             '--length',args.length,'--cases',str(args.cases)]
    if args.check_only:
        return subprocess.run([str(ssh),'-o','ConnectTimeout=15',shlex.join(command+['--check-only'])]).returncode
    # Detach the complete entry (including final CPU audit), not just inference.
    source='COMMAND='+repr(command+['--run'])+'\nBASE='+repr(SERVER_PACKAGE)+'\n'+r'''
from pathlib import Path
import json,subprocess,tempfile,time
b=Path(BASE);root=b/'launches';root.mkdir(exist_ok=True)
launch=Path(tempfile.mkdtemp(prefix='rag.',dir=root))
log=launch/'entry.log'
with log.open('xb') as out:
 p=subprocess.Popen(COMMAND,cwd=b,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='server_entry_dispatched',pid=p.pid,created_unix=time.time(),command=COMMAND,
 log=str(log),gpu_release_not_yet_verified=True,server_process_independent_of_local_mac=True)
(launch/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''
    return subprocess.run([str(ssh),'-o','ConnectTimeout=15',shlex.join([SERVER_PYTHON,'-B','-'])],input=source,text=True).returncode


if __name__=='__main__':sys.exit(main())
