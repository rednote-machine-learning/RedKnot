"""Apply S2 guard only to the official agent-server Python process, not repo tests."""
import os
import sys
try:
    with open('/proc/self/cmdline', 'rb') as f:
        argv = [v.decode(errors='replace') for v in f.read().split(b'\0') if v]
except OSError:
    argv = []
is_agent_server = any(argv[i:i+2] == ['-m', 'openhands.agent_server'] for i in range(len(argv)-1))
if is_agent_server:
    try:
        from routing_guard import install
        install()
        os.environ['S2_ROUTE_GUARD_ACTIVE'] = '1'
        sys.stderr.write('S2_ROUTE_GUARD_ACTIVE local-only greedy natural-context\n')
    except BaseException as exc:
        sys.stderr.write('S2_ROUTE_GUARD_FATAL ' + type(exc).__name__ + '\n')
        sys.stderr.flush()
        os._exit(78)
