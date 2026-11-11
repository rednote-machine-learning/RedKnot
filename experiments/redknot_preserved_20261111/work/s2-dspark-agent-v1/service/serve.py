"""Explicit live Full/S2 server. No GPU import or launch on import/check-only.

Run only under the original bounded GPU lifecycle with ready/release arguments.
The service owns exactly its eight spawned worker handles. The external
supervisor remains responsible for the overall experiment and holder restore.
"""
from __future__ import annotations

import argparse
import functools
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import threading
import time

from .checkpoint_codec import CheckpointCodec
from .config import validate_config
from .native_runtime import NativeInitializer, load_runtime, runtime_args, seal, write_json
from .online_service import ChatService, LiveWorkerPool, make_http_server, rank_worker


def worker_entry(initializer, commands, events, rank):
    rank_worker(rank, commands[rank], events, initializer)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--arm', choices=('full','s2','prefix'), required=True)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--run', action='store_true')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--ready-file', type=Path)
    p.add_argument('--release-file', type=Path)
    p.add_argument('--absolute-deadline-unix', type=float,
        help='shared outer-round deadline; changing arms does not extend it')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = json.loads(args.config.read_text())
    preflight = validate_config(config)
    from tokenizers import Tokenizer
    codec = CheckpointCodec(config['encoder_path'], Tokenizer.from_file(config['tokenizer_path']),
        thinking_mode=config['thinking_mode'], reasoning_effort=config['reasoning_effort'])
    rendered = codec.render(dict(messages=[dict(role='user',content='Reply briefly.')]))
    if args.check_only:
        print(json.dumps(dict(status='cpu_preflight_passed', preflight=preflight,
            sample_render_input_tokens=len(rendered['input_ids']), gpu_initialized=False)))
        return 0
    if not args.run or not all((args.output_dir,args.ready_file,args.release_file)):
        raise ValueError('live GPU service needs --run and original-supervisor output/ready/release paths')
    if args.absolute_deadline_unix is None or args.absolute_deadline_unix <= time.time():
        raise ValueError('future shared outer-round absolute deadline required')
    if args.absolute_deadline_unix-time.time() > 7200+600:
        raise ValueError('service cannot invent a new extended GPU round')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preflight.json', preflight)
    write_json(output/'service-config.json', config)
    base, _, _, _ = load_runtime(config)
    gate = runtime_args(config, output)
    gate.cpu_ready_file, gate.release_file = args.ready_file, args.release_file
    gate.allow_gpu_init, gate.gate_timeout = False, 600
    base.wait_for_release(gate, config['sha256'])
    release = json.loads(args.release_file.read_text())
    if args.absolute_deadline_unix > release['released_unix']+7200:
        raise ValueError('service deadline exceeds7200 seconds after actual release')
    if time.time() >= args.absolute_deadline_unix:
        raise TimeoutError('shared deadline expired before native initialization')
    # Recheck source/data bytes after authenticated release, before imports.
    validate_config(config)
    for key, value in (('RANK','0'),('WORLD_SIZE','1')):
        if key in os.environ and os.environ[key] != value:
            raise ValueError('single-node native launch requires '+key+'='+value)
        os.environ[key] = value
    context = multiprocessing.get_context('spawn')
    commands, events = [context.Queue() for _ in range(8)], context.Queue()
    initializer = NativeInitializer(config, args.arm, str(output))
    records, server, stopping = [], None, threading.Event()
    errors, old_handlers = [], {}
    started = time.time()
    # No global PID/process search: terminate only handles created below.
    from quality_process_group import _cleanup_owned, _snapshot

    def stop(signum, frame):
        stopping.set()

    for signum in (signal.SIGINT,signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, stop)
    write_lock = threading.Lock()
    journal = (output/'requests.jsonl').open('x', buffering=1)

    def record(value):
        with write_lock:
            journal.write(json.dumps(seal(value), ensure_ascii=False, allow_nan=False)+'\n')
            journal.flush()

    try:
        for rank in range(8):
            process = context.Process(target=worker_entry, args=(initializer,commands,events,rank),
                name=f's2-native-{args.arm}-rank-{rank}', daemon=False)
            item = dict(rank=rank,process=process,started=False,start_attempted=True,actions=[])
            records.append(item)
            process.start()
            item['started'] = True
        def alive():
            if stopping.is_set() or time.time() >= args.absolute_deadline_unix:
                return False
            return len(records) == 8 and all(r['process'].is_alive() for r in records) and not any(
                (output/f'failure.rank{r}.json').exists() for r in range(8))
        pool = LiveWorkerPool(commands, events, workers_alive=alive, timeout_s=config['request_timeout_s'])
        ready = pool.wait_ready(config['startup_timeout_s'])
        write_json(output/'service-ready.json', seal(dict(schema='s2-online-service-ready-v1',
            ready=True, pid=os.getpid(), config_sha256=config['sha256'], arm=args.arm, workers=ready,
            host=config['host'], port=config['port'], ready_unix=time.time(),
            absolute_deadline_unix=args.absolute_deadline_unix)))
        service = ChatService(pool, codec, arm=args.arm, model_name=config['model_name'], record=record)
        server = make_http_server(service, host=config['host'], port=config['port'])
        server.timeout = .5
        # Request threads do not own native workers and must not delay bounded
        # process cleanup after a client disconnect or outer stop instruction.
        server.daemon_threads = True
        while not stopping.is_set() and time.time() < args.absolute_deadline_unix:
            if not alive() or pool.failed:
                raise RuntimeError('native pool failed: '+str(pool.failed))
            server.handle_request()
    except BaseException as error:
        errors.append(repr(error))
        raise
    finally:
        stopping.set()
        if server is not None:
            server.server_close()
        for command in commands:
            command.put(None)
        grace = time.monotonic()+3
        while any(r['process'].is_alive() for r in records) and time.monotonic() < grace:
            for item in records:
                item['process'].join(timeout=0)
            time.sleep(.1)
        survivors = _cleanup_owned(records, 3.0, 3.0, .1)
        write_json(output/'service-terminal.json', seal(dict(schema='s2-online-service-terminal-v1',
            status='failed' if errors or survivors else 'stopped', arm=args.arm,
            config_sha256=config['sha256'], started_unix=started, finished_unix=time.time(),
            absolute_deadline_unix=args.absolute_deadline_unix, errors=errors,
            workers=[_snapshot(r) for r in records], survivors=survivors,
            original_supervisor_owns_holder_restoration=True)))
        journal.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        for command in commands:
            command.cancel_join_thread()
            command.close()
        events.cancel_join_thread()
        events.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
