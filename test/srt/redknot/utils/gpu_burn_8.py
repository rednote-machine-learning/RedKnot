#!/usr/bin/env python3
"""Saturate 8 CUDA devices (RTX PRO 5000 Blackwell) at 100% utilization.

Each GPU is driven by a dedicated process that keeps a deep queue of large
matmul kernels in flight across several CUDA streams. The kernels are submitted
in batches so the GPU work queue never drains between Python iterations, which
is what keeps `nvidia-smi` reporting a steady 100% utilization.

Runs continuously until interrupted (Ctrl-C / SIGTERM). Tune --matrix-size and
--mem-fraction to control how much device memory is used per GPU.
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import signal
import time


def burn(
    device_id: int,
    matrix_size: int,
    duration: float,
    dtype_name: str,
    streams_per_gpu: int,
    batch_per_stream: int,
    mem_fraction: float,
) -> None:
    import torch

    torch.cuda.set_device(device_id)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dtype = getattr(torch, dtype_name)

    # Optionally grow memory footprint by allocating several matrix pairs and
    # rotating through them so a large chunk of HBM stays resident.
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    matrix_bytes = matrix_size * matrix_size * bytes_per_elem

    total_mem = torch.cuda.get_device_properties(device_id).total_memory
    target_bytes = int(total_mem * mem_fraction)

    # Each buffer set needs a, b, c => 3 matrices.
    set_bytes = 3 * matrix_bytes
    num_sets = max(1, target_bytes // set_bytes)
    # Keep at least one set per stream so streams don't fight over buffers.
    num_sets = max(num_sets, streams_per_gpu)

    streams = [torch.cuda.Stream(device=device_id) for _ in range(streams_per_gpu)]

    buffers = []
    for _ in range(num_sets):
        a = torch.randn((matrix_size, matrix_size), device=device_id, dtype=dtype)
        b = torch.randn((matrix_size, matrix_size), device=device_id, dtype=dtype)
        c = torch.empty_like(a)
        buffers.append((a, b, c))

    torch.cuda.synchronize(device_id)

    deadline = None if duration == 0 else time.monotonic() + duration
    iterations = 0
    buf_idx = 0

    try:
        while deadline is None or time.monotonic() < deadline:
            try:
                # Submit a deep batch of work on every stream before syncing so
                # the GPU queue stays saturated and utilization holds at 100%.
                for stream in streams:
                    with torch.cuda.stream(stream):
                        for _ in range(batch_per_stream):
                            a, b, c = buffers[buf_idx]
                            torch.mm(a, b, out=c)
                            buf_idx = (buf_idx + 1) % num_sets
                            iterations += 1
                # Sync only once per batch -> Python overhead is amortized away.
                torch.cuda.synchronize(device_id)
            except RuntimeError as exc:
                # Transient CUDA errors must not let a GPU go idle: log, cool
                # down briefly, then keep hammering so the card stays occupied.
                print(f"gpu {device_id}: transient error, retrying: {exc}", flush=True)
                time.sleep(1.0)
    finally:
        torch.cuda.synchronize(device_id)
        print(
            f"gpu {device_id}: {iterations} matmul iterations "
            f"({num_sets} buffer sets, {streams_per_gpu} streams)",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Burn 8 GPUs at a steady 100% utilization."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="seconds to run; 0 means run until interrupted (default)",
    )
    parser.add_argument(
        "--matrix-size", type=int, default=16384, help="square matrix size per GPU"
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--gpus", type=int, default=8, help="number of CUDA devices to use"
    )
    parser.add_argument(
        "--streams-per-gpu",
        type=int,
        default=4,
        help="concurrent CUDA streams per GPU to keep the queue full",
    )
    parser.add_argument(
        "--batch-per-stream",
        type=int,
        default=8,
        help="matmuls queued per stream before each synchronize",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.5,
        help="fraction of each GPU's memory to occupy with rotating buffers",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if torch.cuda.device_count() < args.gpus:
        raise RuntimeError(
            f"need {args.gpus} CUDA devices, found {torch.cuda.device_count()}"
        )

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    ctx = mp.get_context("spawn")

    def spawn(gpu: int):
        proc = ctx.Process(
            target=burn,
            args=(
                gpu,
                args.matrix_size,
                args.duration,
                args.dtype,
                args.streams_per_gpu,
                args.batch_per_stream,
                args.mem_fraction,
            ),
            daemon=False,
        )
        proc.start()
        return proc

    # gpu_id -> process. Each GPU keeps a live worker; dead workers get respawned.
    workers = {gpu: spawn(gpu) for gpu in range(args.gpus)}
    stop_requested = {"flag": False}

    def stop_children(signum, _frame):
        stop_requested["flag"] = True
        for process in workers.values():
            if process.is_alive():
                process.terminate()

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    print(
        f"burning {args.gpus} GPUs | matrix={args.matrix_size} dtype={args.dtype} "
        f"streams={args.streams_per_gpu} batch={args.batch_per_stream} "
        f"mem_fraction={args.mem_fraction} | duration="
        f"{'infinite' if args.duration == 0 else args.duration} | auto-restart=on",
        flush=True,
    )

    # Supervisor loop: hold all GPUs forever (unless --duration set or signalled),
    # respawning any worker that dies so no card is ever left idle.
    try:
        while not stop_requested["flag"]:
            for gpu, process in list(workers.items()):
                if not process.is_alive():
                    if args.duration == 0:
                        # Permanent occupancy mode: a finished/crashed worker
                        # means the GPU would go idle -> bring it back up.
                        print(
                            f"gpu {gpu}: worker exited (code={process.exitcode}), "
                            "restarting to keep the card occupied",
                            flush=True,
                        )
                        process.join()
                        workers[gpu] = spawn(gpu)
                    else:
                        process.join()
            if args.duration != 0 and all(not p.is_alive() for p in workers.values()):
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        stop_requested["flag"] = True

    for process in workers.values():
        if process.is_alive():
            process.terminate()
    for process in workers.values():
        process.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
