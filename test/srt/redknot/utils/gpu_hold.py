#!/usr/bin/env python3
"""
占卡脚本：在指定 GPU 上持续做矩阵乘法，将利用率维持在接近 100%，
并占满显存，防止 GPU 被其他任务抢占。若子进程异常退出会自动重启。

用法:
    python gpu_hold.py                  # 占用全部可见 GPU
    python gpu_hold.py --gpus 0,1,2     # 只占用指定 GPU
    python gpu_hold.py --mem-frac 0.9   # 每卡占用剩余显存比例，默认 0.9
    python gpu_hold.py --size 8192      # 矩阵边长，越大计算越重

前台运行，按 Ctrl+C 优雅退出所有子进程。
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import multiprocessing as mp


def worker(gpu_id: int, size: int, mem_frac: float, stop_evt):
    # 只让当前进程看到这一张卡，进程内即为 cuda:0
    # 必须在 import torch 之前设置，且父进程也不能提前 import torch，
    # 否则 CUDA context 会被父进程污染，子进程报 "driver too old" 的误导错误。
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # 忽略子进程中的 SIGINT，由主进程通过 stop_evt 统一控制退出
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import torch

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    print(f"[GPU {gpu_id}] CUDA init OK (torch={torch.__version__})", flush=True)

    # 用于持续计算的两个矩阵（float16 省显存，算力占用高）
    a = torch.randn(size, size, device=dev, dtype=torch.float16)
    b = torch.randn(size, size, device=dev, dtype=torch.float16)
    c = torch.empty(size, size, device=dev, dtype=torch.float16)

    # 占满剩余显存（真正"占坑"），采用逐步降低块大小的方式尽力吃满
    ballasts = []
    free_bytes, _ = torch.cuda.mem_get_info(dev)
    target_bytes = int(free_bytes * mem_frac)
    # 预留 256MiB 给 matmul workspace/cublas handle 等
    reserve = 256 * 1024 * 1024
    target_bytes = max(0, target_bytes - reserve)

    remaining = target_bytes
    chunk = 1 * 1024 * 1024 * 1024  # 1 GiB 起步
    while remaining > 0 and chunk >= 16 * 1024 * 1024:  # 最小 16 MiB
        try:
            n_elems = min(remaining, chunk) // 2  # float16 = 2 bytes
            if n_elems <= 0:
                break
            t = torch.empty(n_elems, device=dev, dtype=torch.float16)
            ballasts.append(t)
            remaining -= n_elems * 2
        except RuntimeError:
            # 当前块申请失败，缩小块大小重试
            chunk //= 2

    held_gib = sum(t.numel() * 2 for t in ballasts) / (1024**3)
    print(
        f"[GPU {gpu_id}] 开始占卡: matmul size={size}, 显存 ballast={held_gib:.2f} GiB",
        flush=True,
    )

    while not stop_evt.is_set():
        for _ in range(50):
            torch.matmul(a, b, out=c)
        # 同步一次，确保 kernel 真正执行，避免只排队不执行
        torch.cuda.synchronize()

    del a, b, c
    ballasts.clear()
    torch.cuda.empty_cache()
    print(f"[GPU {gpu_id}] 已退出", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="GPU 占卡脚本")
    p.add_argument(
        "--gpus", type=str, default="", help="逗号分隔的 GPU id，如 0,1,2；默认占用全部"
    )
    p.add_argument("--size", type=int, default=8192, help="矩阵边长")
    p.add_argument(
        "--mem-frac",
        type=float,
        default=0.9,
        help="每卡占用剩余显存比例，默认 0.9",
    )
    p.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        help="心跳打印间隔秒数，默认 30s",
    )
    return p.parse_args()


def spawn_worker(ctx, gid, size, mem_frac, stop_evt):
    pr = ctx.Process(
        target=worker,
        args=(gid, size, mem_frac, stop_evt),
        daemon=False,
    )
    pr.start()
    return pr


def _detect_gpu_count() -> int:
    """通过 nvidia-smi 探测卡数，避免主进程 import torch 污染 CUDA context。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return len([l for l in out.splitlines() if l.strip()])
    except Exception as e:
        print(f"nvidia-smi 探测失败: {e}", file=sys.stderr)
        return 0


def main():
    args = parse_args()

    # 关键：主进程严禁 import torch，否则 spawn 出去的子进程会因 CUDA
    # context 已在父进程建立而报 "driver too old" 的误导性错误。
    total = _detect_gpu_count()
    if total == 0:
        print("未检测到可用 GPU", file=sys.stderr)
        sys.exit(1)

    if args.gpus.strip():
        gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    else:
        gpu_ids = list(range(total))

    print(f"将占用 GPU: {gpu_ids}（共 {len(gpu_ids)} 张），Ctrl+C 退出", flush=True)

    ctx = mp.get_context("spawn")
    stop_evt = ctx.Event()

    # gid -> Process
    procs = {}
    # gid -> (连续失败次数, 上次启动时间)
    fail_cnt = {gid: 0 for gid in gpu_ids}
    start_ts = {}
    MAX_FAIL = 5  # 连续失败 5 次放弃该卡
    for gid in gpu_ids:
        procs[gid] = spawn_worker(ctx, gid, args.size, args.mem_frac, stop_evt)
        start_ts[gid] = time.time()

    dead = set()  # 已放弃重启的卡
    last_hb = time.time()
    try:
        while not stop_evt.is_set():
            # 检查每张卡的子进程，异常退出则自动重启
            for gid in gpu_ids:
                if gid in dead:
                    continue
                p = procs[gid]
                if not p.is_alive():
                    lifetime = time.time() - start_ts[gid]
                    # 存活 > 30s 视为一次正常运行，重置失败计数
                    if lifetime > 30:
                        fail_cnt[gid] = 0
                    fail_cnt[gid] += 1
                    print(
                        f"[GPU {gid}] 子进程退出 (exitcode={p.exitcode}, "
                        f"存活 {lifetime:.1f}s, 连续失败 {fail_cnt[gid]}/{MAX_FAIL})",
                        flush=True,
                    )
                    p.join(timeout=1)
                    if fail_cnt[gid] >= MAX_FAIL:
                        print(
                            f"[GPU {gid}] 连续失败达上限，放弃该卡（检查上方 traceback）",
                            flush=True,
                        )
                        dead.add(gid)
                        continue
                    time.sleep(2)
                    procs[gid] = spawn_worker(
                        ctx, gid, args.size, args.mem_frac, stop_evt
                    )
                    start_ts[gid] = time.time()

            # 全部卡都放弃了，直接退出
            if len(dead) == len(gpu_ids):
                print("所有 GPU 都已放弃，退出主进程", flush=True)
                break

            now = time.time()
            if now - last_hb >= args.heartbeat:
                alive = [gid for gid in gpu_ids if gid not in dead and procs[gid].is_alive()]
                print(
                    f"[心跳] 存活 GPU: {alive} ({len(alive)}/{len(gpu_ids)}), "
                    f"放弃 GPU: {sorted(dead)}",
                    flush=True,
                )
                last_hb = now

            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止所有 GPU 进程...", flush=True)
        stop_evt.set()

    for gid, p in procs.items():
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    print("全部退出完成。", flush=True)


if __name__ == "__main__":
    main()
