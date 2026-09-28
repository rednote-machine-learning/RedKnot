#!/usr/bin/env python3
"""Real-tensor component experiment; does not measure full LLM serving.

Run from any cwd. JSON contains every measured repetition, explicit warmups,
checksums/errors, pool-accounted live bytes and actual reserved tensor storage.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/sglang/srt/mem_cache"))
from head_kv import CapacityError, HeadKVManager, HeadPagePool, SegmentPatch, SegmentWrite
from head_kv.attention import paged_attention


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def storage_bytes(tensors):
    storage = {}
    for tensor in tensors:
        backing = tensor.untyped_storage()
        storage[(str(tensor.device), backing.data_ptr())] = backing.nbytes()
    return sum(storage.values())


def timed(device, submit, finish=lambda value: value):
    synchronize(device)
    start = end = None
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
    before = time.perf_counter_ns()
    value = submit()
    submitted = time.perf_counter_ns()
    if end is not None:
        end.record()
    result = finish(value)
    synchronize(device)
    after = time.perf_counter_ns()
    return result, dict(host_submit_ms=(submitted-before)/1e6,
                        synchronized_wall_ms=(after-before)/1e6,
                        cuda_stream_elapsed_ms=start.elapsed_time(end) if start is not None else None)


def make_manager(k, v, page_size, capacity):
    heads, length, dim = k.shape
    pool = HeadPagePool(capacity, page_size, dim, dtype=k.dtype, device=k.device)
    manager = HeadKVManager(pool)
    manager.create_request("base", context_id="benchmark-context")
    manager.update("base", writes=tuple(
        SegmentWrite((0, h, "D"), k[h], v[h], tuple(range(length)), "source-v1")
        for h in range(heads)))
    return manager


def payload(manager, request_id, heads):
    with manager.bind(request_id) as lease:
        pairs = [lease.gather((0, h, "D")) for h in range(heads)]
        return torch.stack([x[0] for x in pairs]), torch.stack([x[1] for x in pairs])


def assert_equal(actual, expected):
    error = max(float((a.float()-e.float()).abs().max()) if a.numel() else 0.0
                for a, e in zip(actual, expected))
    if error:
        raise AssertionError(f"KV mismatch: {error}")
    return error


def repair_spec(k, v, page_size, dirty_head_ratio, dirty_page_ratio, dirty_token_ratio):
    heads, length, _ = k.shape
    pages = math.ceil(length/page_size)
    nh = max(1, math.ceil(heads*dirty_head_ratio))
    np = max(1, math.ceil(pages*dirty_page_ratio))
    nr = max(1, math.ceil(page_size*dirty_token_ratio))
    indices = tuple(p*page_size+r for p in range(np) for r in range(nr) if p*page_size+r < length)
    idx = torch.tensor(indices, device=k.device, dtype=torch.long)
    # The baseline and COW consume these exact same newly computed tensors.
    patches = tuple(SegmentPatch((0, h, "D"), indices,
                                k[h].index_select(0, idx)+0.125,
                                v[h].index_select(0, idx)-0.25,
                                "repaired-v1") for h in range(nh))
    return patches, idx


def eager_repair(k, v, patches, idx):
    outk, outv = k.clone(), v.clone()
    for p in patches:
        outk[p.key[1]].index_copy_(0, idx, p.k)
        outv[p.key[1]].index_copy_(0, idx, p.v)
    return outk, outv


def repair_case(args, heads, pages, ratio, generator):
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    length = pages*args.page_size
    k = torch.randn((heads, length, args.head_dim), generator=generator, dtype=dtype).to(device)
    v = torch.randn((heads, length, args.head_dim), generator=generator, dtype=dtype).to(device)
    patches, idx = repair_spec(k, v, args.page_size, ratio, args.dirty_page_ratio, args.dirty_token_ratio)
    manager = make_manager(k, v, args.page_size, heads*pages*(args.forks+3))
    expected = eager_repair(k, v, patches, idx)
    raw = []
    for rep in range(-args.warmup, args.repeats):
        manager.fork("base", "child")
        pre = manager.stats()
        _, cow = timed(device,
                       lambda: manager.begin_update("child", patches=patches),
                       lambda tx: tx.commit(wait=True))
        assert_equal(payload(manager, "child", heads), expected)
        assert_equal(payload(manager, "base", heads), (k, v))
        post = manager.stats()
        manager.release_request("child")
        synchronize(device); manager.pool.collect()
        dense, eager = timed(device, lambda: eager_repair(k, v, patches, idx))
        assert_equal(dense, expected)
        eager_storage = storage_bytes(dense)
        del dense
        if rep >= 0:
            raw.extend([
                dict(kind="repair", backend="head_page_cow", repetition=rep,
                     copied_payload_bytes=post["copied_bytes"]-pre["copied_bytes"],
                     written_payload_bytes=post["written_bytes"]-pre["written_bytes"], **cow),
                dict(kind="repair", backend="eager_full_clone", repetition=rep,
                     cloned_tensor_storage_bytes=eager_storage,
                     copied_payload_bytes=storage_bytes((k, v)), **eager)])

    # Pure metadata path is exposed independently; no tensor copy in this interval.
    for rep in range(-args.warmup,args.repeats):
        _, control = timed(device, lambda: manager.fork("base", "control"),
                           lambda _: manager.release_request("control"))
        if rep >= 0:
            raw.append(dict(kind="fork_release_metadata", repetition=rep, **control))

    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start_memory = torch.cuda.memory_allocated(device)
    for fork in range(args.forks):
        name = f"fork-{fork}"
        manager.fork("base", name)
        manager.update(name, patches=patches)
    synchronize(device)
    cow_stats = manager.stats()
    cow_peak_delta = (torch.cuda.max_memory_allocated(device)-start_memory) if device.type == "cuda" else None
    for fork in range(args.forks):
        assert_equal(payload(manager, f"fork-{fork}", heads), expected)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start_memory = torch.cuda.memory_allocated(device)
    eager_forks = [eager_repair(k, v, patches, idx) for _ in range(args.forks)]
    synchronize(device)
    eager_peak_delta = (torch.cuda.max_memory_allocated(device)-start_memory) if device.type == "cuda" else None
    for dense in eager_forks:
        assert_equal(dense, expected)
    footprint = dict(
        kind="fork_footprint", forks=args.forks,
        cow_live_payload_bytes=cow_stats["allocated_bytes"],
        cow_peak_live_page_bytes=cow_stats["high_water_pages"]*manager.pool.page_bytes,
        cow_actual_reserved_slab_tensor_bytes=storage_bytes((manager.pool.k, manager.pool.v)),
        eager_actual_tensor_storage_bytes=storage_bytes((k, v, *(x for pair in eager_forks for x in pair))),
        cow_incremental_cuda_peak_allocated_bytes=cow_peak_delta,
        eager_incremental_cuda_peak_allocated_bytes=eager_peak_delta,
        cuda_peak_note="separate incremental allocator peaks; COW slab preallocated before measurement, dense sources/results coexist outside intervals",
        kv_exact_max_error=0.0)
    del eager_forks
    for fork in range(args.forks):
        manager.release_request(f"fork-{fork}")
    manager.release_request("base")
    synchronize(device)
    final = manager.stats()
    assert final["allocated_bytes"] == 0 and final["pinned_pages"] == 0
    return dict(heads=heads, pages_per_head=pages, tokens=length, dirty_head_ratio=ratio,
                dirty_heads=len(patches), dirty_pages_per_head=math.ceil(pages*args.dirty_page_ratio),
                dirty_token_ratio=args.dirty_token_ratio, dirty_rows_per_head=len(idx),
                raw_rows=raw, footprint=footprint, final_pool=final)


def lifecycle_case(args, generator):
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    n = args.page_size+3
    k = torch.randn((1,n,args.head_dim), generator=generator, dtype=dtype).to(device)
    v = torch.randn((1,n,args.head_dim), generator=generator, dtype=dtype).to(device)
    m = make_manager(k, v, args.page_size, 16)
    m.fork("base", "branch")
    before = m.bind("branch")
    a = torch.randn((args.page_size,args.head_dim), generator=generator, dtype=dtype).to(device)
    m.append("branch", (0,0,"D"), a, -a, tuple(range(n,n+len(a))), provenance="append")
    assert_equal(payload(m,"branch",1), (torch.cat((k[0],a))[None],torch.cat((v[0],-a))[None]))
    m.truncate_segment("branch", (0,0,"D"), n-1)
    assert_equal(payload(m,"branch",1), (k[:,:n-1],v[:,:n-1]))
    p = SegmentPatch((0,0,"D"),(0,),a[:1],-a[:1],"cancelled")
    tx = m.begin_update("branch",patches=(p,))
    m.release_request("branch")  # cancellation aborts the prepared write
    assert tx.status == "ABORTED"
    assert_equal(before.gather((0,0,"D")), (k[0],v[0]))
    m.release_request("base")
    pinned = m.stats()
    assert pinned["pinned_pages"] > 0
    before.complete(); synchronize(device)
    final = m.stats()
    assert final["allocated_bytes"] == 0 and final["pinned_pages"] == 0
    return dict(kind="append_truncate_cancel_reclaim", passed=True,
                pinned_after_owner_release=pinned, final_pool=final)


def admission_case(args, generator):
    """Admit real requests until the same bounded slab rejects reservation.

    Inputs/reference tensors are shared fixed experiment fixtures, outside both
    admission budgets. Each backend allocates exactly the same K/V slab size;
    the dense backend physically materializes every child into fresh slab pages.
    No count is calculated from a formula and no unbounded tensor list is used.
    """
    device,dtype=torch.device(args.device),getattr(torch,args.dtype)
    heads,pages=args.admission_heads,args.admission_pages_per_head
    length=pages*args.page_size
    k=torch.randn((heads,length,args.head_dim),generator=generator,dtype=dtype).to(device)
    v=torch.randn((heads,length,args.head_dim),generator=generator,dtype=dtype).to(device)
    patches,idx=repair_spec(k,v,args.page_size,args.admission_dirty_head_ratio,
                            args.dirty_page_ratio,args.dirty_token_ratio)
    expected=eager_repair(k,v,patches,idx)
    full_writes=tuple(SegmentWrite((0,h,"D"),expected[0][h],expected[1][h],
                                   tuple(range(length)),"repaired-v1") for h in range(heads))
    backends=[]
    for backend in ("head_page_cow","eager_independent_full_kv"):
        manager=make_manager(k,v,args.page_size,args.admission_capacity_pages)
        admitted=[];steps=[]
        source_stats=manager.stats()
        actual_reserved_bytes=storage_bytes((manager.pool.k,manager.pool.v))
        assert actual_reserved_bytes==source_stats["reserved_slab_bytes"]
        failure=None
        try:
            while True:
                child=f"admission-{len(admitted)}"
                before=manager.stats()
                if backend=="head_page_cow":
                    manager.fork("base",child)
                else:
                    manager.create_request(child,context_id="benchmark-context")
                try:
                    if backend=="head_page_cow":
                        manager.update(child,patches=patches)
                    else:
                        manager.update(child,writes=full_writes)
                except CapacityError as exc:
                    # Reject the entire child, including its initial shared root.
                    manager.release_request(child);synchronize(device)
                    after=manager.stats()
                    for field in ("live_pages","allocated_bytes","reserved_pages","pending_transactions","requests"):
                        assert before[field]==after[field],(backend,field,before,after)
                    failure=dict(type="CapacityError",message=str(exc),
                                 attempted_child=len(admitted)+1,
                                 pool_before=before,pool_after_rollback=after)
                    break
                admitted.append(child)
                # Full-tensor equality, not a checksum or a calculated capacity.
                assert_equal(payload(manager,child,heads),expected)
                assert_equal(payload(manager,"base",heads),(k,v))
                synchronize(device)
                now=manager.stats()
                steps.append(dict(accepted_child_requests=len(admitted),
                                  live_pages=now["live_pages"],free_pages=now["free_pages"],
                                  allocated_bytes=now["allocated_bytes"],
                                  reserved_pages=now["reserved_pages"],
                                  peak_live_or_retiring_pages=now["high_water_pages"],
                                  cumulative_copied_payload_bytes=now["copied_bytes"],
                                  cumulative_written_payload_bytes=now["written_bytes"],
                                  actual_reserved_slab_tensor_bytes=actual_reserved_bytes,
                                  exact_kv_max_error=0.0))
            # Re-read every survivor after the failed reservation to catch damage.
            for child in admitted:
                assert_equal(payload(manager,child,heads),expected)
            final_live=manager.stats()
        finally:
            for child in admitted:
                manager.release_request(child)
            manager.release_request("base");synchronize(device)
            reclaimed=manager.stats()
            assert reclaimed["allocated_bytes"]==0 and reclaimed["pinned_pages"]==0
        assert failure is not None
        backends.append(dict(backend=backend,accepted_child_requests=len(admitted),
                             accepted_requests_including_source=len(admitted)+1,
                             actual_reserved_slab_tensor_bytes=actual_reserved_bytes,
                             peak_live_page_bytes=final_live["high_water_pages"]*manager.pool.page_bytes,
                             source_pool=source_stats,steps=steps,capacity_rejection=failure,
                             final_live_pool=final_live,final_reclaimed_pool=reclaimed))
        del manager
    return dict(kind="fixed_capacity_admission",heads=heads,pages_per_head=pages,
                capacity_pages=args.admission_capacity_pages,dirty_heads=len(patches),
                dirty_head_ratio=args.admission_dirty_head_ratio,
                dirty_page_ratio=args.dirty_page_ratio,dirty_token_ratio=args.dirty_token_ratio,
                dirty_rows_per_head=len(idx),backends=backends,
                note="Counts observed by real reserve-until-CapacityError. Same fixed physical slab bytes; per-child repair is identical. Inputs/reference and temporary verification buffers are outside both KV slab budgets. This measures resident KV admission capacity, not request throughput or scheduler concurrency.")


def dense_attention(query,k,v,qpos,windows,sinks,num_q_per_kv):
    result=[]
    positions=torch.arange(k.shape[1],device=k.device)
    for qh in range(query.shape[0]):
        h=qh//num_q_per_kv
        mask=positions[None,:] <= qpos[:,None]
        if windows[h] > 0:
            mask &= (positions[None,:] >= qpos[:,None]-windows[h]+1) | (positions[None,:] < sinks[h])
        score=query[qh].float() @ k[h].float().T / math.sqrt(query.shape[-1])
        score.masked_fill_(~mask, -torch.inf)
        weights=torch.nan_to_num(torch.softmax(score,dim=-1),nan=0.0)
        result.append((weights @ v[h].float()).to(query.dtype))
    return torch.stack(result)


def attention_case(args, heads, pages, generator):
    device, dtype=torch.device(args.device),getattr(torch,args.dtype)
    n=pages*args.page_size-3
    k=torch.randn((heads,n,args.head_dim),generator=generator,dtype=dtype).to(device)
    v=torch.randn((heads,n,args.head_dim),generator=generator,dtype=dtype).to(device)
    q=torch.randn((heads*args.q_per_kv,4,args.head_dim),generator=generator,dtype=dtype).to(device)
    m=make_manager(k,v,args.page_size,heads*pages+4)
    lease=m.bind("base")
    desc=lease.descriptor(0,heads)
    # Reverse physical descriptor order; target logical positions retain causality.
    for field in ("page_slots","page_lengths","key_positions"):
        desc[field]=desc[field].flip(1).contiguous()
    qpos=torch.tensor([0,n-3,n-1,n+2],device=device,dtype=torch.int64)
    windows=[0 if h%2==0 else args.page_size+3 for h in range(heads)]
    sinks=[0 if h%2==0 else 2 for h in range(heads)]
    reference=dense_attention(q,k,v,qpos,windows,sinks,args.q_per_kv)
    rows=[]; max_error=0.0
    for rep in range(-args.warmup,args.repeats):
        out,timing=timed(device,lambda:paged_attention(q,**desc,query_positions=qpos,
            num_q_per_kv=args.q_per_kv,windows=windows,sinks=sinks,backend=args.attention_backend))
        error=float((out.float()-reference.float()).abs().max())
        torch.testing.assert_close(out,reference,rtol=0.035 if dtype==torch.bfloat16 else 0.006,
                                   atol=0.025 if dtype==torch.bfloat16 else 0.003)
        max_error=max(max_error,error)
        _,dense_time=timed(device,lambda:dense_attention(q,k,v,qpos,windows,sinks,args.q_per_kv))
        if rep>=0:
            rows.extend([dict(backend="direct_paged_"+args.attention_backend,repetition=rep,**timing),
                         dict(backend="dense_torch_reference",repetition=rep,**dense_time)])
    lease.complete();m.release_request("base");synchronize(device)
    assert m.stats()["allocated_bytes"]==0
    return dict(kind="attention_same_policy",heads=heads,pages_per_head=pages,q_per_kv=args.q_per_kv,
                max_abs_error=max_error,windows=windows,sinks=sinks,descriptor_order="reversed",
                raw_rows=rows,note="dense torch reference is a correctness/ordinary operator baseline, not an optimized FlashAttention speed baseline")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",default="cpu")
    p.add_argument("--dtype",choices=("float32","float16","bfloat16"),default="float32")
    p.add_argument("--heads",type=int,nargs="+",default=[2,8])
    p.add_argument("--pages",type=int,nargs="+",default=[2,8])
    p.add_argument("--dirty-head-ratios",type=float,nargs="+",default=[0.25,1.0])
    p.add_argument("--dirty-page-ratio",type=float,default=0.5)
    p.add_argument("--dirty-token-ratio",type=float,default=0.25)
    p.add_argument("--page-size",type=int,default=16)
    p.add_argument("--head-dim",type=int,default=128)
    p.add_argument("--q-per-kv",type=int,default=4)
    p.add_argument("--forks",type=int,default=4)
    p.add_argument("--admission-heads",type=int,default=8)
    p.add_argument("--admission-pages-per-head",type=int,default=8)
    p.add_argument("--admission-capacity-pages",type=int,default=128)
    p.add_argument("--admission-dirty-head-ratio",type=float,default=0.25)
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=10)
    p.add_argument("--seed",type=int,default=20260928)
    p.add_argument("--attention-backend",choices=("auto","torch","triton"),default="auto")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.warmup<3 or args.repeats<1 or min(*args.heads,*args.pages,args.forks,args.head_dim,args.q_per_kv)<1 or args.page_size<4:
        p.error("warmup >=3; repeats/counts positive; page-size >=4")
    if not all(0<x<=1 for x in [*args.dirty_head_ratios,args.dirty_page_ratio,args.dirty_token_ratio,args.admission_dirty_head_ratio]):
        p.error("dirty ratios must be in (0,1]")
    if min(args.admission_heads,args.admission_pages_per_head)<1 or args.admission_capacity_pages<=args.admission_heads*args.admission_pages_per_head:
        p.error("admission capacity must exceed the positive source page count")
    device=torch.device(args.device)
    if device.type not in ("cpu","cuda"):
        p.error("only cpu/cuda supported")
    if device.type=="cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    generator=torch.Generator(device="cpu").manual_seed(args.seed)
    config=vars(args).copy();config["output"]=str(args.output)
    packages={}
    for name in ("torch","triton","numpy","pytest"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    result=dict(schema_version=1,scope="real-tensor KV manager / attention component validation; not LLM end-to-end",
        config=config,environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        platform=platform.platform(),device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        threads=torch.get_num_threads(),packages=packages),timing_notes=[
            "All timing samples have >=3 excluded warmups; every raw repetition is retained. Admission is one complete fill-to-rejection lifecycle per backend.",
            "CUDA uses synchronize before/after each interval; host_submit includes Python, validation and dispatch.",
            "CUDA event elapsed includes stream idle gaps caused by host dispatch, not pure kernel execution time.",
            "Fork setup/verification/release are outside repair timing; metadata fork+release is measured separately.",
            "Model computation to produce repair payloads is excluded equally from both repair backends.",
            "Live page bytes use actual pool slot counts. Reserved slab bytes use real tensor storage; these are different memory quantities.",
            "CPU RSS peak is not measured; CUDA peaks are incremental allocations in documented intervals."],cases=[],attention=[])
    started=time.time()
    with torch.inference_mode():
        for heads in args.heads:
            for pages in args.pages:
                for ratio in args.dirty_head_ratios:
                    case=repair_case(args,heads,pages,ratio,generator);result["cases"].append(case)
                    print(json.dumps(dict(progress="repair_passed",heads=heads,pages=pages,dirty_head_ratio=ratio)),flush=True)
                result["attention"].append(attention_case(args,heads,pages,generator))
        result["lifecycle"]=lifecycle_case(args,generator)
        result["admission"]=admission_case(args,generator)
    result.update(passed=True,elapsed_seconds=time.time()-started)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+".tmp")
    temporary.write_text(json.dumps(result,indent=2)+"\n");temporary.replace(args.output)
    print(json.dumps(dict(passed=True,output=str(args.output),elapsed_seconds=result["elapsed_seconds"])),flush=True)


if __name__=="__main__":
    main()
