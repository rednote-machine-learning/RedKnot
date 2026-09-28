#!/usr/bin/env python3
"""Independent owner/source/destination processes with real HTTP KV staging.

CPU smoke: python validate_head_kv_distributed_gpu.py --device cpu --output result.json
GPU: ... --device cuda:0 --dtype bfloat16 --output result.json
The source/destination can use different CUDA contexts on the same GPU. This is
a single-physical-host component experiment, not RDMA or multi-host performance.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import warnings

import torch

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/"python/sglang/srt/mem_cache"))
from head_kv import CapacityError,HeadKVManager,HeadPagePool,ReuseProof,SegmentPatch,SegmentWrite
from head_kv.attention import paged_attention
from head_kv.distributed import KVShareClient,KVShareServer,KVShareStore,ShareError
from head_kv.transfer import export_snapshot,import_snapshot

NAMESPACE="head-kv-process-experiment"
CONTRACT="synthetic-position-independent-mha-v1"
SOURCE_CONTEXT="P,D,E"
TARGET_CONTEXT="R,E,D"


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(path)


def sync(device):
    if torch.device(device).type=="cuda": torch.cuda.synchronize(device)


def wait_file(path,timeout,process=None):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        if Path(path).exists(): return json.loads(Path(path).read_text())
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"process {process.pid} exited {process.returncode} before {Path(path).name}")
        time.sleep(0.05)
    raise TimeoutError(f"deadline waiting for {Path(path).name}")


class MeasuredClient(KVShareClient):
    def __init__(self,*args,corrupt_fetch=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.rows=[];self.corrupt_fetch=corrupt_fetch;self.did_corrupt=False

    def _request(self,path,data=None,headers=None,limit=1024*1024):
        start=time.perf_counter_ns()
        raw,response=super()._request(path,data,headers,limit)
        self.rows.append(dict(path_kind="object" if path.startswith("/v1/objects") else "metadata",
                              direction="send" if data is not None else "receive",
                              request_body_bytes=len(data) if data is not None else 0,
                              response_body_bytes=len(raw),elapsed_ms=(time.perf_counter_ns()-start)/1e6))
        if self.corrupt_fetch and path.startswith("/v1/objects/") and not self.did_corrupt:
            self.did_corrupt=True
            raw=bytes([raw[0]^1])+raw[1:]
        return raw,response

    def measurements(self):
        return dict(raw_http_calls=list(self.rows),
                    uploaded_object_payload_bytes=sum(r["request_body_bytes"] for r in self.rows if r["path_kind"]=="object" and r["direction"]=="send"),
                    downloaded_object_payload_bytes=sum(r["response_body_bytes"] for r in self.rows if r["path_kind"]=="object" and r["direction"]=="receive"))


def fixtures(args,device):
    generator=torch.Generator(device="cpu").manual_seed(args.seed)
    dtype=getattr(torch,args.dtype)
    values={}
    for name,pages in (("P",1),("D",3),("E",2),("R",1)):
        shape=(args.heads,pages*args.page_size,args.head_dim)
        values[name]=(torch.randn(shape,generator=generator,dtype=dtype).to(device),
                      torch.randn(shape,generator=generator,dtype=dtype).to(device))
    return values


def make_manager(args,device,capacity):
    return HeadKVManager(HeadPagePool(capacity,args.page_size,args.head_dim,
                                     dtype=getattr(torch,args.dtype),device=device))


def add_source(manager,args,values):
    manager.create_request("source",context_id=SOURCE_CONTEXT,contract=CONTRACT,namespace=NAMESPACE)
    writes=[];offset=0
    for name in ("P","D","E"):
        k,v=values[name]
        for head in range(args.heads):
            writes.append(SegmentWrite((0,head,name),k[head],v[head],tuple(range(offset,offset+len(k[head]))),name+"-source-v1"))
        offset+=k.shape[1]
    manager.update("source",writes=writes)


def hashes(manager,request):
    result={}
    with manager.bind(request) as lease:
        for key in sorted(lease.version.segments):
            k,v=lease.gather(key)
            raw=torch.stack((k,v)).cpu().contiguous().view(torch.uint8).numpy().tobytes()
            result[repr(key)]=hashlib.sha256(raw).hexdigest()
    return result


def expect_code(code,operation):
    try: operation()
    except ShareError as exc:
        if exc.code!=code: raise
        return dict(expected=code,observed=exc.code)
    raise AssertionError(f"expected protocol error {code}")


def source_task(args):
    owner=wait_file(args.run_root/"owner_ready.json",args.timeout)
    device=args.source_device or args.device
    if torch.device(device).type=="cuda": torch.cuda.set_device(device)
    values=fixtures(args,device);manager=make_manager(args,device,args.heads*6)
    client=MeasuredClient(owner["url"],NAMESPACE,timeout=5)
    result={}
    try:
        add_source(manager,args,values)
        before=hashes(manager,"source")
        sync(device);start=time.perf_counter_ns()
        manifest=export_snapshot(manager,"source",client,operation_id="publish-source-process")
        sync(device);elapsed=(time.perf_counter_ns()-start)/1e6
        result=dict(pid=os.getpid(),device=device,manifest=manifest,source_hashes_before=before,
                    stage_and_export_wall_ms=elapsed,source_pool=manager.stats(),http=client.measurements())
        save(args.run_root/"source_ready.json",result)
        wait_file(args.run_root/"source_release.json",args.timeout)
        after=hashes(manager,"source")
        assert after==before,"destination COW changed source process pages"
        result.update(source_hashes_after=after,source_unchanged=True,passed=True)
    finally:
        if "source" in manager._requests: manager.release_request("source")
        sync(device);cleanup=manager.stats()
        assert cleanup["allocated_bytes"]==0 and cleanup["pinned_pages"]==0
        result["cleanup_pool"]=cleanup
    return result


def dense_reference(query,k,v,qpos,windows,sinks,q_per_kv):
    positions=torch.arange(k.shape[1],device=k.device)
    output=[]
    for qh in range(len(query)):
        h=qh//q_per_kv
        visible=positions[None,:]<=qpos[:,None]
        if windows[h]>0:
            visible &= (positions[None,:]>=qpos[:,None]-windows[h]+1) | (positions[None,:]<sinks[h])
        scores=query[qh].float()@k[h].float().T/(query.shape[-1]**0.5)
        scores.masked_fill_(~visible,-torch.inf)
        weights=torch.nan_to_num(scores.softmax(-1),nan=0.0)
        output.append((weights@v[h].float()).to(query.dtype))
    return torch.stack(output)


def destination_task(args):
    owner=wait_file(args.run_root/"owner_ready.json",args.timeout)
    source=wait_file(args.run_root/"source_ready.json",args.timeout)
    device=args.destination_device or args.device
    if torch.device(device).type=="cuda": torch.cuda.set_device(device)
    values=fixtures(args,device);total_pages=args.heads*6
    manager=make_manager(args,device,total_pages*3+args.heads+4)
    manager.create_request("imported",context_id=SOURCE_CONTEXT,contract=CONTRACT,namespace=NAMESPACE)
    client=MeasuredClient(owner["url"],NAMESPACE,timeout=5)
    result=dict(pid=os.getpid(),device=device,manifest=source["manifest"])
    try:
        sync(device);start=time.perf_counter_ns()
        import_snapshot(manager,"imported",client,source["manifest"],holder="destination",ttl_s=30)
        sync(device);result["http_import_and_device_stage_wall_ms"]=(time.perf_counter_ns()-start)/1e6
        result["successful_import_http"]=client.measurements()
        imported_hashes=hashes(manager,"imported")
        assert imported_hashes==source["source_hashes_before"]
        manager.fork("imported","fork")
        manager.create_request("target",context_id=TARGET_CONTEXT,contract=CONTRACT,namespace=NAMESPACE)
        # Target is R,E,D, whereas source is P,D,E. Synthetic K is position-free;
        # this certificate does not claim arbitrary model KV can be relocated.
        offset=args.page_size
        for name in ("E","D"):
            length=values[name][0].shape[1]
            for head in range(args.heads):
                manager.share_segment("fork","target",(0,head,name),(0,head,name),positions=range(offset,offset+length),
                    proof=ReuseProof("certified_transform",name+"-source-v1",TARGET_CONTEXT,
                                     "synthetic-position-independent-identity-transform-v1"))
            offset+=length
        manager.update("target",writes=tuple(SegmentWrite((0,h,"R"),values["R"][0][h],values["R"][1][h],
                             tuple(range(args.page_size)),"R-new-v1") for h in range(args.heads)))
        d_before=manager.version("target").segments[(0,0,"D")].pages
        changed_indices=(1,args.page_size+1)
        index=torch.tensor(changed_indices,device=device)
        new_k=values["D"][0][0].index_select(0,index)+0.125
        new_v=values["D"][1][0].index_select(0,index)-0.25
        before_stats=manager.stats();sync(device);start=time.perf_counter_ns()
        manager.update("target",patches=(SegmentPatch((0,0,"D"),changed_indices,new_k,new_v,"D-target-repaired-v1"),))
        sync(device);result["nonprefix_repair_wall_ms"]=(time.perf_counter_ns()-start)/1e6
        after_stats=manager.stats()
        d_after=manager.version("target").segments[(0,0,"D")].pages
        assert d_after[0]!=d_before[0] and d_after[1]!=d_before[1] and d_after[2]==d_before[2]
        for h in range(1,args.heads):
            assert manager.version("target").segments[(0,h,"D")].pages==manager.version("imported").segments[(0,h,"D")].pages
        assert hashes(manager,"imported")==imported_hashes
        assert hashes(manager,"fork")==imported_hashes
        result["cow"]=dict(changed_kv_heads=1,changed_pages=2,
                           copied_payload_bytes=after_stats["copied_bytes"]-before_stats["copied_bytes"],
                           written_payload_bytes=after_stats["written_bytes"]-before_stats["written_bytes"],
                           source_order=["P","D","E"],target_order=["R","E","D"],
                           imported_and_fork_hashes_unchanged=True,unmodified_heads_and_page_shared=True)
        expected_k=torch.cat([values[x][0] for x in ("R","E","D")],dim=1)
        expected_v=torch.cat([values[x][1] for x in ("R","E","D")],dim=1)
        target_index=index+3*args.page_size
        expected_k[0].index_copy_(0,target_index,new_k);expected_v[0].index_copy_(0,target_index,new_v)
        # Validate every target value, including rows masked out by a particular
        # attention query. Gather exists only in this validation oracle.
        with manager.bind("target") as lease:
            for head in range(args.heads):
                parts=[lease.gather((0,head,name)) for name in ("R","E","D")]
                torch.testing.assert_close(torch.cat([pair[0] for pair in parts]),expected_k[head],rtol=0,atol=0)
                torch.testing.assert_close(torch.cat([pair[1] for pair in parts]),expected_v[head],rtol=0,atol=0)
        result["cow"]["exact_target_kv_max_error"]=0.0
        generator=torch.Generator(device="cpu").manual_seed(args.seed+1)
        query=torch.randn((args.heads*args.q_per_kv,4,args.head_dim),generator=generator,dtype=getattr(torch,args.dtype)).to(device)
        positions=torch.tensor([0,args.page_size-1,3*args.page_size,6*args.page_size-1],device=device,dtype=torch.int64)
        windows=[0 if h%2==0 else args.page_size+3 for h in range(args.heads)]
        sinks=[2]*args.heads
        reference=dense_reference(query,expected_k,expected_v,positions,windows,sinks,args.q_per_kv)
        with manager.bind("target") as lease:
            desc=lease.descriptor(0,args.heads)
            sync(device);start=time.perf_counter_ns()
            actual=paged_attention(query,**desc,query_positions=positions,num_q_per_kv=args.q_per_kv,windows=windows,sinks=sinks)
            sync(device);elapsed=(time.perf_counter_ns()-start)/1e6
        torch.testing.assert_close(actual,reference,atol=0.025 if args.dtype=="bfloat16" else 0.003,
                                   rtol=0.035 if args.dtype=="bfloat16" else 0.006)
        result["attention"]=dict(max_abs_error=float((actual.float()-reference.float()).abs().max()),
                                 synchronized_wall_ms=elapsed,backend="triton" if torch.device(device).type=="cuda" else "torch_paged",
                                 baseline="independent dense same causal/global/local/sink policy",windows=windows,sinks=sinks,
                                 timing_note="single correctness launch including validation/possible JIT; not a throughput benchmark")

        # A real HTTP object response is corrupted at its receiver boundary; the
        # normal KVShareClient content check must reject before publication.
        corrupt=MeasuredClient(owner["url"],NAMESPACE,timeout=5,corrupt_fetch=True)
        manager.create_request("corrupt",context_id=SOURCE_CONTEXT,contract=CONTRACT,namespace=NAMESPACE)
        pre=manager.stats()
        fault=expect_code("CORRUPT",lambda:import_snapshot(manager,"corrupt",corrupt,source["manifest"],holder="corruption-test",max_retries=0))
        sync(device);post=manager.stats()
        assert not manager.version("corrupt").segments and post["pending_transactions"]==0
        assert pre["allocated_bytes"]==post["allocated_bytes"] and pre["pinned_pages"]==post["pinned_pages"]
        assert hashes(manager,"imported")==imported_hashes
        result["corruption"]=dict(**fault,injection="flip first object byte after real HTTP response before normal receiver checksum validation",
                                  http=corrupt.measurements(),pool_before=pre,pool_after=post,partial_root_published=False)

        # Capacity check reaches reserve(): manifest fits total capacity, but a
        # live blocker makes free capacity one page short. No payload may fetch.
        limited=make_manager(args,device,total_pages)
        limited.create_request("blocker",context_id=SOURCE_CONTEXT,contract=CONTRACT,namespace=NAMESPACE)
        limited.update("blocker",writes=(SegmentWrite((0,0,"blocker"),values["P"][0][0],values["P"][1][0],tuple(range(args.page_size)),"blocker"),))
        limited.create_request("oom",context_id=SOURCE_CONTEXT,contract=CONTRACT,namespace=NAMESPACE)
        oom_client=MeasuredClient(owner["url"],NAMESPACE,timeout=5)
        try:
            try: import_snapshot(limited,"oom",oom_client,source["manifest"],holder="oom-test",max_retries=0)
            except CapacityError as exc: oom_message=str(exc)
            else: raise AssertionError("expected real destination CapacityError")
            sync(device);oom_state=limited.stats()
            assert oom_state["live_pages"]==1 and oom_state["reserved_pages"]==0 and oom_state["pending_transactions"]==0
            assert not limited.version("oom").segments
            assert oom_client.measurements()["downloaded_object_payload_bytes"]==0
        finally:
            limited.release_request("oom");limited.release_request("blocker");sync(device)
            oom_cleanup=limited.stats()
            assert oom_cleanup["allocated_bytes"]==0 and oom_cleanup["pinned_pages"]==0
        result["oom"]=dict(observed="CapacityError",message=oom_message,pool_after_rejection=oom_state,
                           cleanup_pool=oom_cleanup,http=oom_client.measurements(),partial_root_published=False)
        result.update(passed=True,imported_hashes=imported_hashes)
        # READY is a promise to retain the complete imported checkpoint until the
        # authority decides. Keep the GPU root resident through this handshake.
        ready=client.rpc("mark_migration_ready",decision_id="move-source-to-destination",
                         target="destination",checkpoint=source["manifest"])
        assert ready["ready"]==1
        save(args.run_root/"destination_ready.json",dict(pid=os.getpid(),manifest=source["manifest"],
             imported_checkpoint_resident=True,pool=manager.stats(),tested=True))
        wait_file(args.run_root/"destination_release.json",args.timeout)
        authority=client.rpc("get_request",request_id="active-request")
        assert (authority["owner"],authority["epoch"])==("destination",2),"cannot release READY checkpoint before committed transfer"
        assert hashes(manager,"imported")==imported_hashes
        result["authority_seen_before_release"]=authority
        result["imported_checkpoint_resident_through_commit"]=True
    finally:
        for name in tuple(manager._requests): manager.release_request(name)
        sync(device);cleanup=manager.stats()
        assert cleanup["allocated_bytes"]==0 and cleanup["pinned_pages"]==0
        result["cleanup_pool"]=cleanup
    return result


def owner_task(args):
    payload_bytes=args.heads*6*args.page_size*args.head_dim*2*torch.empty((),dtype=getattr(torch,args.dtype)).element_size()
    store=KVShareStore(args.run_root/"owner_store",NAMESPACE,payload_bytes*4,max_object_bytes=payload_bytes)
    result={}
    try:
        with KVShareServer(store) as server:
            save(args.run_root/"owner_ready.json",dict(pid=os.getpid(),url=server.url,device="cpu",authority="single_sqlite_no_ha"))
            wait_file(args.run_root/"owner_release.json",args.timeout)
            result=dict(pid=os.getpid(),device="cpu",final_store=store.stats(),passed=True)
    finally: store.close()
    return result


def tensor_paths(value,path="result",seen=None):
    """Check that returning experiment evidence itself cannot retain tensors."""
    seen=set() if seen is None else seen
    if id(value) in seen:return []
    seen.add(id(value))
    if isinstance(value,torch.Tensor):return [path]
    if isinstance(value,dict):
        return [p for key,child in value.items() for p in tensor_paths(child,f"{path}.{key}",seen)]
    if isinstance(value,(list,tuple)):
        return [p for index,child in enumerate(value) for p in tensor_paths(child,f"{path}[{index}]",seen)]
    return []


def cuda_memory_summary(device):
    """Bounded JSON-only allocator evidence; no live tensors/frames are retained."""
    index=torch.device(device).index
    if index is None:index=torch.cuda.current_device()
    summary=dict(device=f"cuda:{index}",allocated_bytes=torch.cuda.memory_allocated(index),
                 reserved_bytes=torch.cuda.memory_reserved(index))
    try:
        segments=[s for s in torch.cuda.memory_snapshot() if s.get("device")==index]
        active=[];state_bytes={};state_counts={}
        for segment in segments:
            for block in segment.get("blocks",[]):
                state=block.get("state","unknown")
                state_bytes[state]=state_bytes.get(state,0)+int(block.get("size",0))
                state_counts[state]=state_counts.get(state,0)+1
                if state.startswith("active"):
                    frames=block.get("frames",[])
                    active.append(dict(state=state,size=int(block.get("size",0)),
                        requested_size=int(block.get("requested_size",0)),
                        frames=[{key:frame.get(key) for key in ("filename","line","name")} for frame in frames[:6]]))
        active.sort(key=lambda block:block["size"],reverse=True)
        summary.update(segment_count=len(segments),block_bytes_by_state=state_bytes,
                       block_counts_by_state=state_counts,largest_active_blocks=active[:16],
                       active_blocks_truncated=max(0,len(active)-16))
    except Exception as exc:
        summary["snapshot_error"]=type(exc).__name__+": "+str(exc)
    return summary


def cuda_tensor_inventory(device):
    """Return metadata only; the temporary GC object list dies before cleanup."""
    index=torch.device(device).index
    if index is None:index=torch.cuda.current_device()
    items=[];storages={}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",FutureWarning)
        for obj in gc.get_objects():
            try:
                if not isinstance(obj,torch.Tensor) or obj.device.type!="cuda" or obj.device.index!=index:
                    continue
                storage=obj.untyped_storage()
                storages[storage.data_ptr()]=storage.nbytes()
                items.append(dict(shape=list(obj.shape),dtype=str(obj.dtype),device=str(obj.device),
                                  tensor_bytes=obj.numel()*obj.element_size(),storage_bytes=storage.nbytes()))
            except (RuntimeError,ReferenceError):
                continue
    return dict(python_gc_visible_tensor_count=len(items),
                unique_storage_bytes=sum(storages.values()),largest_tensors=sorted(items,key=lambda x:x["storage_bytes"],reverse=True)[:16],
                note="Python GC-visible tensor inventory is diagnostic, not a complete census of C++ library workspace ownership.")


def finish_cuda_cleanup(args,result,device):
    """Release this process's documented BLAS cache, then retain strict zero check.

    PyTorch 2.9 keeps a cuBLAS workspace per handle/stream after matmul; ordinary
    empty_cache() does not release it. Our destination's dense oracle uses matmul.
    https://docs.pytorch.org/docs/2.9/notes/cuda.html#cublas-workspaces
    The before/after evidence distinguishes library cache from KV/tensor leaks.
    """
    diagnostic_path=args.run_root/f"{args.role}_memory_cleanup.json"
    diagnostics=dict(stage="started",pytorch=torch.__version__,
        source="https://docs.pytorch.org/docs/2.9/notes/cuda.html#cublas-workspaces",
        result_tensor_paths=tensor_paths(result))
    result["cuda_cleanup"]=diagnostics
    assert not diagnostics["result_tensor_paths"],"experiment result retains tensors"
    with torch.cuda.device(device):
        sync(device)
        diagnostics["gc_collected_before_library_clear"]=gc.collect()
        torch.cuda.empty_cache();sync(device)
        diagnostics["before_library_cache_clear"]=cuda_memory_summary(device)
        diagnostics["python_tensors_before_library_clear"]=cuda_tensor_inventory(device)
        diagnostics["stage"]="before_library_cache_clear"
        save(diagnostic_path,diagnostics)
        clear=getattr(torch._C,"_cuda_clearCublasWorkspaces",None)
        diagnostics["cublas_clear_api_available"]=callable(clear)
        if callable(clear):clear()
        sync(device)
        diagnostics["gc_collected_after_library_clear"]=gc.collect()
        torch.cuda.empty_cache();sync(device)
        diagnostics["after_library_cache_clear"]=cuda_memory_summary(device)
        diagnostics["python_tensors_after_library_clear"]=cuda_tensor_inventory(device)
        before=diagnostics["before_library_cache_clear"]["allocated_bytes"]
        after=diagnostics["after_library_cache_clear"]["allocated_bytes"]
        diagnostics["allocated_bytes_released_by_library_cleanup"]=before-after
        diagnostics["stage"]="finished" if after==0 else "nonzero_allocation_failure"
        save(diagnostic_path,diagnostics)
        result["post_scope_cuda_allocated_bytes"]=after
        result["post_scope_cuda_reserved_bytes"]=diagnostics["after_library_cache_clear"]["reserved_bytes"]
        result["gpu_name"]=torch.cuda.get_device_name(device)
        assert after==0,f"live CUDA allocation after GC, synchronization and BLAS cache release: {after} bytes"


def role_main(args):
    torch.set_num_threads(2)
    device=(args.source_device if args.role=="source" else args.destination_device) or args.device
    result={}
    try:
        with torch.inference_mode(): result={"owner":owner_task,"source":source_task,"destination":destination_task}[args.role](args)
        # Preserve protocol/numerical evidence separately even if process-scope
        # allocator cleanup fails next. This is not the final success marker.
        save(args.run_root/f"{args.role}_protocol_result.json",result)
        gc.collect()
        if args.role!="owner" and torch.device(device).type=="cuda":
            finish_cuda_cleanup(args,result,device)
        result["finished_at"]=time.time()
        save(args.run_root/f"{args.role}_result.json",result)
        return 0
    except BaseException as exc:
        # Do not overwrite successful protocol checks or allocation diagnostics.
        result.update(protocol_passed_before_process_cleanup=bool(result.get("passed")),
                      passed=False,pid=os.getpid(),role=args.role,error_type=type(exc).__name__,
                      error=str(exc),traceback=traceback.format_exc())
        if args.role!="owner" and torch.device(device).type=="cuda":
            try:result["cuda_memory_at_failure"]=cuda_memory_summary(device)
            except Exception as memory_exc:result["cuda_memory_at_failure_error"]=type(memory_exc).__name__+": "+str(memory_exc)
        save(args.run_root/f"{args.role}_result.json",result)
        raise


def orchestrate(args):
    args.run_root=args.run_root or args.output.with_suffix(".processes")
    args.run_root=args.run_root.resolve();args.run_root.mkdir(parents=True,exist_ok=False)
    processes={};logs=[];cleanup=[];started=time.time();result={}
    def launch(role):
        argv=[sys.executable,str(Path(__file__).resolve()),"--role",role,"--run-root",str(args.run_root),
              "--output",str(args.output.resolve()),"--device",args.device,"--dtype",args.dtype,
              "--heads",str(args.heads),"--page-size",str(args.page_size),"--head-dim",str(args.head_dim),
              "--q-per-kv",str(args.q_per_kv),"--seed",str(args.seed),"--timeout",str(args.timeout)]
        for option,value in (("--source-device",args.source_device),("--destination-device",args.destination_device)):
            if value: argv.extend([option,value])
        log=(args.run_root/f"{role}.log").open("xb");logs.append(log)
        process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
        processes[role]=process;return process
    try:
        owner=launch("owner");owner_ready=wait_file(args.run_root/"owner_ready.json",args.timeout,owner)
        source=launch("source");source_ready=wait_file(args.run_root/"source_ready.json",args.timeout,source)
        client=MeasuredClient(owner_ready["url"],NAMESPACE,timeout=5)
        manifest=source_ready["manifest"];digest=hashlib.sha256(b"accepted-token-0").hexdigest()
        authority=client.rpc("create_request",request_id="active-request",owner="source",checkpoint=manifest)
        assert authority["epoch"]==1
        client.rpc("commit_output",request_id="active-request",owner="source",epoch=1,token_index=0,token_digest=digest)
        client.rpc("prepare_migration",request_id="active-request",owner="source",epoch=1,target="destination",checkpoint=manifest,decision_id="move-source-to-destination")
        not_ready=expect_code("NOT_READY",lambda:client.rpc("decide_migration",decision_id="move-source-to-destination",commit=True))
        quiesced=expect_code("QUIESCED",lambda:client.rpc("commit_output",request_id="active-request",owner="source",epoch=1,token_index=1,token_digest=digest))
        destination=launch("destination")
        destination_ready=wait_file(args.run_root/"destination_ready.json",args.timeout,destination)
        assert destination_ready["imported_checkpoint_resident"] and destination.poll() is None
        decision=client.rpc("decide_migration",decision_id="move-source-to-destination",commit=True)
        assert decision["state"]=="COMMITTED"
        authority=client.rpc("get_request",request_id="active-request")
        assert (authority["owner"],authority["epoch"],authority["frontier"])==("destination",2,1)
        fenced=expect_code("FENCED",lambda:client.rpc("commit_output",request_id="active-request",owner="source",epoch=1,token_index=1,token_digest=digest))
        duplicate=client.rpc("commit_output",request_id="active-request",owner="destination",epoch=2,token_index=0,token_digest=digest)
        assert duplicate["status"]=="DUPLICATE"
        committed=client.rpc("commit_output",request_id="active-request",owner="destination",epoch=2,token_index=1,token_digest=digest)
        assert committed["frontier"]==2
        save(args.run_root/"destination_release.json",{"authority_committed":True,"epoch":2})
        destination_result=wait_file(args.run_root/"destination_result.json",args.timeout,destination)
        destination.wait(timeout=30)
        if destination.returncode or not destination_result.get("passed"):
            raise RuntimeError("destination failed: "+str(destination_result))
        client.rpc("finish_request",request_id="active-request",owner="destination",epoch=2)
        save(args.run_root/"source_release.json",{"release":True})
        source_result=wait_file(args.run_root/"source_result.json",args.timeout,source);source.wait(timeout=30)
        assert source.returncode==0 and source_result["passed"] and source_result["source_unchanged"]
        client.retire_manifest(manifest);collection=client.gc();final_store=client.stats()
        assert final_store["used_bytes"]==0 and final_store["inflight_bytes"]==0 and final_store["pinned_objects"]==0
        save(args.run_root/"owner_release.json",{"release":True})
        owner_result=wait_file(args.run_root/"owner_result.json",args.timeout,owner);owner.wait(timeout=30)
        assert owner.returncode==0 and owner_result["passed"]
        result=dict(passed=True,source=source_result,destination=destination_result,owner=owner_result,
                    migration=dict(not_ready_gate=not_ready,source_quiesced=quiesced,decision=decision,authority_after_commit=authority,
                                   destination_ready_resident=destination_ready,
                                   stale_source_fenced=fenced,accepted_output_duplicate=duplicate,new_frontier=committed["frontier"],
                                   note="Control-plane API gates validated. This is not a live decoder/router handoff or replicated authority failover."),
                    cleanup_store=dict(gc=collection,final_stats=final_store))
    except BaseException as exc:
        result=dict(passed=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        # Cooperative signals release GPU owners first, then bounded process exit.
        save(args.run_root/"destination_release.json",{"release":True,"cleanup":True})
        save(args.run_root/"source_release.json",{"release":True})
        save(args.run_root/"owner_release.json",{"release":True})
        for role,process in reversed(list(processes.items())):
            action="already_exited"
            if process.poll() is None:
                try: process.wait(timeout=10);action="cooperative_exit"
                except subprocess.TimeoutExpired:
                    process.terminate();action="terminated_owned_child"
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill();process.wait(timeout=10);action="killed_owned_child"
            cleanup.append(dict(role=role,pid=process.pid,returncode=process.returncode,action=action))
        for log in logs: log.close()
        result.update(scope="single physical host, independent source/destination processes and CUDA contexts, HTTP host staging; not RDMA or cross-physical-host validation",
                      process_cleanup=cleanup,orchestrator_pid=os.getpid(),seed=args.seed,dtype=args.dtype,
                      config=dict(heads=args.heads,page_size=args.page_size,head_dim=args.head_dim,q_per_kv=args.q_per_kv,
                                  source_device=args.source_device or args.device,destination_device=args.destination_device or args.device),
                      environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda),
                      elapsed_seconds=time.time()-started,run_root=str(args.run_root),
                      validity_note="Random synthetic position-independent KV; exact tensor isolation and same-policy attention are checked. Non-prefix reuse validity for an LLM is not established by this experiment.")
        save(args.output,result)
    print(json.dumps(dict(passed=result["passed"],output=str(args.output),pids={k:v.pid for k,v in processes.items()},elapsed_seconds=result["elapsed_seconds"])),flush=True)
    return 0 if result["passed"] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",default="cpu")
    p.add_argument("--source-device");p.add_argument("--destination-device")
    p.add_argument("--dtype",choices=("float32","float16","bfloat16"),default="float32")
    p.add_argument("--heads",type=int,default=4);p.add_argument("--page-size",type=int,default=16)
    p.add_argument("--head-dim",type=int,default=64);p.add_argument("--q-per-kv",type=int,default=2)
    p.add_argument("--seed",type=int,default=20260928);p.add_argument("--timeout",type=float,default=300)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--run-root",type=Path)
    p.add_argument("--role",choices=("owner","source","destination"),help=argparse.SUPPRESS)
    args=p.parse_args()
    def interrupted(signum,_frame):
        raise InterruptedError(f"owned experiment received signal {signum}")
    signal.signal(signal.SIGTERM,interrupted)
    if min(args.heads,args.page_size,args.head_dim,args.q_per_kv)<2 or not 1<=args.timeout<=3600:
        p.error("dimensions >=2 and timeout 1..3600 required")
    for device in [args.device,args.source_device,args.destination_device]:
        if device and torch.device(device).type not in ("cpu","cuda"):p.error("cpu/cuda only")
    if args.role and args.run_root is None:p.error("internal roles require run-root")
    return role_main(args) if args.role else orchestrate(args)


if __name__=="__main__":raise SystemExit(main())
