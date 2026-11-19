"""Explicit small GPU oracle/benchmark. Root owns when/where --run-cuda runs."""
import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-cuda',action='store_true');p.add_argument('--rows',type=int,default=4096)
    p.add_argument('--layers',type=int,choices=(2,4),default=4);p.add_argument('--repeat',type=int,default=20)
    p.add_argument('--uncached-indices',action='store_true')
    p.add_argument('--device',default='cuda:0');a=p.parse_args(argv)
    if not a.run_cuda:p.error('explicit --run-cuda required; do not run against current experiments')
    if a.rows<4 or a.repeat<1:p.error('rows>=4 and repeat>=1 required')
    import torch as t
    return t.inference_mode()(lambda: _run(a,t))()


def _run(a,t):
    from kvpack import KVPackSession,gather_plan_from_cache
    from types import SimpleNamespace
    from kvpack.kernels import launch_scatter
    device=t.device(a.device);t.cuda.set_device(device);layers=tuple(range(3,3+a.layers));n=a.rows
    # Raw 16-bit payloads include nonfinite BF16 patterns: copy must preserve bits.
    halos={l:((t.arange(n*512,device=device,dtype=t.int32)+l*1024)%65536).to(t.int16).view(t.bfloat16).reshape(n,1,512) for l in layers}
    positions=tuple(range(100,100+n));memory=tuple(range(90,100+n))
    src=tuple(range(n-1,-1,-2));dst=tuple(x+10 for x in src)
    session=KVPackSession('rank0/input-native-test/bank1',max_bytes=128*1024**2)
    bank_docs=[dict(key='docA',start=1000,end=1008),dict(key='docB',start=2000,end=2007)]
    banks={(l,d['key']):((t.arange((d['end']-d['start'])*512,device=device,dtype=t.int32)+l*100+d['start'])%65536)
        .to(t.int16).view(t.bfloat16).reshape(d['end']-d['start'],512)for l in layers for d in bank_docs}
    cache=SimpleNamespace(docs=bank_docs,index20_offline_positions=[1000,1007,2001,2005],
        index20_halo_groups=[('docA',[0,1],[0,7]),('docB',[2,3],[1,5])],
        entry=lambda key,kind,layer:banks[layer,key])
    plan=gather_plan_from_cache(cache,layers,owner_key=session.owner_key,stage_key='normalized-gather',group_key='all-halo')
    gather_proofs=[]
    for selected_layers in (layers[:2],layers[-2:]):
        with session.gather_cache(cache,plan.for_layers(selected_layers))as raw:
            for layer in selected_layers:
                reference=t.stack([banks[layer,plan.documents[d].key][r]for d,r in zip(plan.document_indices,plan.local_rows)])
                assert t.equal(raw.normalized(layer).view(t.uint8),reference.view(t.uint8))
            gather_proofs.append(raw.report())
    assert gather_proofs[1]['gather_geometry_cache_hit']is True
    source_rows=t.tensor(src,device=device,dtype=t.int64);destination=t.tensor(dst,device=device,dtype=t.int64)
    if a.uncached_indices:
        lease=session.scatter_halos(layers,halos,src,dst,memory,source_positions=positions,stage_key='stage0',group_key='group0')
    else:
        kinds=['halo'if i in set(dst)else'current'for i in range(len(memory))]
        row_by_destination=dict(zip(dst,src))
        # Original mapping transfer order follows ascending destination rows.
        ordered_dst=sorted(dst);ordered_src=[row_by_destination[i]for i in ordered_dst]
        cached_src=t.tensor(ordered_src,device=device,dtype=t.int64);cached_dst=t.tensor(ordered_dst,device=device,dtype=t.int64)
        mapping=dict(event=dict(memory_positions=memory,source_kinds=kinds,
            source_rows=[row_by_destination.get(i,0)for i in range(len(memory))]),transfers=[('halo',cached_dst,cached_src)])
        geometry=session.geometry_from_mapping(mapping,source_positions=positions,stage_key='stage0',group_key='group0')
        lease=session.scatter_cached_halos(layers,halos,geometry)
    lease.verify_offline()
    holes=[i for i in range(len(memory))if i not in set(dst)]
    cd,pd=holes[::2],holes[1::2]
    for l in layers:
        current=t.full((len(cd),1,512),float(l),device=device,dtype=t.bfloat16)
        previous=t.full((len(pd),1,512),float(-l),device=device,dtype=t.bfloat16)
        actual=lease.assemble_memory(l,current,previous,current_dst=cd,current_src=range(len(cd)),previous_dst=pd,previous_src=range(len(pd)))
        expected=t.empty_like(actual);expected.index_copy_(0,destination,halos[l].index_select(0,source_rows))
        expected.index_copy_(0,t.tensor(cd,device=device),current);expected.index_copy_(0,t.tensor(pd,device=device),previous)
        assert t.equal(actual.view(t.uint8),expected.view(t.uint8))
    try:lease.normalized(layers[0],owner_key='wrong-owner')
    except ValueError:pass
    else:raise AssertionError('owner mismatch accepted')
    pointers=t.tensor([halos[l].data_ptr() for l in layers],device=device,dtype=t.int64)
    target=t.empty_like(lease.memory)
    def fused():launch_scatter(pointers,source_rows,destination,target,len(src))
    def baseline():
        for i,l in enumerate(layers):target[i].index_copy_(0,destination,halos[l].index_select(0,source_rows))
    fused();baseline();t.cuda.synchronize(device)
    def measure(fn):
        values=[]
        for _ in range(a.repeat):
            start,end=t.cuda.Event(enable_timing=True),t.cuda.Event(enable_timing=True)
            start.record();fn();end.record();end.synchronize();values.append(start.elapsed_time(end))
        return statistics.median(values)
    f,b=measure(fused),measure(baseline);proof=lease.report()
    lease.release();session.close();assert session.live_bytes==0
    print(json.dumps(dict(passed=True,bitwise_bfloat16=True,online_current_previous_exact=True,layer_count=a.layers,
        halo_rows=n,moved_rows=len(src),kernel_scope='standalone cross-layer row movement only; not model TTFT',
        fused_median_ms=f,baseline_index_select_index_copy_median_ms=b,row_move_speedup=b/f,
        proof=proof,normalized_cross_layer_gather_proofs=gather_proofs,
        session_live_bytes_after_release=session.live_bytes),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
