"""CPU contracts for the independently registered S12 group17 execution.

Tensor tests exercise actual queue assembly, not native numerical equivalence.
The latter remains a fresh eight-rank GPU admission requirement per group.
"""
import copy
import inspect
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from kv_cache_engram import s1_system_batch as batch
from kv_cache_engram.core import digest
from kv_cache_engram.s1_runtime import S1Controller
from kv_cache_engram.s1_system_runtime import S1SystemController, _validate_schedule

try:
    import torch
except ImportError:
    torch = None


def make_schedule(n, width, docs=None):
    docs = docs or [dict(start=0, end=n, key='metadata-proof')]
    return docs, batch.batch_schedule(docs, n, request_id='metadata-proof',
        phase='metadata-proof', phase_generation=1, input_ids_sha256='a'*64,
        group_chunks=width)


def acknowledgments(schedule):
    return [dict(schema='s1-system-batch-rank-installation-v1', rank=rank,
        schedule_sha256=schedule['sha256'], worker_loop_installed=True,
        native_first3_per_chunk=True, deep_only_on_flush=True,
        empty_deep_ep_on_flush=True) for rank in range(8)]


def case2_documents():
    # Exact registered case2 boundaries; token contents are irrelevant to this
    # geometry test and are independently bound by production input digests.
    boundaries = [2, 16031, 32785, 49775, 66098, 78184, 90297, 100504,
                  110711, 121502, 132285]
    return [dict(start=a, end=z, key='doc'+str(i))
            for i, (a, z) in enumerate(zip(boundaries, boundaries[1:]))]


class Group17ScheduleTests(unittest.TestCase):
    def test_case2_one_flush_preserves_all_seventeen_native_chunks(self):
        _, schedule = make_schedule(132360, 17, case2_documents())
        _validate_schedule(schedule)
        steps = schedule['steps']
        self.assertEqual(len(steps), 17)
        self.assertEqual([s['flush'] for s in steps], [False]*16+[True])
        self.assertEqual([s['selected_rows'] for s in steps],
            [2,128,0,111,17,0,128,0,128,128,0,128,128,128,128,0,203])
        self.assertEqual(sum(s['selected_rows'] for s in steps), 1357)
        self.assertLessEqual(sum(s['selected_rows'] for s in steps), 8192)
        self.assertTrue(all(s['layers']==[0,1,2] for s in steps[:-1]))
        self.assertEqual(steps[-1]['layers'], list(range(40)))
        self.assertFalse(steps[-1]['nonfinal_sample_discarded'])

    def test_full_seventeen_member_flush_and_odd_singleton_tail(self):
        n = 17*8192+1
        docs, schedule = make_schedule(n, 17)
        _validate_schedule(schedule)
        self.assertEqual([i for i,s in enumerate(schedule['steps']) if s['flush']], [16,17])
        self.assertEqual([s['group_start'] for s in schedule['steps']], [0]*17+[17*8192])
        for start, end, count, rows, dst in (
                (16*8192,17*8192,127,139,list(range(12,139))),
                (17*8192,n,1,1,[0])):
            p = batch.plan(docs,n,start,end)
            self.assertEqual(p['selected_rows'],count)
            self.assertEqual(len(p['sampler_output_positions']),rows)
            self.assertEqual(p['sampler_destination_rows'],dst)

    def test_old_group_shapes_and_default_metadata_gate_remain_registered(self):
        for width, expected in ((2,[1,3,4]),(4,[3,4])):
            _, schedule = make_schedule(32769,width)
            _validate_schedule(schedule)
            self.assertEqual([i for i,s in enumerate(schedule['steps']) if s['flush']],expected)
        default = inspect.signature(batch.run_s1_system_metadata_qualification).parameters['group_chunks'].default
        self.assertEqual(default,(2,4))

    def test_unregistered_width_and_forged_worker_schedule_are_rejected(self):
        for width in (True,0,1,3,8,16,18):
            with self.assertRaises(ValueError):make_schedule(32769,width)
        _, schedule = make_schedule(132360,17)
        schedule['steps'][3]['flush']=True
        schedule['sha256']=digest({k:v for k,v in schedule.items() if k!='sha256'})
        with self.assertRaisesRegex(ValueError,'EP schedule'):_validate_schedule(schedule)

    def test_all_eight_rank_receipts_bind_the_new_width(self):
        _, wider = make_schedule(132360,17)
        _, older = make_schedule(132360,4)
        accepted = batch.validate_schedule_agreement(wider,acknowledgments(wider))
        self.assertFalse(accepted['gpu_execution_parity_claim'])
        with self.assertRaises(ValueError):
            batch.validate_schedule_agreement(wider,acknowledgments(older))

    def test_group17_has_its_own_fresh_qualification_flag(self):
        with patch.object(S1Controller,'__init__',return_value=None):
            controller=S1SystemController()
        controller.state='IDLE'
        self.assertEqual(controller.system_qualified,{1:False,2:False,4:False,17:False})
        controller.system_qualified.update({1:True,2:True,4:True})
        controller.configure(system_enabled=True,batch_group_chunks=17)
        with self.assertRaisesRegex(ValueError,'lacks actual per-chunk numerical'):
            controller.reset('r','sensitive',verify_sparse=False)
        with self.assertRaises(ValueError):
            controller.configure(system_enabled=False,batch_group_chunks=17)

    def test_metadata_gate_rejects_invalid_groups_before_cuda_access(self):
        for groups in ((),(17,17),(8,),[True],17,None):
            with self.assertRaises(ValueError):
                batch.run_s1_system_metadata_qualification(None,None,group_chunks=groups)


@unittest.skipUnless(torch is not None,'Torch unavailable in local CPU environment')
class Group17TensorTests(unittest.TestCase):
    def run_queue(self,n,docs):
        _, schedule=make_schedule(n,17,docs)
        queue=batch.CrossChunkDeepBatch(torch_module=torch,documents=docs,
            full_input_tokens=n,request_id='metadata-proof',phase='metadata-proof',
            phase_generation=1,input_ids_sha256='a'*64,group_chunks=17,
            all_rank_receipts=acknowledgments(schedule))
        groups=[];consumed=[];last_cross=None
        for step in schedule['steps']:
            a,z=step['start'],step['end']
            meta,hidden,lazy,cross=batch._metadata_qualification_inputs(torch,torch.device('cpu'),a,z)
            # A distinct source2 index buffer per member proves that the deep
            # frame uses the final native prefix, while preserving all topk rows.
            cross.loaded_prefill_index_kv=(torch.full((z//2,1),z,dtype=torch.int32),
                                           torch.full((z//2,1),z,dtype=torch.int32))
            positions=batch.plan(docs,n,a,z)['selected_positions']
            rows=batch.device_integer_values([p-a for p in positions],meta.positions,torch)
            def fetch(h=hidden):
                consumed.append(True)
                return h+100
            out=queue.push(full_meta=meta,hidden_after_layer2=hidden,lazy_after_layer2=lazy,
                cross_after_layer2=cross,selected_rows=rows,selected_positions_cpu=positions,
                engram14_hook=fetch)
            self.assertEqual(len(consumed),len(queue.schedule['steps'][:queue.next_step]))
            if out is None:continue
            expected=batch.plan(docs,n,out.start,out.end)['selected_positions']
            self.assertEqual(out.positions_cpu,expected)
            self.assertEqual(out.hidden[:,0].tolist(),expected)
            self.assertEqual(out.compact_meta.positions.tolist(),expected)
            self.assertEqual(out.cross.topk_indices[:,0].tolist(),expected)
            self.assertEqual(out.engram14_hook()[:,0].tolist(),[p+100 for p in expected])
            for i,name in enumerate(batch.LAZY):
                self.assertEqual(getattr(out.lazy,name)[0,:,0,0].tolist(),[p+i for p in expected])
            self.assertIs(out.cross.loaded_prefill_kv,cross.loaded_prefill_kv)
            self.assertIs(out.cross.loaded_prefill_index_kv,cross.loaded_prefill_index_kv)
            self.assertEqual(out.full_meta.cu_seqlens_q.tolist(),[0,out.end-out.start])
            self.assertEqual(out.full_meta.cu_seqlens_k.tolist(),[0,out.end])
            self.assertEqual(len(out.member_metadata),len(out.member_chunks))
            self.assertEqual(sum(out.member_selected_counts),len(expected))
            # Check member boundaries in the merged native SWA map; future
            # prefix availability must not change each query's causal keys.
            for lo,hi in out.member_chunks:
                for p in (lo,hi-1):
                    self.assertEqual(out.full_meta.kv_swa_full_index[p-out.start].tolist(),
                        [k-out.start+128 if k>=0 else -1 for k in range(p-127,p+1)])
            groups.append(out);last_cross=cross
        queue.finish()
        self.assertEqual(len(consumed),len(schedule['steps']))
        return groups

    def test_real_case2_geometry_concatenates_1357_rows_once(self):
        groups=self.run_queue(132360,case2_documents())
        self.assertEqual(len(groups),1)
        out=groups[0]
        self.assertEqual(out.selected_count,1357)
        self.assertEqual(len(out.member_chunks),17)
        self.assertEqual(out.prepared['sampler_output_rows'],140)
        self.assertEqual(out.prepared['sample_dst'].tolist(),list(range(140)))
        self.assertEqual(out.prepared['sample_src'].tolist(),list(range(1217,1357)))

    def test_cuda_qualification_geometry_on_cpu_includes_full_group_and_tail(self):
        n=17*8192+1
        groups=self.run_queue(n,[dict(start=0,end=n,key='metadata-proof')])
        self.assertEqual([len(g.member_chunks) for g in groups],[17,1])
        self.assertEqual([g.selected_count for g in groups],[127,1])
        self.assertEqual([g.prepared['sampler_output_rows'] for g in groups],[139,1])
        self.assertEqual(groups[0].prepared['sample_dst'].tolist(),list(range(12,139)))
        tail=groups[-1]
        self.assertEqual(tail.full_meta.positions_c['2'].tolist(),[0])
        self.assertEqual(tail.full_meta.slot_mapping_store_ca['2'].tolist(),[-1])
        self.assertEqual(tail.prepared['sample_src'].tolist(),[0])
        self.assertEqual(tail.prepared['sample_dst'].tolist(),[0])


if __name__=='__main__':unittest.main()
