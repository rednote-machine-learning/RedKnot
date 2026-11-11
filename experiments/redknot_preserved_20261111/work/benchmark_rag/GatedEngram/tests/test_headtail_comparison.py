"""Single hook ownership, isolated policy admission, and shared-bank semantics."""
import importlib.util
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from kv_cache_engram.core import ValidationError
from kv_cache_engram.headtail_comparison import ComparisonController
from kv_cache_engram.headtail_link import HeadTailLinkController
from kv_cache_engram.index_hit20 import IndexHit20Controller
from kv_cache_engram.tail_link import TailLinkController


def controller():
    layers = [NS(layer_idx=i,self_attn=NS(yoco_swa_share_hidden=False,use_swa=True,
        has_packed_kv=i>=2,is_sparse=i>=2,kv_compress_ratio=2 if i>=2 else 0,
        layer_cfg={'index_topk':512 if i>=2 else None})) for i in range(40)]
    return ComparisonController(layers,model_sha256='a'*64,execution_sha256='b'*64,tokenizer_sha256='c'*64)


class ComparisonPolicyTests(unittest.TestCase):
    def test_one_inherited_install_no_second_policy_owner(self):
        c = controller()
        with patch.object(IndexHit20Controller,'install',return_value=c) as install:
            self.assertIs(c.install(),c)
        install.assert_called_once_with()
        self.assertIs(c.native_yoco_swa_share_hidden,False)
        self.assertNotIn('static_controller',c.__dict__)

    def test_runtime_diagnostic_topology_mismatch_rejected_before_install(self):
        for layer,field,value in [(0,'has_packed_kv',True),(2,'kv_compress_ratio',1),
                                  (2,'is_sparse',False),(0,'use_swa',False)]:
            c=controller();setattr(c.layers[layer].self_attn,field,value)
            with patch.object(IndexHit20Controller,'install') as install:
                with self.assertRaises(ValidationError):c.install()
                install.assert_not_called()
        c=controller();c.layers[2].self_attn.layer_cfg['index_topk']=256
        with patch.object(IndexHit20Controller,'install') as install:
            with self.assertRaises(ValidationError):c.install()
            install.assert_not_called()

    def test_arm_mode_matrix_rejects_accidental_algorithm_substitution(self):
        allowed = [('recomputed','native'),('tail128_link','sensitive'),('tail128_link','reference_sparse'),
            ('headtail128_link','sensitive'),('headtail128_link','reference_sparse'),
            ('qualification','qualified_native'),('qualification','all_selected'),('offline','capture_independent')]
        for arm,mode in allowed: ComparisonController._validate_arm_mode(arm,mode)
        for arm in ('recomputed','tail128_link','headtail128_link','qualification','offline','tail128_dynamic'):
            for mode in ('native','sensitive','reference_sparse','all_selected','capture_independent'):
                if (arm,mode) not in allowed:
                    with self.assertRaises(ValidationError): ComparisonController._validate_arm_mode(arm,mode)

    def test_same_policy_qualification_survives_rotation_but_never_cross_admits(self):
        c = controller(); c._bind_policy('tail128_link')
        c.sparse_reference={(0,8192,3):'static'}; c.sparse_reference_input='static-input'; c.sparse_qualified=True
        c._bind_policy('headtail128_link')
        self.assertEqual(c.sparse_reference,{})
        self.assertIsNone(c.sparse_reference_input); self.assertFalse(c.sparse_qualified)
        c.sparse_reference={(0,8192,3):'headtail'}; c.sparse_reference_input='headtail-input'; c.sparse_qualified=True
        c._bind_policy('tail128_link')
        self.assertEqual(c.sparse_reference,{(0,8192,3):'static'})
        self.assertEqual(c.sparse_reference_input,'static-input'); self.assertTrue(c.sparse_qualified)
        c._bind_policy('headtail128_link')
        self.assertEqual(c.sparse_reference_input,'headtail-input')

    def test_phase_clear_preserves_shared_immutable_bank_and_policy_proofs(self):
        c = controller(); entry=object(); c.bank[('doc',3)] = entry
        c._bind_policy('tail128_link'); c.sparse_qualified=True
        c.restored[3]=object(); c.nondoc[3]=object(); c.boundaries[3]=object(); c.ring[3]=object()
        c._clear_phase()
        self.assertIs(c.bank[('doc',3)],entry)
        self.assertTrue(c.sparse_qualified)
        for name in ('restored','nondoc','boundaries','ring'): self.assertEqual(getattr(c,name),{})
        self.assertIsNone(c.staging)

    def test_active_or_failed_phase_cannot_switch_arm(self):
        for state in ('RUNNING','FAILED','CLOSED'):
            c=controller();c.state=state
            with self.assertRaises(ValidationError):
                c.reset('r','native',arm='recomputed',input_ids=[1],phase='p')
            self.assertIsNone(c.arm)

    def test_dispatch_uses_one_owner_with_two_unmodified_policy_methods(self):
        c=controller();c.mode='sensitive';c.arm='tail128_link'
        args=(None,)*6
        with patch.object(TailLinkController,'_selected_attention',return_value='static') as static, \
             patch.object(HeadTailLinkController,'_selected_attention',return_value='headtail') as dynamic:
            self.assertEqual(c._selected_attention(*args),'static')
            static.assert_called_once_with(c,*args);dynamic.assert_not_called()
            c.arm='headtail128_link'
            self.assertEqual(c._selected_attention(*args),'headtail')
            dynamic.assert_called_once_with(*args)

    def test_static_refresh_is_always_noop_and_other_arm_cannot_use_it(self):
        c=controller();c.mode='sensitive';c.arm='tail128_link'
        self.assertEqual(c._refresh_document_tails(*(None,)*8),[])
        c.arm='headtail128_link'
        with self.assertRaises(ValidationError): c._refresh_document_tails(*(None,)*8)

    def test_native_report_exposes_no_active_candidate_despite_resident_bank(self):
        c=controller();c.mode='native';c.arm='recomputed';c.native_yoco_swa_share_hidden=False
        c.bank[('doc',3)]={'scope':{'layer':3}}
        with patch.object(HeadTailLinkController,'finish',return_value={'identity_check_count':80}):
            r=c.finish()
        self.assertEqual(r['identity_check_count'],80)
        self.assertEqual((r['head_tokens'],r['tail_tokens']),(0,0))
        self.assertIsNone(r['dynamic_refresh_start_layer'])
        self.assertFalse(r['cached_document_kv_consumed'])
        self.assertEqual(r['offline_bank_bytes'],32768*512*2)
        self.assertEqual(r['cache_storage_policy'],'shared_gpu_full_canonical_bank_all_arms')

    def test_shared_full_bank_is_bound_across_native_static_dynamic_rotations(self):
        c=controller();c.docs=[dict(key=str(i)) for i in range(8)]
        c.layers[0].self_attn.q_a_kv_swa_proj=NS(weight=NS(device='cuda:0'))
        c.bank={(d['key'],layer):dict(tensor=NS(device='cuda:0'),version=0,scope={'layer':layer,'key':d['key']})
                for d in c.docs for layer in range(40)}
        c._entry=lambda key,layer:c.bank[(key,layer)]
        for arm in ('recomputed','tail128_link','headtail128_link'):
            c.arm=arm;c._bind_case_bank()
        c.bank[('0',3)]['tensor']=NS(device='cuda:0')
        with self.assertRaises(ValidationError):c._bind_case_bank()
        c.bank_case_binding=None;c.bank.pop(('0',4))
        with self.assertRaises(ValidationError):c._bind_case_bank()

    def test_static_finish_dispatch_uses_gpu_schema_without_dma_claim(self):
        c=controller();c.mode='sensitive';c.arm='tail128_link';c.active_policy='tail128_link'
        c.gpu_tail_restore_events=[dict(layer=i) for i in range(3,40)]
        with patch.object(TailLinkController,'finish',return_value={}) as static, \
             patch.object(HeadTailLinkController,'finish') as headtail:
            report=c.finish()
        static.assert_called_once_with(c);headtail.assert_not_called()
        self.assertEqual(report['schema'],'tail-link-gpu-bank-controller-v1')
        self.assertEqual(report['static_tail_source'],'shared_gpu_full_bank_tail128_slice')
        self.assertIsNone(report['staging'])
        self.assertFalse(report['historical_static_cpu_staging_comparable'])


class ComparisonDiagnosticHookTests(unittest.TestCase):
    def test_diagnostic_flag_is_native_only_and_strict_boolean(self):
        for arm,mode,value in [('tail128_link','sensitive',True),('headtail128_link','reference_sparse',True),
                               ('offline','capture_independent',True),('qualification','all_selected',True),
                               ('recomputed','native',1),('recomputed','native',None)]:
            c=controller();c.state='IDLE'
            with self.assertRaises(ValidationError):
                c.reset('r',mode,arm=arm,input_ids=[1],phase='p',diagnostic=value)
            self.assertFalse(c.diagnostic)

    def test_prepare_observation_is_after_original_and_preserves_exact_return_object(self):
        calls=[];result=(object(),object());raw=object()
        class Attn:
            layer_idx=3
        attn=Attn();c=controller();c.diagnostic=True;c.mode='native';c.arm='recomputed'
        def prepare(mod,value,cache,meta,cp):calls.append(('native',value));return result
        c.diagnostic_observer=NS(observe_swa=lambda mod,value,meta:calls.append(('observe',value)))
        c._patch(attn,'_prepare_swa_kv',prepare)
        meta=NS(is_prompt=True,canonical_request_binding={'start':0,'end':8})
        self.assertIs(attn._prepare_swa_kv(raw,None,meta,False),result)
        self.assertEqual(calls,[('native',raw),('observe',raw)])
        for enabled,layer,prompt in ((False,3,True),(True,4,True),(True,3,False)):
            calls.clear();c.diagnostic=enabled;attn.layer_idx=layer;meta.is_prompt=prompt
            self.assertIs(attn._prepare_swa_kv(raw,None,meta,False),result)
            self.assertEqual(calls,[('native',raw)])

    def test_global_index_observer_only_first_three_question_suffix_and_same_return(self):
        calls=[];result=object()
        class Attn:
            layer_idx=2
            def _compute_kv_ca_index(self,*args,**kwargs):calls.append('native');return result
        c=controller();c.diagnostic=True;c.mode='native';c.arm='recomputed';c.docs=[dict(end=10)]
        c.diagnostic_observer=NS(observe_global_index=lambda mod,value,meta:calls.append(('observe',value)))
        attn=Attn()
        with patch.object(IndexHit20Controller,'_install_attention',return_value=None):c._install_attention(attn)
        for end,expected in ((8,['native']),(12,['native',('observe',result)])):
            calls.clear();meta=NS(is_prompt=True,canonical_request_binding={'start':0,'end':end})
            self.assertIs(attn._compute_kv_ca_index(None,None,None,None,meta,None),result)
            self.assertEqual(calls,expected)
        late=Attn();late.layer_idx=3
        with patch.object(IndexHit20Controller,'_install_attention',return_value=None):c._install_attention(late)
        self.assertNotIn('_compute_kv_ca_index',late.__dict__)

    def test_diagnostic_report_is_separate_and_phase_clear_discards_observer(self):
        c=controller();c.mode='native';c.arm='recomputed';c.diagnostic=True
        data={'schema':'test-diagnostic','accuracy_claim':False}
        c.diagnostic_observer=NS(finish=lambda:data)
        with patch.object(HeadTailLinkController,'finish',return_value={}):r=c.finish()
        self.assertIs(r['early_token_diagnostics'],data)
        self.assertTrue(r['diagnostic'])
        c._clear_phase()
        self.assertFalse(c.diagnostic);self.assertIsNone(c.diagnostic_observer)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'requires CPU Torch; remote gate must run unskipped')
class ComparisonTorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch

    def restore_fixture(self,mode='reference_sparse'):
        t=self.t;c=controller();c.t=t;c.arm='tail128_link';c.mode=mode;c.verify_sparse=False;c.phase_generation=3
        c.docs=tuple(dict(start=7+i*32768,end=7+(i+1)*32768,key=str(i)) for i in range(8))
        c.tail_positions_cpu=[p for d in c.docs for p in range(d['end']-128,d['end'])]
        # Each document uses a distinct per-position canonical pattern. Only
        # the last128 are materialized by the static arm, without modifying it.
        entries={d['key']:dict(tensor=(t.arange(32768,dtype=t.int32)%31+i).to(t.bfloat16)[:,None].expand(-1,512))
                 for i,d in enumerate(c.docs)}
        c._entry=lambda key,layer:entries[key]
        calls=[]
        def transform(x,*,norm_args,rope_args,cast_args,out_args):
            calls.append(('transform',len(x),rope_args.positions.clone()))
            return x+(rope_args.positions%7)[:,None]
        def postprocess(mod,x,md):
            return (x+(md.positions%7)[:,None]).reshape(-1,1,512),None
        def identity(a,b):
            if not t.equal(a,b):raise ValidationError('independent restore byte mismatch')
        c.ops=NS(postprocess=postprocess,identity_error=identity,transform=transform,
            RopeArgs=lambda p,*a,**kw:NS(positions=p),CastArgs=lambda *a,**kw:NS())
        mod=NS(layer_idx=3,rotary_emb=NS(cos_sin_cache=None,is_neox_style=False),swa_cache_num_per_channels=128)
        meta=NS(positions=t.tensor([0],dtype=t.int32))
        return c,mod,meta,entries,calls

    def test_static_gpu_restore_slices_full_bank_uses_absolute_rope_and_is_immutable(self):
        c,m,md,entries,calls=self.restore_fixture()
        versions={k:v['tensor']._version for k,v in entries.items()}
        actual=c._restored_tails(m,md)
        expected=self.t.cat([v['tensor'][-128:] for v in entries.values()])
        expected=(expected+(self.t.tensor(c.tail_positions_cpu)%7)[:,None]).reshape(1024,1,512)
        self.assertTrue(self.t.equal(actual,expected))
        self.assertEqual(versions,{k:v['tensor']._version for k,v in entries.items()})
        self.assertEqual(c.tail_restore_events[0]['checks_named'],['gpu_full_bank_tail_slice','absolute_rope_nativecast'])
        self.assertFalse(c.gpu_tail_restore_events[0]['host_to_device_copy'])
        self.assertIs(c._restored_tails(m,md),actual)
        self.assertEqual(len(c.gpu_tail_restore_events),1)

    def test_static_gpu_restore_detects_wrong_rope_cast_and_clears_between_phases(self):
        c,m,md,entries,calls=self.restore_fixture()
        c.ops.postprocess=lambda mod,x,meta:(x.reshape(-1,1,512),None)
        with self.assertRaises(ValidationError):c._restored_tails(m,md)
        c,m,md,entries,calls=self.restore_fixture('sensitive')
        first=c._restored_tails(m,md)
        docs,tails=c.docs,c.tail_positions_cpu
        c._clear_phase();c.docs=docs;c.tail_positions_cpu=tails
        second=c._restored_tails(m,md)
        self.assertIsNot(first,second)
        self.assertTrue(self.t.equal(first,second))

    def test_static_forward_matches_old_tail_math_without_installing_old_owner(self):
        from tests.test_tail_link_controller import TorchForwardTests
        old,m,md,cache,cross,h,calls,modules=TorchForwardTests.fixture(self)
        with patch.dict(sys.modules,modules): expected=old._selected_attention(m,h,cache,md,h,cross)
        old,m,md,cache,cross,h,calls,modules=TorchForwardTests.fixture(self)
        c=controller();c.__dict__.update(old.__dict__);c.arm='tail128_link'
        with patch.dict(sys.modules,modules):actual=c._selected_attention(m,h,cache,md,h,cross)
        self.assertTrue(self.t.equal(actual,expected))
        self.assertEqual(c.swa_events[0]['document_online_rows'],0)

    def test_headtail_dispatch_retains_actual_reference_sparse_semantics(self):
        from tests.test_headtail_link_controller import HeadTailTorchTests
        base,m,md,cache,cross,h,calls,modules=HeadTailTorchTests.forward_fixture(self,'reference_sparse')
        c=controller();c.__dict__.update(base.__dict__);c.arm='headtail128_link'
        with patch.dict(sys.modules,modules):actual=c._selected_attention(m,h,cache,md,h,cross)
        self.assertEqual(actual.shape,h.shape)
        self.assertTrue(c.headtail_memory_events[0]['gpu_union_oracle_checked'])
        self.assertEqual(c.headtail_swa_events[0]['checks'],2)


if __name__=='__main__': unittest.main()
