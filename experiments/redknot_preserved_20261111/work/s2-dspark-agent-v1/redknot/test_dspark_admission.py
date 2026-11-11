"""CPU orchestration regressions; these synthetic objects are not GPU proof."""
import contextlib
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

STAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(STAGE),str(STAGE/'GatedEngram')]
from redknot.decode_evidence import validate_decode_phase
from redknot.gate import verify_sparse_pair
from redknot import test_adapter as fixtures
from redknot.admission import MAX_BANK_BYTES
from service.native_runtime import NativeRuntime, TokenRequest, token_digest


class DSparkAdmissionTests(unittest.TestCase):
    def test_actual_codec_produces_four_controlled_prefix_cases(self):
        try:
            import tokenizers
        except ImportError:
            sys.path.insert(0,str(STAGE.parent/'code-agent-redknot-v1/_deps'))
        asset = STAGE if (STAGE/'NativeSnapshot').is_dir() else STAGE.parent/'s2-agent-v1'
        tokenizer = Path(os.environ.get('S2_TOKENIZER',str(STAGE.parent/'benchmark_rag/tokenizer/tokenizer.json')))
        config = dict(native_repo=str(asset/'NativeSnapshot'),encoder_path=str(asset/'checkpoint-encoding/encoding.py'),
            tokenizer_path=str(tokenizer),thinking_mode='thinking',reasoning_effort=75)
        runtime=NativeRuntime(config,'prefix',Path('/unused'),0)
        class Captured(Exception): pass
        cases=[]
        def capture(value):
            cases.extend(value)
            raise Captured()
        runtime.driver=NS(gather=capture)
        with self.assertRaises(Captured): runtime._qualify_prefix()
        self.assertEqual([len(c['input_ids']) for c in cases],[8194,10000,8194,8194])
        cold,hit,changed,short=[c['input_ids'] for c in cases]
        self.assertEqual(cold[:8192],hit[:8192])
        self.assertNotEqual(cold[:8192],changed[:8192])
        self.assertEqual(cold[:8192],short[:8192])
        self.assertNotEqual(cold[-2:],short[-2:])
        self.assertNotEqual(cases[0]['source']['rendered_source_input_sha256'],
                            cases[2]['source']['rendered_source_input_sha256'])

    def phase(self):
        graph = dict(runner_present=True,captured_graph_sizes=[1],successful_replay_calls=10)
        reports=[]
        for rank in range(8):
            reports.append(dict(rank=rank,profile='dspark',includes_native_idle=True,
                before={k:dict(graph,successful_replay_calls=0) for k in ('main','dflash')},
                after={k:dict(graph) for k in ('main','dflash')},
                successful_replay_deltas=dict(main=10,dflash=10)))
        return dict(decode_graph_evidence=reports,native_idle_bypass_installed=False,
            scheduled_rows_reports=[dict(decode_steps=10 if rank==0 else 0) for rank in range(8)],
            record=dict(output_ids=[2]*32),native_cache_storage_evidence=[dict(rank=rank,
                baseline_sha256='a'*64,before_sha256='a'*64,after_sha256='a'*64,
                native_cache_storage_unchanged=True) for rank in range(8)])

    def test_actual_decode_requires_calls_not_flags_and_stable_storage(self):
        self.assertEqual(validate_decode_phase(self.phase())['successful_dflash_replays'],10)
        for field in ('no_replay','no_decode','changed_storage','old_idle'):
            phase=self.phase()
            if field=='no_replay':
                item=phase['decode_graph_evidence'][0]
                item['after']['dflash']['successful_replay_calls']=0
                item['successful_replay_deltas']['dflash']=0
            elif field=='no_decode': phase['scheduled_rows_reports'][0]['decode_steps']=0
            elif field=='changed_storage': phase['native_cache_storage_evidence'][3]['after_sha256']='b'*64
            else: phase['native_idle_bypass_installed']=True
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_decode_phase(phase)

    def test_same_output_does_not_hide_bad_dflash_tail(self):
        for field in ('feature','cache','positions','update','placeholder'):
            plan,ids,ref,candidate,record=fixtures.NumericalGateTests().fixture()
            tail=candidate['dflash_tail']
            if field=='feature': tail['qualification_fingerprints']['target_features']['sha256']='a'*64
            elif field=='cache': tail['qualification_fingerprints']['valid_main_swa'][0]['value']['sha256']='a'*64
            elif field=='positions': tail['retained_positions'].pop()
            elif field=='update': tail['native_main_cache_update_calls']=0
            else: tail['sampler_placeholder_features_used']=True
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_sparse_pair(plan,ids,ref,candidate,record,record)

    def test_qualification_closes_actual_candidate_without_extra_formal_request(self):
        ids=list(range(300));doc=fixtures.document(ids,0,300)
        runtime=NativeRuntime(dict(sha256='a'*64),'s2',Path('/unused'),0)
        runtime.scope=fixtures.SCOPE;runtime.bank_budget=MAX_BANK_BYTES;runtime.allow_sync=contextlib.nullcontext
        runtime.driver=NS(gather=lambda value:[value]+[None]*7,barrier=lambda:None)
        state=dict(resident_bank_bytes=1,resident_document_ids=[doc['document_id']])
        runtime.controller=NS(resident_bank_state=lambda:dict(state),begin_request=lambda *a,**k:[],
            prepare_case=lambda **kw:None)
        runtime.layers=[NS(self_attn=NS(q_a_kv_swa_proj=NS(weight=NS(device='cpu'))))]
        runtime.publish=lambda name,value:dict(path='/unused/'+name,sha256='a'*64)
        calls=[];formal=[]
        def internal(request,phase,mode,**kwargs):
            calls.append((mode,request.max_tokens,kwargs))
            return dict(controller={},record={'output_ids':[2]*32},sha256='a'*64)
        runtime._internal=internal
        runtime._configure=lambda *a,**k:formal.append((a,k))
        request=TokenRequest('qualification',tuple(ids),32,'s2',0,documents=(doc,))
        with patch('redknot.gate.verify_sparse_pair',return_value={'passed':True}):
            result=runtime.prepare(request,qualification_only=True)
        self.assertEqual([c[0] for c in calls],['reference_sparse','sensitive'])
        self.assertEqual(formal,[])
        self.assertTrue(result['redundant_formal_qualification_request_omitted'])
        self.assertIsNone(runtime.current_request)
        self.assertIsNotNone(runtime.last_qualification)


if __name__=='__main__': unittest.main()
